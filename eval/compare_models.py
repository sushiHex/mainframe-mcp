"""Run a frozen retrieval comparison in scratch state, without changing defaults.

Inputs and outputs belong outside Git. See eval/README.md for the manifest
format. Experimental model adapters use the production Models factory hooks;
chunking, storage, ranking, and result shaping remain the real v2 code paths.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trial_guard import TrialAborted, open_gate, write_json


def trial_call(operation, *args, **kwargs):
    """Keep trial failures outside production recovery and result fallbacks."""
    try:
        return operation(*args, **kwargs)
    except Exception as error:
        raise TrialAborted(f'forward failed: {type(error).__name__}: {error}') from error


def index_identity(config, corpus_hash, fingerprint, native, profile):
    return {'schema_version': 2, 'corpus_sha256': corpus_hash, 'embedder': config['embedder'],
            'fingerprint': fingerprint, 'encoding': {
                'native': native, 'profile': profile,
                'dtype': 'bfloat16' if native else 'configured', 'normalize': True}}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def corpus_digest(manifest):
    """Questions do not change vectors; document IDs also determine lane metadata."""
    from mainframe.core.paths import canonical

    files = sorted((f['id'], canonical(f['path']), f['sha256']) for f in manifest['files'])
    return digest(json.dumps(files, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def reuse_index(index, identity, original_manifest=None):
    """Accept only a completed, matching index; never rewrite historical evidence."""
    if not index.exists():
        return False
    marker = index / 'comparison.json'
    if not marker.exists():
        raise ValueError('incomplete index; choose a new scratch index')
    recorded = json.loads(marker.read_text(encoding='utf-8'))
    if recorded == identity:
        return True
    if original_manifest is not None:
        manifest, manifest_hash = load_manifest(original_manifest)
        legacy = {k: v for k, v in identity.items() if k not in ('schema_version', 'corpus_sha256')}
        legacy['manifest_sha256'] = manifest_hash
        if recorded == legacy and corpus_digest(manifest) == identity['corpus_sha256']:
            return True
    raise ValueError('index belongs to different inputs; use its original --index-manifest '
                     'for a legacy index, or choose a new scratch index')


def load_manifest(path: Path) -> tuple[dict, str]:
    from mainframe.core.paths import canonical

    raw = path.read_bytes()
    manifest = json.loads(raw)
    ids = set()
    paths = set()
    for item in manifest["files"]:
        if item["id"] in ids:
            raise ValueError(f"duplicate document ID: {item['id']}")
        ids.add(item["id"])
        identity = canonical(item['path'])
        if identity in paths:
            raise ValueError(f"duplicate document path: {item['id']}")
        paths.add(identity)
        if digest(Path(item["path"]).read_bytes()) != item["sha256"]:
            raise ValueError(f"corpus changed: {item['id']}")
    query_ids = set()
    for query in manifest["queries"]:
        if query["id"] in query_ids:
            raise ValueError(f"duplicate query ID: {query['id']}")
        query_ids.add(query["id"])
        if not query["relevant"] or not set(query["relevant"]).issubset(ids):
            raise ValueError(f"invalid relevance labels: {query['id']}")
    if not ids or not query_ids:
        raise ValueError("corpus and queries must be nonempty")
    return manifest, digest(raw)


def ranking_metrics(query: dict, results: list[dict], file_ids: dict) -> dict:
    """Score returned chunk slots; repeated documents never earn extra credit.

    A duplicate still occupies its actual rank. This measures the API's top-3
    chunk output, not a deduplicated top-3 document retrieval leaderboard.
    Optional expected_text also measures whether the answer passage was found.
    """
    relevant = set(query["relevant"])
    seen = set()
    gains = []
    passage = []
    expected = " ".join(query.get("expected_text", "").lower().split())
    for row in results[:3]:
        doc = file_ids[row["file"]]
        gains.append(int(doc in relevant and doc not in seen))
        seen.add(doc)
        text = " ".join(row.get("text", "").lower().split())
        passage.append(int(doc in relevant and (not expected or expected in text)))
    first = next((i + 1 for i, gain in enumerate(gains) if gain), None)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(3, len(relevant))))
    return {"hit1": float(bool(gains and gains[0])), "hit3": float(any(gains)),
            "mrr3": 1 / first if first else 0.0,
            "ndcg3": sum(g / math.log2(i + 2) for i, g in enumerate(gains)) / ideal,
            "passage1": float(bool(passage and passage[0])),
            "passage3": float(any(passage))}


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation, matching numpy's default percentile convention."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = math.floor(pos)
    return ordered[lo] + (ordered[math.ceil(pos)] - ordered[lo]) * (pos - lo)


def experimental_factories(native_embedder: bool, bf16_reranker: bool, *, query_prompt=None, gate=None,
                           trust_remote_code=False):
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from mainframe.core.embedder import Embedder
    from mainframe.core.reranker import Reranker

    class NativeEmbedder(Embedder):
        """Honor model-owned prompts, attention, pooling, and normalization."""

        def _load_fp(self, model_name, cache_dir, device):
            if gate is not None:
                gate.wait()
            self.model = SentenceTransformer(
                model_name, cache_folder=str(cache_dir), device=device,
                trust_remote_code=trust_remote_code,
                model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})
            self.dimension = self.model.get_sentence_embedding_dimension()
            if gate is not None:
                # ST invokes its root forward directly; ST 5.4 also invokes the
                # HF forward directly. Its enclosing Transformer module is the
                # stable __call__ boundary enclosing actual GPU work in both.
                transformers = [m for m in self.model.modules() if hasattr(m, 'auto_model')]
                if len(transformers) != 1:
                    raise ValueError('guarded native encoding requires one transformer')
                gate.attach(transformers[0])

        def _embed_at(self, texts, bs):
            with torch.inference_mode():
                return self.model.encode_document(
                    texts, batch_size=bs, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_tensor=True).float().cpu().tolist()

        def embed_query(self, query):
            if gate is not None:
                gate.wait()
            with torch.inference_mode():
                return self.model.encode_query(
                    [query], show_progress_bar=False, normalize_embeddings=True,
                    convert_to_tensor=True,
                    **({'prompt_name': query_prompt} if query_prompt else {})).float().cpu().tolist()[0]

    class BF16Reranker(Reranker):
        """Use the trained CrossEncoder modules with the production rerank path."""

        def __init__(self, config):
            cfg = config["reranker"]
            self.model_name = cfg["model"]
            self.enabled = True
            self.heading_inject = cfg.get("heading_inject", False)
            self.default_top_k = config["search"]["rerank_top_k"]
            self._backend = "cross-encoder"
            self._quantized = False
            self.model = CrossEncoder(
                self.model_name, device="cuda", max_length=2048, trust_remote_code=False,
                model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "sdpa"})

    embedder = NativeEmbedder if native_embedder else Embedder
    reranker = BF16Reranker if bf16_reranker else Reranker
    if gate is None:
        return embedder if native_embedder else None, reranker if bf16_reranker else None

    class PacedReranker(reranker):
        def __init__(self, config):
            from mainframe.core.reranker import detect_backend
            cfg = config['reranker']
            if bf16_reranker or detect_backend(cfg['model'], cfg.get('backend')) != 'qwen3-logit':
                raise ValueError('paced retrieval currently requires the native Qwen3 reranker')
            gate.wait()
            trial_call(super().__init__, config)
            # Qwen's production path calls the causal LM through __call__.
            # Convert errors before its OOM handler can change INT8 batches.
            from functools import wraps
            forward = self.model.forward

            @wraps(forward)
            def checked_forward(*args, **kwargs):
                return trial_call(forward, *args, **kwargs)

            self.model.forward = checked_forward
            gate.attach(self.model)

    class PacedEmbedder(embedder):
        def __init__(self, config):
            gate.wait()
            trial_call(super().__init__, config)
            if not native_embedder:
                transformers = [m for m in self.model.modules() if hasattr(m, 'auto_model')]
                if len(transformers) != 1:
                    raise TrialAborted('paced encoding requires one transformer')
                gate.attach(transformers[0])

        def _load_manual_quantized(self, *args):
            raise TrialAborted('trial model loading failed; fallback prohibited')

        def embed(self, texts, batch_size=None):
            gate.wait()
            if not texts:
                return []
            bs = min(batch_size or self.batch_size, self._safe_batch)
            return trial_call(self._embed_at, texts, bs)

    return PacedEmbedder, PacedReranker


def run(args):
    # Validate immutable inputs and output isolation before initializing CUDA.
    import subprocess

    manifest, manifest_hash = load_manifest(args.manifest)
    for path in (args.output, args.index):
        parent = path.resolve()
        while not parent.exists():
            parent = parent.parent
        probe = subprocess.run(["git", "-C", str(parent), "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            raise ValueError("comparison output and indexes must be outside Git worktrees")
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.native_embedder and config["embedder"].get("quantize", True):
        raise ValueError("the native BF16 adapter requires embedder.quantize=false")
    config["paths"]["mainframe_dir"] = str(args.output)
    gate = None
    profile = None
    if args.permit:
        gate, settings = open_gate(args.permit, args.nonce, stdin=args.permit_stdin)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    if args.candidate:
        from embedding_probe import validate_snapshot
        profile = validate_snapshot(args.candidate, Path(config['embedder']['model']))
        if not args.native_embedder:
            raise ValueError('candidate profiles require native encoding')

    import torch
    from mainframe.core.indexer import DocFailure, LaneFile, prepare_document
    from mainframe.core.models import Models
    from mainframe.core.paths import canonical
    from mainframe.core.search import SearchService
    from mainframe.core.store import Store

    fp = Store.fingerprint_from_config(config)
    corpus_hash = corpus_digest(manifest)
    identity = index_identity(config, corpus_hash, fp, args.native_embedder, profile)
    marker = args.index / 'comparison.json'
    existing = reuse_index(args.index, identity, args.index_manifest)
    if not torch.cuda.is_available():
        raise RuntimeError("this comparison requires CUDA")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    if gate:
        gate.configure_cuda(torch)
    factories = experimental_factories(args.native_embedder, args.bf16_reranker,
                                       query_prompt=profile['query_prompt'] if profile else None, gate=gate,
                                       trust_remote_code=bool(profile and profile.get('reviewed_code')))

    class TrialModels(Models):
        def invoke(self, role, operation):
            if gate:
                gate.wait()
                return super().invoke(role, lambda model: trial_call(operation, model))
            return super().invoke(role, operation)

    models = TrialModels(config, embedder_factory=factories[0], reranker_factory=factories[1])
    report = {"schema_version": 1, "label": args.label, "manifest_sha256": manifest_hash,
              "corpus_sha256": corpus_hash,
              "config": config, "native_embedder": args.native_embedder,
              "embedding_implementation": ("experimental-native" if args.native_embedder else
                                           "production-" + config['embedder'].get('encoding', 'legacy')),
              "bf16_reranker": args.bf16_reranker, "queries": [], "complete": False}
    report.update(guard_nonce=args.nonce, candidate=profile, memory_policy='observe',
                  responsiveness='unvalidated', versions={name: importlib.metadata.version(name) for name in
                  ('torch', 'sentence-transformers', 'transformers', 'bitsandbytes', 'lancedb')},
                  source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])): digest(p.read_bytes())
                  for p in [Path(__file__).resolve(), Path(__file__).with_name('trial_guard.py').resolve(),
                            *sorted((Path(__file__).resolve().parents[1] / 'src' / 'mainframe' / 'core').glob('*.py'))]})

    def save():
        write_json(args.output / 'report.json', report)

    def timing():
        return {'forward_s': gate.forward_seconds if gate else 0.,
                'pacing_s': gate.idle_seconds if gate else 0.,
                'monitor_wait_s': gate.monitor_wait_seconds if gate else 0.,
                'forwards': gate.forwards if gate else 0}

    def elapsed_timing(before):
        return {key: value - before[key] for key, value in timing().items()}

    def memory():
        return {"allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}

    def describe(model):
        obj = getattr(model, "model", getattr(model, "_model", None))
        params = list(obj.parameters())
        devices = sorted({str(p.device) for p in params})
        if any(not d.startswith("cuda") for d in devices):
            raise RuntimeError(f"model offload would invalidate this comparison: {devices}")
        return {**model.info, "parameter_devices": devices,
                "parameter_dtypes": sorted({str(p.dtype) for p in params})}

    try:
        start = time.perf_counter()
        report["embedder"] = models.invoke("embedder", describe)
        report["embedder_load_s"] = time.perf_counter() - start
        print(json.dumps({"stage": "embedder.loaded", "seconds": report["embedder_load_s"]}), flush=True)
        store = Store(args.index, fingerprint=fp, **Store.search_kwargs_from_config(config))
        file_ids = {canonical(f["path"]): f["id"] for f in manifest["files"]}
        start = time.perf_counter()
        before = timing()
        if not existing:
            # Built once, not per document: only the embed call itself needs
            # the model lock (mirrors core/pipeline.py's IndexPipeline._run_batch).
            embed = lambda texts: models.invoke("embedder", lambda e: e.embed(texts))
            batch = []
            for i, f in enumerate(sorted(manifest["files"], key=lambda f: canonical(f["path"]))):
                lf = LaneFile(canonical(f["path"]), "knowledge", f["id"].split("/")[0])
                doc = prepare_document(lf, config["chunker"], embed)
                if isinstance(doc, DocFailure) or not doc.rows:
                    raise RuntimeError(f"document failed or empty: {f['id']}")
                batch.append(doc)
                if len(batch) == 50 or i == len(manifest["files"]) - 1:
                    store.upsert_batch(batch, [])
                    batch.clear()
                    print(json.dumps({"stage": "index", "files": i + 1,
                                      "seconds": time.perf_counter() - start}), flush=True)
            if not store.optimize():
                raise RuntimeError("index optimization failed")
            marker.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
        report["index_s"] = time.perf_counter() - start
        report['index_timing'] = elapsed_timing(before)
        report["index_reused"] = existing
        report["index_health"] = store.index_health()
        report["index_memory"] = memory()
        start = time.perf_counter()
        report["reranker"] = models.invoke("reranker", describe)
        report["reranker_load_s"] = time.perf_counter() - start
        report["load_and_index_memory"] = memory()
        print(json.dumps({"stage": "reranker.loaded", "seconds": report["reranker_load_s"]}), flush=True)
        search = SearchService(models, store, config)
        candidates = {}
        original_search = store.search

        def tracked_search(*a, **kw):
            rows = original_search(*a, **kw)
            candidates["keys"] = [r["chunk_key"] for r in rows]
            candidates["files"] = [file_ids[r["doc_path"]] for r in rows]
            return rows

        store.search = tracked_search
        order = list(manifest["queries"])
        random.Random(20260910).shuffle(order)
        start = time.perf_counter()
        for q in order[:5]:
            search.search(q["query"], limit=3, response_format="full")
        torch.cuda.synchronize()
        report["warmup_s"] = time.perf_counter() - start
        report["warmup_memory"] = memory()
        torch.cuda.reset_peak_memory_stats()
        save()
        for i, q in enumerate(order):
            torch.cuda.synchronize()
            start = time.perf_counter()
            before = timing()
            results = search.search(q["query"], limit=3, response_format="full")["results"]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            query_timing = elapsed_timing(before)
            query_timing['active_s'] = elapsed - query_timing['pacing_s'] - query_timing['monitor_wait_s']
            scored = ranking_metrics(q, results, file_ids)
            scored["candidate_hit"] = float(bool(set(q["relevant"]) & set(candidates["files"])))
            report["queries"].append({"id": q["id"], "set": q["set"], "seconds": elapsed,
                                      **query_timing,
                                      "metrics": scored, "candidate_keys": candidates["keys"],
                                      "results": results})
            if (i + 1) % 10 == 0:
                print(json.dumps({"stage": "search", "queries": i + 1, "last_s": elapsed}), flush=True)
                save()
        report["warm_memory"] = memory()
        report["summary"] = {}
        for group in sorted({q["set"] for q in order}):
            rows = [q for q in report["queries"] if q["set"] == group]
            times = [r["seconds"] for r in rows]
            report["summary"][group] = {
                "queries": len(rows), "p50_s": statistics.median(times), "p95_s": percentile(times, .95),
                'active_p50_s': statistics.median(r['active_s'] for r in rows),
                'active_p95_s': percentile([r['active_s'] for r in rows], .95),
                **{k: statistics.mean(r["metrics"][k] for r in rows) for k in rows[0]["metrics"]}}
        report['timing'] = timing()
        if gate and gate.forwards == 0:
            raise TrialAborted('no paced forwards observed')
        import psutil
        report['host'] = {'peak_process_commit_gib': psutil.Process().memory_info().peak_pagefile / 2**30}
        if gate:
            gate.wait()
            gate.close()
        start = time.perf_counter()
        models.invalidate_all()
        torch.cuda.synchronize()
        report['release_s'] = time.perf_counter() - start
        report['allocated_after_release_gib'] = torch.cuda.memory_allocated() / 2**30
        report["complete"] = True
    except BaseException as e:
        report["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        save()
        if gate:
            gate.close()
        models.invalidate_all()
    print(json.dumps({"complete": True, "summary": report["summary"], "memory": report["warm_memory"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "config", "output", "index"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument('--index-manifest', type=Path,
                        help='original manifest proving the corpus of a legacy completed index')
    parser.add_argument("--label", required=True)
    parser.add_argument("--native-embedder", action="store_true")
    parser.add_argument("--bf16-reranker", action="store_true")
    parser.add_argument('--candidate', choices=['nemotron', 'harrier', 'voyage'])
    parser.add_argument('--permit', type=Path)
    parser.add_argument('--nonce')
    parser.add_argument('--permit-stdin', action='store_true')
    run(parser.parse_args())


if __name__ == "__main__":
    main()
