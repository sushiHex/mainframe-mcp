"""Mainframe MCP Server — GPU-accelerated semantic knowledge base.

Provides: search, ingest_file, list_files, delete_file, curate, status
Runs as stdio MCP server for Claude Code.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mainframe_mcp.chunker import chunk_markdown
from mainframe_mcp.config import load_config
from mainframe_mcp.dedup import content_hash, check_near_dupes
from mainframe_mcp.embedder import Embedder
from mainframe_mcp.manifest import Manifest
from mainframe_mcp.nli import find_contradictions_nn
from mainframe_mcp.reranker import Reranker
from mainframe_mcp.secrets import scrub_secrets
from mainframe_mcp.session_capture import (
    build_frontmatter,
    git_branch_from_repo,
    library_sessions_dir,
    pending_sessions_by_project,
    resolve_sessions_dir,
    session_dirs,
    unique_session_path,
)
from mainframe_mcp.store import SESSION_SOURCE_TYPE, Store, classify_source_type

logger = logging.getLogger(__name__)

# The single heading for the surfaced-contradictions section — used by BOTH the
# writer (consolidate_sessions) and the stripper regex (and the tests), so they
# can't drift. The section is stripped before a note is re-fed so prior flags
# aren't re-NLI'd / re-accreted and the markup isn't fed back to the consolidator.
# Caller-supplied names that become path components (capture_memory /
# consolidate_sessions `project`). No separators, no leading dot — blocks
# ../ traversal out of repos_dir (prompt-injection hardening).
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

CONTRADICTION_HEADING = "## ⚠ Contradictions reconciled (newer supersedes)"

# search(): hard ceiling on returned results (the tool schema documents it).
MAX_SEARCH_RESULTS = 10
# consolidate_sessions(): B-guardrail bloat heuristic (see the call site).
NOTE_BLOAT_FACTOR = 2
NOTE_BLOAT_FLOOR = 2000
# consolidate_sessions(): cap the raw report text packed into ONE consolidation
# run (~28K-token prompt budget at ~3 chars/token, conservatively). Sources
# beyond the budget stay PENDING for the next run instead of being fed to the
# consolidator's tail-truncation and then archived unseen — with days=0
# consolidating the full backlog, an unbounded batch would silently lose the
# truncated-away sessions from the index (review finding, HIGH).
CONSOLIDATION_CHAR_BUDGET = 80_000
# ingest_file's non-error status protocol, consumed by _ingest_paths' counters.
# Grown by literal-editing twice in three commits — keep it canonical here, and
# _ingest_paths warns on anything unclassified instead of silently dropping it.
_SKIP_STATUSES = frozenset({"unchanged", "all_chunks_duplicate", "empty"})
_CONTRA_SECTION = re.compile(r"\n*" + re.escape(CONTRADICTION_HEADING) + r"\n.*?(?=\n##\s|\Z)", re.DOTALL)
# Memory-note scaffolding stripped when re-feeding a note: the "# title" + the
# "_Consolidated …_" preamble, and the trailing "## Sources" provenance block.
_NOTE_PREAMBLE = re.compile(r"\A\s*#[^\n]*\n+_Consolidated[^\n]*_\s*")
_SOURCES_SECTION = re.compile(r"\n##\s+Sources\b.*\Z", re.DOTALL)


def _canonical_body(note_text: str) -> str:
    """Extract just the curated knowledge body of a memory note — drop the title +
    '_Consolidated…_' preamble, the ⚠ section, and the ## Sources block. Feeding
    only the body to re-consolidation stops the title/preamble/Sources from
    accreting each round and keeps provenance filenames out of the NLI fact pool."""
    body = _CONTRA_SECTION.sub("", note_text)
    body = _NOTE_PREAMBLE.sub("", body)
    body = _SOURCES_SECTION.sub("", body)
    return body.strip()


class MainframeMCP:
    """Core Mainframe logic, wrapping all components."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.mainframe_dir = Path(self.config["paths"]["mainframe_dir"])
        self.mainframe_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Initializing Mainframe MCP...")

        # Load GPU models (stay resident)
        self.embedder = Embedder(self.config)
        self.reranker = Reranker(self.config)

        # Storage
        self.store = Store(
            db_path=self.mainframe_dir / ".lancedb",
            embedding_dim=self.embedder.dimension,
            **Store.search_kwargs_from_config(self.config),
        )
        self.manifest = Manifest(self.mainframe_dir)

        logger.info("Mainframe MCP ready.")

    def ingest_file(self, file_path: str, donor_cache=None) -> dict:
        """Ingest a markdown file into the Mainframe.

        `donor_cache` (dedup.DonorCache, batch ingests only) replaces the
        per-file full-table hash scan with an incrementally-maintained index —
        an 88-file sync otherwise scans 37K rows 88 times."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        if not path.suffix.lower() in (".md", ".txt"):
            return {"error": f"Unsupported file type: {path.suffix}"}

        text = path.read_text(encoding="utf-8")
        file_hash = content_hash(text)

        # Check if file has changed (change detection uses the RAW file hash)
        if not self.manifest.needs_reindex(file_path, file_hash):
            return {"status": "unchanged", "file": file_path}

        # Determine source type from path (single shared classifier)
        source_type = classify_source_type(str(path), self.config["paths"]["mainframe_dir"])

        # Choke-point secret scrub: every session-typed chunk is scrubbed before
        # indexing regardless of which write path produced the file (defense in
        # depth alongside capture_memory/hook scrubbing and the gitignore).
        if source_type == SESSION_SOURCE_TYPE:
            text = scrub_secrets(text)

        # Chunk
        chunk_cfg = self.config["chunker"]
        chunks = chunk_markdown(
            text,
            max_tokens=chunk_cfg.get("chunk_size", 256),
            overlap_ratio=chunk_cfg.get("overlap_ratio", 0.35),
        )
        if not chunks:
            # Empty/whitespace files are SKIPS, not failures — 4 stub files
            # polluted the first real sync's error count (2026-07-16).
            return {"status": "empty", "file": file_path,
                    "chunks": 0, "source_type": source_type}

        # API contextual retrieval (opt-in, ingest-time only): situate each chunk
        # within its document and prepend the context BEFORE hashing/embedding/FTS
        # so dedup + change detection see the contextualized text. Failures leave
        # chunks plain; the context is scrubbed like any server-authored text.
        ctxr = self._ensure_contextualizer()
        if ctxr.enabled:
            contexts = ctxr.contextualize(text, [c.text for c in chunks])
            for c, cx in zip(chunks, contexts):
                if cx:
                    c.text = f"{scrub_secrets(cx)}\n\n{c.text}"

        # Check for exact dupes. Exclude session chunks from the donor set (a
        # low-signal session chunk can't evict an identical curated chunk, plan
        # §6/M1) AND exclude this file's own prior chunks (so re-ingesting an
        # edited file doesn't see its unchanged sections as dupes-of-themselves
        # and then lose them to delete_by_doc — review finding #1).
        if donor_cache is None:
            existing_hashes = self.store.get_existing_hashes(
                exclude_source_type=SESSION_SOURCE_TYPE, exclude_doc_path=file_path
            )
        chunk_dicts = []
        chunk_hashes = []
        for c in chunks:
            h = content_hash(c.text)
            dupe = (donor_cache.is_donor(h, file_path) if donor_cache is not None
                    else h in existing_hashes)
            if dupe:
                logger.debug(f"Skipping exact dupe chunk: {c.heading}/{c.chunk_index}")
                continue
            chunk_dicts.append({
                "text": c.text,
                "heading": c.heading,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "token_count": c.token_count,
            })
            chunk_hashes.append(h)

        if not chunk_dicts:
            return {"status": "all_chunks_duplicate", "file": file_path,
                    "chunks": 0, "source_type": source_type}

        # Delete old chunks for this file (re-ingest). Cache first: a crash
        # between delete and insert then leaves the donor cache conservative
        # (doc absent) rather than stale (doc's dead hashes still listed).
        if donor_cache is not None:
            donor_cache.remove_doc(file_path)
        self.store.delete_by_doc(file_path)

        # Embed
        texts = [c["text"] for c in chunk_dicts]
        embeddings = self.embedder.embed(texts)

        # Near-dupe check (warn but don't reject). Build the donor matrix once.
        existing_embs, existing_ids = self.store.get_embeddings_for_dedup(limit=500)
        existing_mat = np.asarray(existing_embs) if existing_embs else existing_embs
        near_dupes = []
        for emb in embeddings[:3]:  # check first 3 chunks only (performance)
            dupes = check_near_dupes(emb, existing_mat, existing_ids)
            near_dupes.extend(dupes)

        # Insert. authored_at = file mtime: the recency signal that survives
        # re-ingest (created_at is ingest time and resets on every sync).
        try:
            authored_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        except OSError:
            authored_at = ""
        doc_id = hashlib.md5(file_path.encode()).hexdigest()[:12]
        count = self.store.insert_chunks(
            chunks=chunk_dicts,
            doc_id=doc_id,
            doc_path=file_path,
            source_type=source_type,
            embeddings=embeddings,
            content_hashes=chunk_hashes,
            authored_at=authored_at,
        )

        if donor_cache is not None:
            donor_cache.add_doc(file_path, set(chunk_hashes),
                                is_donor=source_type != SESSION_SOURCE_TYPE)

        # Update manifest
        self.manifest.record_ingest(file_path, file_hash, count)

        result = {
            "status": "ingested",
            "file": file_path,
            "chunks": count,
            "source_type": source_type,
        }
        if near_dupes:
            result["near_duplicates"] = near_dupes[:5]
        return result

    # --- Ambient capture (Phase 1) ---

    @staticmethod
    def _slugify(text: str, max_words: int = 8, max_len: int = 48) -> str:
        words = re.findall(r"[A-Za-z0-9]+", text.lower())[:max_words]
        return "-".join(words)[:max_len].strip("-")

    def capture_memory(
        self,
        content: str,
        title: str,
        project: str | None = None,
        kind: str = "session",
        session_id: str = "",
        cwd: str = "",
    ) -> dict:
        """Write one immutable session-memory markdown file, then ingest it.

        Files are the source of truth: compose templated markdown, write a
        uniquely-named file under research/sessions/ (or the library fallback),
        then reuse ingest_file verbatim. Never overwrites an existing capture —
        overwriting re-triggers the verified append/dedup data-loss bug.
        Returns ingest_file's {status, file, chunks, source_type}.

        `cwd` is the AGENT's working directory used to infer the repo when
        `project` is omitted. It must be passed by the caller — Path.cwd() here
        is the long-lived MCP server process's cwd (fixed at launch), NOT the
        agent's project, so relying on it silently mis-files captures (finding
        #3). Falls back to Path.cwd() only if neither project nor cwd is given.
        """
        if project and not _SAFE_NAME_RE.match(project):
            return {"status": "error", "error": f"invalid project name: {project!r}"}
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        hhmmss = now.strftime("%H%M%S")
        # Sanitize before it becomes a filename component (a raw "../.." here
        # would escape the sessions lane).
        sid = re.sub(r"[^A-Za-z0-9-]", "", session_id or "")[:8] or "adhoc"
        eff_cwd = cwd or str(Path.cwd())

        sessions_dir, repo_dir = resolve_sessions_dir(
            eff_cwd,
            self.config["paths"]["repos_dir"],
            self.config["paths"]["mainframe_dir"],
            project=project,
        )
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Secret-scrub the agent-authored text before it touches disk.
        safe_content = scrub_secrets(content or "")
        safe_title = scrub_secrets(title or "").strip() or "untitled"

        # Unique filename — one immutable file per capture event; never overwrite.
        slug = self._slugify(safe_title)
        chash = content_hash(safe_content)[:6]
        base = f"{date}-{hhmmss}-{sid}" + (f"-{slug}" if slug else f"-{chash}")
        path = unique_session_path(sessions_dir, base)

        # Compose templated markdown (heading-aware -> each section a chunk).
        proj_label = project or (repo_dir.name if repo_dir else "adhoc")
        frontmatter = build_frontmatter(
            kind, proj_label, sid, date, hhmmss, eff_cwd,
            git_branch_from_repo(repo_dir), "capture_memory",
        )
        h1 = f"# Session {date} — {safe_title}" if kind == "session" else f"# {safe_title}"
        md = f"{frontmatter}{h1}\n\n{safe_content.strip()}\n"

        path.write_text(md, encoding="utf-8")
        return self.ingest_file(str(path))

    def search(self, query: str, top_k: int = 3, include_sessions: bool = False) -> list[dict]:
        """Search the Mainframe with hybrid search + reranking.

        `top_k` is the RESULT count (honest cap, ceiling MAX_SEARCH_RESULTS); the
        candidate pool fed to the reranker is independently floored by
        search.candidate_pool so a small limit never starves the cross-encoder
        (10-POV review, pool starvation)."""
        # Clamp the caller-supplied limit BEFORE it sizes anything: the JSON
        # schema's "max 10" is advisory text, so limit=100000 would otherwise
        # size the LanceDB fetch (pool) to the whole corpus. int() also guards
        # non-int JSON numbers.
        top_k = max(1, min(int(top_k), MAX_SEARCH_RESULTS))
        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Initial retrieval (over-fetch for reranking, hybrid vector+FTS).
        # Default-exclude session captures so accreted, low-signal session chunks
        # never crowd curated knowledge out of the candidate set (plan §4.2 — this
        # filter, not the near-inert tier boost, is the real precision defense).
        fetch_mult = self.config["search"].get("fetch_multiplier", 2)
        pool = max(self.config["search"].get("candidate_pool", 20), top_k * fetch_mult)
        exclude = None if include_sessions else SESSION_SOURCE_TYPE
        raw_results = self.store.search(
            query_embedding,
            top_k=pool,
            query_text=query,
            exclude_source_type=exclude,
        )

        if not raw_results:
            return []

        # Rerank (pass headings for context injection); top_k already clamped.
        # The category is passed unconditionally — the Reranker owns the
        # category_instructions gate (off by default: measured 0.3760 vs 0.4187,
        # 2026-07-02, the 4th model-card-vs-corpus inversion).
        texts = [r["text"] for r in raw_results]
        headings = [r.get("heading", "") for r in raw_results]
        reranked = self.reranker.rerank(query, texts, top_k=top_k, headings=headings,
                                        category=self.embedder._classify_query(query))

        # Build final results
        results = []
        for orig_idx, score in reranked:
            r = raw_results[orig_idx]
            full_text = r["text"]
            results.append({
                "file": r["doc_path"],
                "heading": r["heading"],
                "text": full_text[:500],
                "truncated": len(full_text) > 500,
                # exact source span so an agent can Read the full section to cite
                "char_start": int(r["char_start"]),
                "char_end": int(r["char_end"]),
                "score": float(r.get("boosted_score", r.get("_distance", 0))),
                "rerank_score": float(score),
                "source_type": r["source_type"],
                "tier": int(r["tier"]),
            })
        # Add 1-based line_start/line_end so line-oriented read_file tools (most
        # MCP hosts) can pull the evidence directly — char offsets alone don't
        # map. Each unique file is read once; a vanished/changed file just omits
        # the line fields (char offsets remain).
        self._add_line_spans(results)

        # Retrieval analytics stay OFF the hot path: the old per-query LanceDB
        # table.update was the primary fragmentation driver (666 fragments /
        # 1349 versions live) and retrieval_count is never used for ranking.
        # Manifest counts accumulate in memory and flush every 25th search.
        for r in results:
            self.manifest.bump_retrieval(r["file"])
        self._search_count = getattr(self, "_search_count", 0) + 1
        if self._search_count % 25 == 0:
            self.manifest.save()

        return results

    @staticmethod
    def _add_line_spans(results: list[dict]) -> None:
        """Annotate each result with 1-based line_start/line_end from its char
        span, reading each unique file at most once. Best-effort: an unreadable
        file (moved/deleted/binary) leaves the result's line fields absent —
        never raises, so a stale index can't break search."""
        cache: dict[str, str | None] = {}
        for r in results:
            f = r["file"]
            if f not in cache:
                try:
                    cache[f] = Path(f).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    cache[f] = None
            text = cache[f]
            if text is None:
                continue
            r["line_start"] = text.count("\n", 0, r["char_start"]) + 1
            r["line_end"] = text.count("\n", 0, r["char_end"]) + 1

    def list_files(self) -> dict:
        """List all files in the Mainframe with stats."""
        stats = self.store.get_stats()
        manifest_stats = self.manifest.get_stats()
        return {**stats, **manifest_stats}

    def delete_file(self, file_path: str) -> dict:
        """Delete a file and its chunks from the index (keeps the manifest in
        sync — otherwise a re-created identical file is seen as 'unchanged' and
        silently never re-indexed, plan §4.3)."""
        self.delete_files([file_path])
        return {"status": "deleted", "file": file_path}

    def delete_files(self, file_paths: list[str]) -> int:
        """Batch-delete: drop each file's chunks, then ONE manifest save.
        Shared by delete_file / sync_index / consolidate_sessions so all prune
        the same (manifest-safe) way without N per-file disk writes. Only the
        files whose store delete actually succeeded are removed from the manifest
        — so a mid-loop failure never leaves store and manifest divergent."""
        removed = []
        for fp in file_paths:
            try:
                self.store.delete_by_doc(fp)
                removed.append(fp)
            except Exception as e:
                logger.warning(f"delete_by_doc failed for {fp}: {e}")
        if removed:
            self.manifest.remove_many(removed)
        return len(removed)

    def _ingest_paths(self, paths: list[str]) -> tuple[int, int, list[dict]]:
        """Ingest each path via ingest_file; return (ingested, skipped, errors).
        Shared by ingest_projects and sync_index."""
        from mainframe_mcp.dedup import DonorCache
        donor_cache = DonorCache(
            self.store.get_hash_doc_map(exclude_source_type=SESSION_SOURCE_TYPE)
        ) if paths else None
        ingested = skipped = 0
        errors = []
        for fp in paths:
            try:
                result = self.ingest_file(fp, donor_cache=donor_cache)
            except Exception as e:
                # One poison file (CUDA fault, LanceDB error) must not abort the
                # whole batch — the historical "rebuild killed repeatedly" pain.
                logger.warning(f"ingest failed for {fp}: {e}")
                errors.append({"file": fp, "error": str(e)})
                continue
            status = result.get("status")
            if status == "ingested":
                ingested += 1
            elif status in _SKIP_STATUSES:
                skipped += 1
            elif result.get("error"):
                errors.append({"file": fp, "error": result["error"]})
            else:
                logger.warning(f"unclassified ingest status {status!r} for {fp}")
                errors.append({"file": fp, "error": f"unclassified status {status!r}"})
        return ingested, skipped, errors

    def ingest_projects(self) -> dict:
        """Scan all projects for research files and CLAUDE.md, ingest new/changed ones."""
        from mainframe_mcp.scanner import scan_projects

        files = scan_projects(config=self.config)
        ingested, skipped, errors = self._ingest_paths([f["path"] for f in files])
        if ingested:
            self.store.optimize()  # compaction + FTS freshness after the batch
        return {
            "scanned": len(files),
            "ingested": ingested,
            "skipped": skipped,
            "errors": len(errors),
            "error_details": errors[:10],
        }

    def _all_known_files(self) -> list[str]:
        """The full set of files the index should hold: scanned project
        knowledge (incl. research/sessions/) plus the library/sessions fallback
        lane that the repo scanner doesn't reach."""
        from mainframe_mcp.scanner import scan_projects

        files = [f["path"] for f in scan_projects(config=self.config)]
        lib_sessions = library_sessions_dir(self.config["paths"]["mainframe_dir"])
        if lib_sessions.exists():
            files += [str(p) for p in sorted(lib_sessions.glob("*.md"))]
        return files

    def sync_index(self) -> dict:
        """Phase 2: one-shot incremental sync. Ingests new/modified knowledge
        files AND prunes deleted ones (the deletion pruning ingest_projects
        lacked). Single-process, 0 extra VRAM. Run this from the live server —
        never concurrently with another writer (the SessionStart hook only
        nudges; it does not ingest)."""
        from mainframe_mcp.watcher import diff_files

        files = self._all_known_files()
        diff = diff_files(files, self.manifest)

        ingested, _, errors = self._ingest_paths(diff["new"] + diff["modified"])
        self.delete_files(diff["deleted"])  # prune (chunks + manifest, one save)
        if ingested or diff["deleted"]:
            self.store.optimize()  # compaction + FTS freshness after the batch

        # Memory-loop signal: surface the per-project consolidation backlog so
        # the agent (not a daemon) decides when to consolidate — files-as-truth.
        pending = pending_sessions_by_project(
            self.config["paths"]["repos_dir"], self.config["paths"]["mainframe_dir"])
        threshold = self.config.get("memory", {}).get("consolidate_threshold", 5)
        ripe = sorted(p for p, n in pending.items() if n >= threshold)
        hint = None
        if ripe:
            hint = (f"{', '.join(ripe)} have {threshold}+ pending session captures — "
                    f"consider consolidate_sessions(project=...) to merge them into "
                    f"curated memory notes")

        return {
            "scanned": len(files),
            "ingested": ingested,
            "pruned": len(diff["deleted"]),
            "new": len(diff["new"]),
            "modified": len(diff["modified"]),
            "pending_sessions": pending,
            "consolidation_hint": hint,
            "unchanged": len(diff["unchanged"]),
            "errors": len(errors),
            "error_details": errors[:10],
        }

    def _ensure_consolidator(self, caller: str):
        """Lazy-load the Qwen2.5-3B consolidator once; stays resident."""
        if not hasattr(self, "consolidator"):
            from mainframe_mcp.consolidator import Consolidator
            logger.info(f"Loading consolidator for {caller}...")
            self.consolidator = Consolidator(self.config)

    def _ensure_nli(self):
        """Lazy-load the DeBERTa contradiction detector once; stays resident."""
        if not hasattr(self, "nli"):
            from mainframe_mcp.nli import ContradictionDetector
            logger.info("Loading NLI for contradiction surfacing...")
            self.nli = ContradictionDetector(self.config)

    def _ensure_contextualizer(self):
        """Lazy-init the (API, opt-in) chunk contextualizer once — cheap, no GPU;
        a fake can be injected by assigning self.contextualizer before ingest."""
        if not hasattr(self, "contextualizer"):
            from mainframe_mcp.contextualizer import Contextualizer
            self.contextualizer = Contextualizer(self.config)
        return self.contextualizer

    def _detect_contradictions(self, existing: str, reports: list[dict]) -> tuple[list[dict], str]:
        """C (mem0 update-phase, files-as-truth slice): surface facts in the NEW
        reports that contradict the EXISTING canonical note (the NN-gated NLI
        mechanics live in nli.find_contradictions_nn). Returns (pairs, status)
        where status is "ran" / "disabled" / "error" — so an empty list is
        distinguishable from "didn't run". Non-critical — any failure degrades to
        ([], "error") and must NEVER abort the consolidation pipeline."""
        try:
            self._ensure_nli()
        except Exception as e:
            logger.warning(f"NLI failed to load; skipping contradiction surfacing: {e}")
            return [], "error"
        if not getattr(self.nli, "enabled", False):
            return [], "disabled"
        try:
            reports_text = "\n".join(r["text"] for r in reports)
            return find_contradictions_nn(self.nli, self.embedder, existing, reports_text), "ran"
        except Exception as e:
            logger.warning(f"Contradiction detection failed; skipping: {e}")
            return [], "error"

    def build_raptor(self, distance_threshold: float = 0.8, min_cluster_size: int = 3) -> dict:
        """Build RAPTOR tier-1 summaries from tier-0 chunks."""
        from mainframe_mcp.raptor import build_raptor_summaries

        self._ensure_consolidator("RAPTOR")
        return build_raptor_summaries(
            store=self.store,
            embedder=self.embedder,
            consolidator=self.consolidator,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
        )

    def _recent_session_files(self, project: str | None, days: int) -> list[Path]:
        """Session capture files (project's lane, or all + library fallback)
        modified within `days`; days<=0 means ALL pending sessions — a fixed
        window would strand anything older than it forever (10-POV review:
        unconsolidated sessions accumulate unbounded and are never pruned)."""
        days = min(int(days), 36500)  # clamp; timedelta overflows on huge values
        cutoff = None if days <= 0 else (datetime.now() - timedelta(days=days)).timestamp()
        dirs = session_dirs(
            self.config["paths"]["repos_dir"],
            self.config["paths"]["mainframe_dir"],
            project,
        )
        out = []
        for sd in dirs:
            if not sd.exists():
                continue
            for f in sorted(sd.glob("*.md")):
                try:
                    if cutoff is None or f.stat().st_mtime >= cutoff:
                        out.append(f)
                except OSError:
                    continue  # vanished between glob and stat
        return out

    def consolidate_sessions(
        self,
        project: str | None = None,
        topic: str = "",
        days: int = 0,
        archive_sources: bool = True,
    ) -> dict:
        """Phase 3: merge recent raw session captures into ONE curated, git-tracked
        research/memory/<topic>.md note (local Qwen2.5-3B), index it, then prune
        the raw logs from the index and archive (or delete) the files. Bounds
        index growth so accreted sessions don't crowd out curated knowledge."""
        if project and not _SAFE_NAME_RE.match(project):
            return {"status": "error", "error": f"invalid project name: {project!r}"}
        # Memory-loop V1: topic defaults to the project name — one curated note
        # per project, zero clustering. (Topic-level notes remain available by
        # passing an explicit topic.)
        if not topic and project:
            topic = project
        if not topic:
            return {"status": "error",
                    "error": "topic required when no project is given (topic defaults to the project name)"}
        sources = self._recent_session_files(project, days)
        if not sources:
            return {"status": "no_sessions", "sources": 0}

        self._ensure_consolidator("consolidate_sessions")

        # Read defensively — one unreadable/vanished/non-UTF-8 file shouldn't
        # abort the whole consolidation. Keep only the files we actually read.
        # Greedy-pack sources into the consolidation budget: everything past it
        # stays PENDING (not archived, not pruned, picked up by the next run) —
        # otherwise the consolidator would tail-truncate the overflow out of the
        # merge while we archive it as if it had been consolidated (silent
        # knowledge loss at days=0 backlog scale).
        reports = []
        pending = unreadable = truncated_sources = 0
        budget = CONSOLIDATION_CHAR_BUDGET
        for f in sources:
            if budget <= 0:
                pending += 1
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"consolidate: skipping unreadable {f}: {e}")
                unreadable += 1
                continue
            if len(text) > budget:
                # A single file can exceed the whole budget; leaving it pending
                # would deadlock it forever, and admitting it whole re-opens the
                # consolidator's silent tail-truncation. Clamp it VISIBLY.
                logger.warning(f"consolidate: {f.name} exceeds the budget — clamping "
                               f"{len(text)} -> {budget} chars")
                text = text[:budget]
                truncated_sources += 1
            reports.append({"title": f.stem, "path": str(f), "text": text})
            budget -= len(text)
        if not reports:
            return {"status": "no_sessions", "sources": 0}
        if pending:
            logger.info(f"consolidate: {pending} session(s) beyond the budget stay pending")
        sources = [Path(r["path"]) for r in reports]

        # Curated, git-tracked memory note (research/memory/ is NOT gitignored,
        # and classify_source_type types it "research" — checked before /research/).
        # Resolve its path FIRST so an existing note can be fed back to the
        # consolidator for a conflict-aware update (newer facts supersede stale).
        repos_dir = Path(self.config["paths"]["repos_dir"])
        if project:
            target_repo = repos_dir / project
        else:
            src0, repos_res = sources[0].resolve(), repos_dir.resolve()
            if src0.is_relative_to(repos_res):
                target_repo = repos_dir / src0.relative_to(repos_res).parts[0]
            else:  # library-lane sources (not under repos)
                target_repo = Path(self.config["paths"]["mainframe_dir"]) / "library"
        mem_dir = target_repo / "research" / "memory"
        slug = self._slugify(topic) or "session-memory"
        note = mem_dir / f"{slug}.md"
        # Feed only the prior note's curated BODY back in (drop title/preamble/⚠/
        # Sources) so scaffolding + provenance don't accrete across runs and the
        # ⚠ markup / Sources filenames don't pollute the consolidator or NLI input.
        existing = _canonical_body(note.read_text(encoding="utf-8")) if note.exists() else ""

        merged = self.consolidator.consolidate(reports, topic, existing=existing)
        if not merged or not merged.strip():
            return {"status": "empty_consolidation", "sources": len(sources)}

        # Only now (merge confirmed) spend the NLI load to surface contradictions —
        # never abort the pipeline on its failure (degrades to no surfacing).
        contradictions, nli_status = (self._detect_contradictions(existing, reports)
                                      if existing else ([], "no_prior_note"))
        if contradictions:
            flagged = "\n".join(
                f"- **prior:** {c['prior']}\n  **newer:** {c['newer']}  _(NLI {c['confidence']})_"
                for c in contradictions)
            merged = merged.rstrip() + "\n\n" + CONTRADICTION_HEADING + "\n" + flagged
        # Re-scrub the final note body — AFTER the ⚠ section is appended, because
        # its prior/newer excerpts come from RAW session-report text. This note is
        # GIT-TRACKED and searchable as source_type="research"; a secret the
        # capture-time denylist missed must not be promoted out of the gitignored
        # session lane through EITHER the LLM merge or the NLI excerpts.
        merged = scrub_secrets(merged)
        # B guardrail: the anti-bloat thesis rests on the LLM superseding rather
        # than accreting — surface runaway growth instead of trusting the prompt.
        # Two conditions: grew past NOTE_BLOAT_FACTOR x the prior body (with a
        # floor so young notes can grow), AND exceeds prior + all new material
        # (legitimate growth is bounded by what was actually added).
        new_chars = sum(len(r["text"]) for r in reports)
        note_bloat = (bool(existing)
                      and len(merged) > max(NOTE_BLOAT_FACTOR * len(existing), NOTE_BLOAT_FLOOR)
                      and len(merged) > len(existing) + new_chars)
        if note_bloat:
            logger.warning(
                f"Consolidated note grew {len(existing)} -> {len(merged)} chars — "
                "the merge may be accreting instead of superseding")
        date = datetime.now().strftime("%Y-%m-%d")
        provenance = "\n".join(f"- {f.name}" for f in sources)
        mem_dir.mkdir(parents=True, exist_ok=True)  # deferred past the empty-guard (no orphan dir)
        note.write_text(
            f"# {topic or 'Session memory'}\n\n"
            f"_Consolidated {date} from {len(sources)} session capture(s)._\n\n"
            f"{merged.strip()}\n\n## Sources\n{provenance}\n",
            encoding="utf-8",
        )

        # Prune/archive the raw logs only once the curated note's content is in the
        # index. "unchanged"/"all_chunks_duplicate" both mean it IS indexed (so
        # pruning is safe) — only a hard error/no-content means it isn't.
        ingest_res = self.ingest_file(str(note))
        if ingest_res.get("status") not in ("ingested", "unchanged", "all_chunks_duplicate"):
            return {"status": "memory_not_indexed", "sources": len(sources),
                    "memory_file": str(note), "error": ingest_res.get("error")}

        # Archive (or delete) each source FIRST, then prune ONLY the ones we
        # actually removed — so a mid-loop move failure never orphans a source
        # (pruned from the index but still on disk) or diverges store/manifest.
        removed = self._archive_sources(sources, archive_sources)
        self.delete_files(removed)

        return {
            "status": "consolidated",
            "topic": topic,
            "sources": len(sources),
            "memory_file": str(note),
            "memory_chunks": ingest_res.get("chunks", 0),
            "archived": len(removed) if archive_sources else 0,
            "pending": pending,  # sources beyond the char budget — next run's work
            "unreadable": unreadable,  # skipped (stay on disk + indexed) — surfaced, not silent
            "truncated_sources": truncated_sources,  # single files clamped to the budget
            "contradictions": len(contradictions),
            "nli": nli_status,  # ran / no_prior_note / disabled / error
            "note_bloat_warning": note_bloat,
        }

    def _archive_sources(self, sources: list[Path], archive_sources: bool) -> list[str]:
        """Move each source into the archive (collision-safe, never overwrites) or
        delete it. Returns the paths actually removed from disk — the caller prunes
        exactly those from the index, so a mid-loop failure leaves the rest intact."""
        archive_dir = Path(self.config["paths"]["mainframe_dir"]) / "archive"
        if archive_sources:
            archive_dir.mkdir(parents=True, exist_ok=True)
        removed = []
        for f in sources:
            try:
                if archive_sources:
                    dest = archive_dir / f.name
                    n = 2
                    while dest.exists():  # never overwrite an archived file
                        dest = archive_dir / f"{f.stem}-{n}{f.suffix}"
                        n += 1
                    shutil.move(f, dest)
                else:
                    f.unlink(missing_ok=True)
                removed.append(str(f))
            except OSError as e:
                logger.warning(f"consolidate: could not archive/remove {f}: {e}")
        return removed

    def reload_config(self) -> dict:
        """Hot-reload config.json for the LIVE-READ knobs (chunker, search
        pool/fetch/recency, tiers, memory threshold) without a restart. Models
        are NOT reloaded — embedder/reranker/nli/consolidator/contextual
        changes are reported back as restart-required instead of being
        silently half-applied."""
        old = self.config
        new = load_config()
        self.config = new
        # Store caches these at construction — reconfigure the live instance
        # (Store owns the halflife clamp AND the config extraction) so search
        # behavior actually changes and the defaults can't drift per-site.
        self.store.reconfigure(**Store.search_kwargs_from_config(new))
        changed = {k for k in set(old) | set(new) if old.get(k) != new.get(k)}
        model_sections = ("embedder", "reranker", "nli", "consolidator", "contextual")
        return {
            "status": "reloaded",
            "changed_sections": sorted(changed),
            "restart_required_for": [k for k in model_sections if k in changed],
        }

    def status(self) -> dict:
        """Return Mainframe health status: models (incl. WHICH reranker backend
        is actually loaded — they span 0.33-0.42 quality and 2-8s/query), index
        compaction health, VRAM, and the memory-loop backlog."""
        vram = None
        try:
            import torch
            if torch.cuda.is_available():
                vram = {
                    "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                    "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                }
        except Exception:
            pass
        return {
            "embedder": self.embedder.info,
            "reranker": getattr(self.reranker, "info",
                                {"model": getattr(self.reranker, "model_name", "?")}),
            "vram": vram,
            "store": self.store.get_stats(),
            "index_health": self.store.index_health(),
            "manifest": self.manifest.get_stats(),
            "pending_sessions": pending_sessions_by_project(
                self.config["paths"]["repos_dir"], self.config["paths"]["mainframe_dir"]),
        }


# --- MCP Server ---

app = Server("mainframe-mcp")
mainframe: MainframeMCP | None = None
_mainframe_lock = threading.Lock()

# Read-only mode (MAINFRAME_READ_ONLY): the recall surface an always-on gateway
# injects into ordinary conversations — no deletion/consolidation/bulk-index/
# RAPTOR maintenance. Smaller tool-schema prompt, no accidental mutation.
READ_ONLY_TOOLS = frozenset({"search", "list_files", "status"})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _prewarm_enabled() -> bool:
    return _env_flag("MAINFRAME_PREWARM")


def _read_only() -> bool:
    return _env_flag("MAINFRAME_READ_ONLY")


def get_mainframe() -> MainframeMCP:
    # Lock: the pre-warm thread and the first tool call may race the singleton
    # construction (loading two 9GB embedders would OOM the card).
    global mainframe
    with _mainframe_lock:
        if mainframe is None:
            mainframe = MainframeMCP()
            # Search analytics batch in memory (flush every 25th search) — persist
            # whatever accumulated when the process exits, or up to 24 searches of
            # counts silently vanish every session.
            import atexit
            atexit.register(lambda: mainframe is not None and mainframe.manifest.save())
    return mainframe


@app.list_tools()
async def list_tools() -> list[Tool]:
    tools = [
        Tool(
            name="search",
            description=(
                "Search the Mainframe knowledge base (hybrid vector+keyword, then cross-encoder "
                "reranked). Results are sorted best-first by rerank_score (0-1, HIGHER = more "
                "relevant — the field to trust: >0.85 strong, <0.3 likely noise). `score` is the "
                "raw vector distance (lower = closer; secondary signal only). Each result carries "
                "file/heading/char_start/char_end for citation and truncated=true when the 500-char "
                "text preview was clipped. Session captures are excluded by default; set "
                "include_sessions=true to include them."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": (
                        "Use SPECIFIC technical terms: proper nouns, function/class names, exact "
                        "API/config keywords, error strings — these rerank ~0.99. Natural-language "
                        "questions ('how does X work') rerank ~0.05 and scatter; avoid. One topic "
                        "per search (two searches beat one broad one). Semantic+keyword matching — "
                        "boolean operators/quotes/negation are ignored."
                    )},
                    "limit": {"type": "integer", "description": "Number of results returned (default 3, max 10). More = broader coverage but more noise.", "default": 3},
                    "include_sessions": {"type": "boolean", "description": "Include source_type=session captures (default false — curated knowledge only)", "default": False},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ingest_file",
            description="Ingest a markdown file into the Mainframe. Chunks, embeds, deduplicates, and indexes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to .md file"},
                    "filePath": {"type": "string", "description": "Legacy alias for file_path"},
                },
                # No `required`: hard-requiring file_path would make the MCP
                # SDK's schema validation reject legacy filePath callers before
                # dispatch ever sees them; the handler validates instead.
            },
        ),
        Tool(
            name="capture_memory",
            description="Capture session memory as a local markdown file under <repo>/research/sessions/ and ingest it. Call near session end, or right after a decision/gotcha/hard-won finding worth remembering. LIFECYCLE: captures are raw and EXCLUDED from default search (only include_sessions=true sees them) until a separate consolidate_sessions run curates them into a git-tracked research/memory note. Files are the source of truth: one immutable file per capture. 100% local, no extra model load. The agent authors `content` as markdown sections (## Summary, ## Decisions, ## Action items, etc.).",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Markdown body — dense, retrieval-optimized sections (## Summary with proper nouns/model names/exact terms, ## Decisions, ## Action items, ## Key facts, ## Files touched, ## Open questions)"},
                    "title": {"type": "string", "description": "One-line title (becomes the filename slug + H1)"},
                    "project": {"type": "string", "description": "Repo name. STRONGLY PREFERRED — pass it explicitly. If omitted, the repo is inferred from `cwd` (below), and if that is also omitted it falls back to the MCP server process's own directory, which is usually the WRONG repo."},
                    "kind": {"type": "string", "description": "session | decision | note (frontmatter type)", "default": "session", "enum": ["session", "decision", "note"]},
                    "session_id": {"type": "string", "description": "Current 8-char session id. Pass it when known so the SessionEnd backstop can dedup against this capture; 'adhoc' if empty."},
                    "cwd": {"type": "string", "description": "The agent's current working directory (absolute), used to infer the repo when `project` is omitted. Pass your actual cwd — the server cannot see it."},
                },
                "required": ["content", "title"],
            },
        ),
        Tool(
            name="list_files",
            description="List all files in the Mainframe with ingestion status and retrieval counts.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="delete_file",
            description="Delete a file and its chunks from the Mainframe index.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to file"},
                    "filePath": {"type": "string", "description": "Legacy alias for file_path"},
                },
                # No `required` — see ingest_file (legacy-alias validation lives
                # in the handler).
            },
        ),
        Tool(
            name="ingest_projects",
            description="DEPRECATED — prefer sync_index (a superset that also prunes deleted files). Scan all ~/repos/*/research/*.md and ~/repos/*/CLAUDE.md files and ingest new or changed ones.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="sync_index",
            description="MAINTENANCE — single-writer; do not run while another Mainframe process is writing. Incrementally sync the index with the filesystem: ingest new/modified knowledge files (research/, docs/, CLAUDE.md, research/sessions/ + the library/sessions fallback) AND prune files deleted from disk, then compact the index. Returns per-project pending session-capture counts and a consolidation hint when a backlog is ripe. 0 extra VRAM.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="status",
            description="Return Mainframe health: model info (incl. which reranker backend is loaded), VRAM, index stats + compaction health (fragments/versions), retrieval analytics, and pending session captures per project.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="reload_config",
            description="Hot-reload the config file (MAINFRAME_CONFIG env, else ~/.claude/mainframe/config.json) for live-read knobs (chunker, search pool/fetch/recency, tiers). Model changes (embedder/reranker/nli/consolidator/contextual) are NOT applied — they are returned as restart_required_for.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="build_raptor",
            description="MAINTENANCE/OPS — run deliberately, not inline; single-writer + loads the consolidator (~2.1GB VRAM). Build RAPTOR tier-1 summary chunks by clustering and summarizing tier-0 chunks. NOTE: currently REDUCES the composite retrieval score (helps h@3 recall, hurts text-match) — skip unless deliberately optimizing recall.",
            inputSchema={
                "type": "object",
                "properties": {
                    "distance_threshold": {"type": "number", "description": "Cosine distance threshold for clustering (default 0.8)", "default": 0.8},
                    "min_cluster_size": {"type": "integer", "description": "Minimum chunks per cluster (default 3)", "default": 3},
                },
            },
        ),
        Tool(
            name="consolidate_sessions",
            description="MAINTENANCE/OPS — run deliberately, not inline; single-writer, moves/archives files, and loads the consolidator (~2.1GB VRAM) lazily. Merge raw session captures into ONE curated, git-tracked research/memory/<topic>.md note (local Qwen2.5-3B), index it, and prune/archive the raw logs. Bounds index growth so accreted sessions don't crowd out curated knowledge.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Repo name whose research/sessions/ to consolidate; all repos + library fallback if omitted"},
                    "topic": {"type": "string", "description": "Topic/title for the consolidated note (-> filename slug + H1). Defaults to the project name (one curated note per project) when omitted."},
                    "days": {"type": "integer", "description": "Only consolidate session files modified within this many days; 0 (default) = ALL pending sessions", "default": 0},
                    "archive_sources": {"type": "boolean", "description": "Move raw logs to ~/.claude/mainframe/archive/ for provenance (default true); false deletes them", "default": True},
                },
            },
        ),
    ]
    # Read-only mode: advertise only the recall surface (smaller tool-schema
    # prompt for the client, no accidental-mutation risk).
    if _read_only():
        return [t for t in tools if t.name in READ_ONLY_TOOLS]
    return tools


def _with_file_path(arguments: dict, fn) -> dict:
    """Resolve the file-path arg (canonical `file_path`, legacy `filePath`
    alias) and call `fn` with it — the schema deliberately doesn't hard-require
    either spelling, so missing-arg validation lives here."""
    fp = arguments.get("file_path") or arguments.get("filePath")
    return fn(fp) if fp else {"status": "error", "error": "file_path required"}


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Fast path: post-warm-up the singleton is published and immutable, so read
    # it directly (write-once under _mainframe_lock -> safe). Only calls racing
    # the ~2min pre-warm window take the to_thread detour, which keeps the
    # event-loop thread from blocking on the model-load lock.
    # Read-only mode: refuse mutating tools even if a client somehow calls one
    # (list_tools already hides them). Belt-and-suspenders against injection.
    if _read_only() and name not in READ_ONLY_TOOLS:
        return [TextContent(type="text", text=json.dumps(
            {"error": f"'{name}' is disabled: server is in read-only mode "
                      "(MAINFRAME_READ_ONLY). Only search/list_files/status are available."}))]

    mf = mainframe if mainframe is not None else await asyncio.to_thread(get_mainframe)

    if name == "search":
        results = mf.search(
            arguments["query"],
            arguments.get("limit", 3),
            include_sessions=arguments.get("include_sessions", False),
        )
        # STRUCTURED response — valid JSON for every MCP client (no prose
        # appended to the JSON body). Dead-ends still steer the agent, but via a
        # `guidance` field: the rephrase technique is otherwise tribal knowledge,
        # and a weak top rerank_score is indistinguishable from a real hit.
        if not results:
            confidence, guidance = "none", (
                "No matches. Rephrase with SPECIFIC technical terms (function/class/"
                "proper names, exact config keys, error strings) — natural-language "
                "questions rank poorly. Or set include_sessions=true to also search "
                "raw session captures.")
        elif max(r["rerank_score"] for r in results) < 0.3:
            confidence, guidance = "low", (
                "Best rerank_score < 0.3 (likely noise). Rephrase with more specific "
                "technical terms. Trust rerank_score (higher = better), not score.")
        else:
            confidence, guidance = "high", None
        payload = {"results": results, "confidence": confidence}
        if guidance:
            payload["guidance"] = guidance
        return [TextContent(type="text", text=json.dumps(payload, indent=2))]

    elif name == "ingest_file":
        result = _with_file_path(arguments, mf.ingest_file)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "capture_memory":
        result = mf.capture_memory(
            content=arguments["content"],
            title=arguments["title"],
            project=arguments.get("project"),
            kind=arguments.get("kind", "session"),
            session_id=arguments.get("session_id", ""),
            cwd=arguments.get("cwd", ""),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "list_files":
        result = mf.list_files()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "delete_file":
        result = _with_file_path(arguments, mf.delete_file)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "ingest_projects":
        result = mf.ingest_projects()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "sync_index":
        result = mf.sync_index()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "status":
        result = mf.status()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "reload_config":
        result = mf.reload_config()
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "build_raptor":
        result = mf.build_raptor(
            distance_threshold=arguments.get("distance_threshold", 0.8),
            min_cluster_size=arguments.get("min_cluster_size", 3),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    elif name == "consolidate_sessions":
        result = mf.consolidate_sessions(
            project=arguments.get("project"),
            topic=arguments.get("topic", ""),
            days=arguments.get("days", 0),
            archive_sources=arguments.get("archive_sources", True),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


def main():
    """Entry point for stdio MCP server."""
    import os
    # Reduce CUDA fragmentation OOM under VRAM pressure (set before any model
    # loads, i.e. before the first tool call constructs the embedder).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    # A hub outage must not hang the pre-warm: when every enabled model is
    # already cached, skip HuggingFace HEAD checks entirely (hit live 2026-07-16:
    # 504 retry loops with all weights cached).
    from mainframe_mcp.config import hf_offline_if_cached
    hf_offline_if_cached(load_config())

    # Pre-warm (OPT-IN via MAINFRAME_PREWARM): load models NOW in a daemon
    # thread so the ~2min embedder+4B load overlaps the MCP handshake instead of
    # taxing the FIRST search. Default is LAZY — an always-on gateway sharing a
    # GPU with other agents must not claim ~13GB of VRAM merely because a client
    # connected; models load on the first search instead. Claude-Code users who
    # want instant first-search set MAINFRAME_PREWARM=1.
    if _prewarm_enabled():
        threading.Thread(target=get_mainframe, name="prewarm", daemon=True).start()

    async def _run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
