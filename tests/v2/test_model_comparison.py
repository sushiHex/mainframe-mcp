"""GPU-free checks for evaluation integrity and experimental model routing."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_models", Path(__file__).resolve().parents[2] / "eval" / "compare_models.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def test_repeated_document_does_not_earn_multiple_relevance_gains():
    query = {"relevant": ["a", "b"]}
    rows = [{"file": p} for p in ("a.md", "a.md", "b.md")]
    score = comparison.ranking_metrics(query, rows, {"a.md": "a", "b.md": "b"})
    # Duplicate a consumes rank 2; b earns its first credit at rank 3.
    assert score["ndcg3"] == pytest.approx(1.5 / (1 + 1 / math.log2(3)))
    assert score["mrr3"] == 1


def test_passage_score_requires_answer_in_relevant_document():
    query = {"relevant": ["a"], "expected_text": "the answer"}
    rows = [{"file": "b.md", "text": "the answer"},
            {"file": "a.md", "text": "unrelated section"},
            {"file": "a.md", "text": "THE\n answer"}]
    score = comparison.ranking_metrics(query, rows, {"a.md": "a", "b.md": "b"})
    assert score["mrr3"] == .5
    assert score["passage1"] == 0
    assert score["passage3"] == 1
    assert comparison.ranking_metrics(query, [], {})["hit3"] == 0


def test_manifest_refuses_changed_corpus_and_unknown_labels(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("original")
    manifest = {"files": [{"id": "a", "path": str(doc), "sha256": comparison.digest(doc.read_bytes())}],
                "queries": [{"id": "q", "query": "question", "relevant": ["a"]}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert comparison.load_manifest(path)[0] == manifest
    doc.write_text("changed")
    with pytest.raises(ValueError, match="corpus changed"):
        comparison.load_manifest(path)
    doc.write_text("original")
    manifest["queries"][0]["relevant"] = ["missing"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid relevance"):
        comparison.load_manifest(path)


def test_manifest_refuses_one_file_with_two_relevance_identities(tmp_path):
    doc = tmp_path / 'doc.md'
    doc.write_text('frozen document')
    entry = {'path': str(doc), 'sha256': comparison.digest(doc.read_bytes())}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'files': [{'id': 'a', **entry}, {'id': 'b', **entry}],
                                'queries': [{'id': 'q', 'query': 'question', 'relevant': ['a']}]}))
    with pytest.raises(ValueError, match='duplicate document path'):
        comparison.load_manifest(path)


@pytest.mark.parametrize('prompt', [None, 'web_search_query'])
def test_native_embedding_uses_model_query_and_document_routes(monkeypatch, tmp_path, prompt):
    import sentence_transformers
    import torch
    from mainframe.config import DEFAULTS, deep_copy

    calls = []

    class FakeModel:
        def __init__(self, *a, **kw):
            calls.append(("load", kw))

        def get_sentence_embedding_dimension(self):
            return 2

        def encode_document(self, texts, **kw):
            calls.append(("documents", texts))
            return torch.tensor([[1., 0.] for _ in texts])

        def encode_query(self, texts, **kw):
            calls.append(("query", texts, kw))
            return torch.tensor([[0., 1.] for _ in texts])

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    native, _ = comparison.experimental_factories(True, False, query_prompt=prompt)
    config = deep_copy(DEFAULTS)
    config["embedder"]["quantize"] = False
    config["paths"]["model_cache"] = str(tmp_path)
    model = native(config)
    assert model.embed(["document"]) == [[1., 0.]]
    assert model.embed_query("Python API question") == [0., 1.]
    assert calls[-2] == ("documents", ["document"])
    assert calls[-1][:2] == ("query", ["Python API question"])
    assert calls[-1][2].get('prompt_name') == prompt
    assert calls[0][1]["trust_remote_code"] is False


def test_bf16_adapter_preserves_production_heading_and_ranking(monkeypatch):
    import sentence_transformers
    import torch
    from mainframe.config import DEFAULTS, deep_copy

    calls = []

    class FakeModel:
        def __init__(self, *a, **kw):
            calls.append(kw)

        def predict(self, pairs, **kw):
            calls.append(pairs)
            return [.2, .8]

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeModel)
    _, factory = comparison.experimental_factories(False, True)
    reranker = factory(deep_copy(DEFAULTS))
    assert reranker.rerank("query", ["first", "second"], top_k=1,
                           headings=["Title", "(no heading)"]) == [(1, .8)]
    assert calls[0]["model_kwargs"]["dtype"] == torch.bfloat16
    assert calls[-1] == [("query", "## Title\nfirst"), ("query", "second")]


def test_trial_operation_failure_never_enters_production_recovery():
    from trial_guard import TrialAborted

    calls = []

    def fail():
        calls.append('forward')
        raise RuntimeError('CUDA out of memory')

    with pytest.raises(TrialAborted, match='CUDA out of memory'):
        comparison.trial_call(fail)
    assert calls == ['forward']


def test_index_identity_covers_prompt_precision_and_candidate_revision():
    config = {'embedder': {'model': 'local-model', 'quantize': False}}
    first = comparison.index_identity(config, 'corpus', 'fingerprint', True,
                                      {'revision': 'a', 'query_prompt': None})
    changed = comparison.index_identity(config, 'corpus', 'fingerprint', True,
                                        {'revision': 'b', 'query_prompt': 'search'})
    assert first != changed
    assert first['encoding']['dtype'] == 'bfloat16'
    assert first['encoding']['normalize'] is True


def test_corpus_identity_ignores_questions_and_file_order(tmp_path):
    files = [{'id': name, 'path': str(tmp_path / name), 'sha256': name * 64}
             for name in ('a', 'b')]
    original = {'files': files, 'queries': [{'query': 'old question'}]}
    fresh = {'files': list(reversed(files)), 'queries': [{'query': 'unseen question'}]}
    assert comparison.corpus_digest(original) == comparison.corpus_digest(fresh)
    for field, value in [('id', 'renamed'), ('path', str(tmp_path / 'moved')), ('sha256', 'c' * 64)]:
        changed = json.loads(json.dumps(fresh))
        changed['files'][0][field] = value
        assert comparison.corpus_digest(original) != comparison.corpus_digest(changed)


def test_legacy_index_reuse_requires_matching_original_manifest(tmp_path):
    doc = tmp_path / 'doc.md'
    doc.write_text('Frozen answer passage.')
    original = {'files': [{'id': 'a', 'path': str(doc), 'sha256': comparison.digest(doc.read_bytes())}],
                'queries': [{'id': 'old', 'query': 'old question', 'relevant': ['a']}]}
    prior = tmp_path / 'original.json'
    prior.write_text(json.dumps(original))
    fresh = {**original, 'queries': [{'id': 'new', 'query': 'unseen question', 'relevant': ['a']}]}
    identity = comparison.index_identity({'embedder': {'model': 'fixture'}},
                                         comparison.corpus_digest(fresh), 'fingerprint', False, None)
    index = tmp_path / 'index'
    assert comparison.reuse_index(index, identity) is False
    index.mkdir()
    marker = index / 'comparison.json'
    # A partial index cannot be reused even if its source manifest is supplied.
    with pytest.raises(ValueError, match='incomplete|different inputs'):
        comparison.reuse_index(index, identity, prior)
    legacy = {k: v for k, v in identity.items() if k not in ('schema_version', 'corpus_sha256')}
    legacy['manifest_sha256'] = comparison.digest(prior.read_bytes())
    marker.write_text(json.dumps(legacy))
    marker_before = marker.read_bytes()
    with pytest.raises(ValueError, match='different inputs'):
        comparison.reuse_index(index, identity)
    assert comparison.reuse_index(index, identity, prior) is True
    assert marker.read_bytes() == marker_before  # Evidence stays immutable.
    for field, value in [('corpus_sha256', 'changed'), ('fingerprint', 'changed'), ('embedder', {})]:
        with pytest.raises(ValueError, match='different inputs'):
            comparison.reuse_index(index, {**identity, field: value}, prior)
    prior.write_text(json.dumps(fresh))
    with pytest.raises(ValueError, match='different inputs'):
        comparison.reuse_index(index, identity, prior)
    marker.write_text(json.dumps(identity))
    assert comparison.reuse_index(index, identity) is True


def test_paced_reranker_keeps_inputs_and_stops_failed_forward(monkeypatch):
    import torch
    from mainframe.config import DEFAULTS, deep_copy
    from mainframe.core.reranker import Reranker
    from trial_guard import BatchGate, TrialAborted

    gate = BatchGate(Path('unused'), 'test', synchronize=lambda: None)
    monkeypatch.setattr(gate, 'check', lambda: None)
    calls = []

    class Forward(torch.nn.Module):
        def forward(self, **kwargs):
            calls.append(kwargs)
            raise torch.cuda.OutOfMemoryError('fixture OOM')

    def load(self, config):
        self.model = Forward()

    monkeypatch.setattr(Reranker, '__init__', load)
    _, factory = comparison.experimental_factories(False, False, gate=gate)
    model = factory(deep_copy(DEFAULTS))
    with pytest.raises(TrialAborted, match='forward failed'):
        model.model(input_ids=torch.zeros((8, 42)), logits_to_keep=1)
    assert len(calls) == 1
    assert calls[0]['logits_to_keep'] == 1
    gate.close()


def test_uninstrumented_reranker_is_refused_before_loading(monkeypatch):
    from mainframe.config import DEFAULTS, deep_copy
    from mainframe.core.reranker import Reranker
    from trial_guard import BatchGate

    monkeypatch.setattr(Reranker, '__init__', lambda *a: pytest.fail('must refuse before loading'))
    gate = BatchGate(Path('unused'), 'test')
    monkeypatch.setattr(gate, 'check', lambda: None)
    config = deep_copy(DEFAULTS)
    config['reranker']['backend'] = 'cross-encoder'
    _, factory = comparison.experimental_factories(True, False, gate=gate)
    with pytest.raises(ValueError, match='Qwen3'):
        factory(config)
