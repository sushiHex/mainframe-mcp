"""Opt-in native encoding and same-dimension index isolation, without CUDA."""

import json

import pytest

from mainframe.config import DEFAULTS, deep_copy, load_config
from mainframe.core.store import IndexStaleError, Store


def native_config():
    config = deep_copy(DEFAULTS)
    config['embedder'].update(encoding='native', revision='a' * 40,
                               quantize=False, query_prompt='web_search_query')
    return config


def test_native_settings_load_from_json(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'embedder': native_config()['embedder']}))
    assert load_config(path)['embedder']['encoding'] == 'native'


@pytest.mark.parametrize('change', [{'revision': None}, {'revision': 'main'}, {'quantize': True},
                                   {'encoding': 'typo'}, {'query_prompt': 3},
                                   {'max_seq_length': 0}, {'max_seq_length': False}])
def test_invalid_native_contract_fails_before_device_selection(monkeypatch, change):
    import torch
    from mainframe.core.embedder import Embedder

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: pytest.fail('validation must precede CUDA'))
    config = native_config()
    config['embedder'].update(change)
    with pytest.raises(ValueError):
        Embedder(config)


def test_native_context_cannot_silently_ignore_the_recorded_limit(monkeypatch):
    import weakref
    import torch
    from mainframe.core import encoding
    from mainframe.core.embedder import Embedder

    class FixedContext:
        @property
        def max_seq_length(self):
            return 8192

        @max_seq_length.setter
        def max_seq_length(self, value):
            pass

    allocated = []

    def load(self, *args):
        self.model = FixedContext()
        allocated.append(weakref.ref(self.model))
        self.dimension = 2

    monkeypatch.setattr(encoding, 'require_native_runtime', lambda: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(Embedder, '_load_fp', load)
    with pytest.raises(ValueError, match='context'):
        Embedder(native_config())
    assert allocated[0]() is None


def test_native_loader_and_routes_preserve_model_contract(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import sentence_transformers
    import torch
    import transformers
    from mainframe.core import encoding
    from mainframe.core.embedder import Embedder

    calls = []

    class Model:
        def __init__(self, *args, **kwargs):
            calls.append(('load', kwargs))

        def get_sentence_embedding_dimension(self):
            return 2

        def encode_document(self, texts, **kwargs):
            calls.append(('documents', texts, kwargs))
            return torch.tensor([[1., 0.] for _ in texts])

        def encode_query(self, texts, **kwargs):
            calls.append(('queries', texts, kwargs))
            return torch.tensor([[0., 1.] for _ in texts])

    monkeypatch.setattr(encoding, 'require_native_runtime', lambda: None)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', lambda *a, **kw: SimpleNamespace(auto_map={}))
    monkeypatch.setattr(sentence_transformers, 'SentenceTransformer', Model)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    config = native_config()
    config['paths']['model_cache'] = str(tmp_path)
    model = Embedder(config)
    assert model.embed([]) == []
    assert model.embed(['document']) == [[1., 0.]]
    model.info['encoding']['query_prompt'] = 'metadata must not reconfigure encoding'
    assert model.embed_query('Python API setup') == [0., 1.]
    assert calls[0][1]['revision'] == 'a' * 40
    assert calls[0][1]['trust_remote_code'] is False
    assert calls[0][1]['model_kwargs'] == {'dtype': torch.bfloat16, 'attn_implementation': 'sdpa'}
    assert calls[-1][1] == ['Python API setup']
    assert calls[-1][2]['prompt_name'] == 'web_search_query'
    assert calls[-1][2]['normalize_embeddings'] is True
    assert model.info['encoding']['dtype'] == 'bfloat16'


def test_native_custom_architecture_cannot_silently_fall_back_to_builtin(monkeypatch):
    from types import SimpleNamespace
    import sentence_transformers
    import torch
    import transformers
    from mainframe.core import encoding
    from mainframe.core.embedder import Embedder

    monkeypatch.setattr(encoding, 'require_native_runtime', lambda: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained',
                        lambda *a, **kw: SimpleNamespace(auto_map={'AutoModel': 'custom.Model'}))
    monkeypatch.setattr(sentence_transformers, 'SentenceTransformer',
                        lambda *a, **kw: pytest.fail('must reject before model loading'))
    with pytest.raises(ValueError, match='custom'):
        Embedder(native_config())


def test_native_runtime_refuses_older_dependency_pair(monkeypatch):
    from mainframe.core import encoding

    monkeypatch.setattr(encoding, 'version', lambda name: '5.3.0')
    with pytest.raises(RuntimeError, match='native'):
        encoding.require_native_runtime()


@pytest.mark.parametrize('change', [{'revision': 'b' * 40}, {'query_prompt': None},
                                   {'max_seq_length': 1024}, {'encoding': 'legacy'}])
def test_native_index_blocks_reads_and_writes_when_contract_changes(tmp_path, fake_embedder, change):
    from mainframe.core.indexer import LaneFile, prepare_document
    from mainframe.core.paths import canonical

    config = native_config()
    path = tmp_path / 'document.md'
    path.write_text('# Context\nA useful semantic search document.\n')
    doc = prepare_document(LaneFile(canonical(path), 'knowledge', 'fixture'), config['chunker'], fake_embedder.embed)
    store = Store(tmp_path / 'db', fingerprint=Store.fingerprint_from_config(config))
    store.upsert_batch([doc], [])
    config['embedder'].update(change)
    changed = Store(tmp_path / 'db', fingerprint=Store.fingerprint_from_config(config))
    with pytest.raises(IndexStaleError):
        changed.search(fake_embedder.embed_query('useful'))
    with pytest.raises(IndexStaleError):
        changed.upsert_batch([doc], [])


def test_native_identity_survives_loss_of_legacy_sidecar(tmp_path, fake_embedder):
    from mainframe.core.indexer import LaneFile, prepare_document
    from mainframe.core.paths import canonical

    config = native_config()
    path = tmp_path / 'document.md'
    path.write_text('# Context\nUseful documentation.\n')
    doc = prepare_document(LaneFile(canonical(path), 'knowledge', 'fixture'), config['chunker'], fake_embedder.embed)
    store = Store(tmp_path / 'db', fingerprint=Store.fingerprint_from_config(config))
    store.upsert_batch([doc], [])
    store._marker_path().unlink()
    config['embedder']['encoding'] = 'legacy'
    reopened = Store(tmp_path / 'db', fingerprint=Store.fingerprint_from_config(config))
    with pytest.raises(IndexStaleError):
        reopened.search(fake_embedder.embed_query('useful'))


def test_unknown_vectors_cannot_acquire_native_identity_without_rebuild(tmp_path, fake_embedder):
    from v2.test_store import _rows

    original = Store(tmp_path / 'db', embedding_dim=64)
    original.upsert_batch([_rows(fake_embedder, 'c:/fixture/doc.md', ['alpha'])])
    original._marker_path().unlink()
    fingerprint = Store.fingerprint_from_config(native_config())
    native = Store(tmp_path / 'db', fingerprint=fingerprint)
    with pytest.raises(IndexStaleError):
        native.search(fake_embedder.embed_query('alpha'))
    assert not native._marker_path().exists()
    native.drop()
    native.upsert_batch([_rows(fake_embedder, 'c:/fixture/doc.md', ['alpha'])])
    assert Store(tmp_path / 'db', fingerprint=fingerprint).config_drift == {}
    with pytest.raises(IndexStaleError):
        Store(tmp_path / 'db').search(fake_embedder.embed_query('alpha'))


def test_v1_refuses_native_config_instead_of_using_legacy_query_prefix(tmp_path):
    from mainframe_mcp.config import load_config as load_v1

    path = tmp_path / 'config.json'
    path.write_text(json.dumps(native_config()))
    with pytest.raises(ValueError, match='v2'):
        load_v1(path)


def test_offline_native_rebuild_records_new_contract_and_preserves_rollback(cfg, tmp_path):
    from mainframe.adapters.cli import run_offline_rebuild
    from mainframe.app import Mainframe
    from mainframe.core.store import INDEX_DIRNAME
    from v2.fakes import fake_models
    from v2.helpers import write_md

    write_md(tmp_path / 'repos' / 'p' / 'docs' / 'a.md', '# Alpha\n\nUseful documentation.\n')
    original = Mainframe(cfg, models=fake_models(cfg))
    assert run_offline_rebuild(cfg, app=original)['upserted_rows'] == 1
    rollback = original.store.ledger()
    candidate = deep_copy(cfg)
    candidate['paths']['mainframe_dir'] = str(tmp_path / 'native-state')
    candidate['embedder'].update(encoding='native', revision='a' * 40, quantize=False)
    native = Mainframe(candidate, models=fake_models(candidate))
    assert run_offline_rebuild(candidate, app=native)['upserted_rows'] == 1
    candidate['embedder']['revision'] = 'b' * 40
    changed = Mainframe(candidate, models=fake_models(candidate))
    assert changed.store.config_drift
    assert run_offline_rebuild(candidate, app=changed)['upserted_rows'] == 1
    assert not Store(tmp_path / 'native-state' / INDEX_DIRNAME,
                     fingerprint=Store.fingerprint_from_config(candidate)).config_drift
    assert original.store.ledger() == rollback
