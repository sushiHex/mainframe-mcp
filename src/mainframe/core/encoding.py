"""Explicit, reproducible embedding contracts; no model imports or allocation."""

from importlib.metadata import version
import re


def native_contract(config: dict) -> dict | None:
    cfg = config.get('embedder', {})
    mode = cfg.get('encoding', 'legacy')
    if mode == 'legacy':
        return None
    if mode != 'native':
        raise ValueError('embedder.encoding must be legacy or native')
    revision = cfg.get('revision')
    if not isinstance(revision, str) or not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('native encoding requires a pinned 40-character embedder.revision')
    if cfg.get('quantize', True) is not False:
        raise ValueError('native BF16 encoding requires embedder.quantize=false')
    prompt = cfg.get('query_prompt')
    if prompt is not None and (not isinstance(prompt, str) or not prompt):
        raise ValueError('embedder.query_prompt must be a saved prompt name or null')
    length = cfg.get('max_seq_length')
    if length is None:
        length = 2048
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError('native max_seq_length must be a positive integer')
    return {'version': 1, 'revision': revision, 'dtype': 'bfloat16', 'attention': 'sdpa',
            'pooling': 'model', 'normalize': True, 'max_seq_length': length,
            'document_route': 'encode_document', 'query_route': 'encode_query',
            'query_prompt': prompt}


def require_native_runtime():
    """Use the library pair exercised by the pinned local-model trials."""
    from packaging.version import Version

    for package, minimum in (('sentence-transformers', '5.4.1'), ('transformers', '5.7.0')):
        if Version(version(package)) < Version(minimum):
            raise RuntimeError(f'native encoding requires {package}>={minimum}; upgrade mainframe-mcp in its environment')
