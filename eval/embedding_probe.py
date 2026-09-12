"""Eight synthetic encoding cases; invoked only by run_trial's owned worker."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trial_guard import open_gate, write_json

PROFILES = {
    'nemotron': {'model': 'nvidia/Nemotron-3-Embed-1B-BF16',
                 'revision': 'c0c9fea93ea424587517f2c59e20db9f1d6bf615',
                 'dimension': 2048, 'query_prompt': None},
    'harrier': {'model': 'microsoft/harrier-oss-v1-0.6b',
                'revision': 'f9b9dc8d367d443f2479d27aa5d8d2850c0774ee',
                'dimension': 1024, 'query_prompt': 'web_search_query'},
    'voyage': {'model': 'voyageai/voyage-4-nano',
               'revision': '67fabc9bef010dabc5f6024aa1b1b6b93410426f',
               'dimension': 2048, 'query_prompt': None,
               'reviewed_code': {'modeling_qwen3_bidirectional.py':
                   'f4340347ce92a764e6ce6dc76fb4e412a6ab876dd33e61a9ed4de7595c697728'}},
}
CASES = (
    ('short', 'Where is the configuration file?'),
    ('code', 'def embed_query(query: str) -> list[float]:'),
    ('config', '{"embedder": {"quantize": false, "max_seq_length": 2048}}'),
    ('empty', ''),
    ('unicode', 'Where is caf\u00e9.md? \u65e5\u672c\u8a9e \u2014 \u0645\u0631\u062d\u0628\u0627 \U0001f50e'),
    ('paths', 'src/mainframe/core/embedder.py and tests/v2/test_search.py'),
    ('question', 'How does an offline rebuild detect a changed embedding model?'),
    ('long', 'semantic ' * 2048),
)
NORM_TOLERANCE = .01


def validate_snapshot(candidate, snapshot):
    profile = PROFILES[candidate]
    if not snapshot.is_dir() or snapshot.name != profile['revision']:
        raise ValueError('use the pinned local snapshot directory from the trial plan')
    if candidate == 'voyage':
        code = {str(p.relative_to(snapshot)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in snapshot.rglob('*.py')}
        if code != profile['reviewed_code']:
            raise ValueError('Voyage custom code differs from the reviewed source')
    settings = json.loads((snapshot / 'config_sentence_transformers.json').read_text(encoding='utf-8'))
    prompts = settings.get('prompts', {})
    if candidate == 'nemotron' and (prompts.get('query') != 'query: ' or prompts.get('document') != 'passage: '):
        raise ValueError('Nemotron query/document prompts changed')
    if candidate == 'harrier':
        if not prompts.get('web_search_query'):
            raise ValueError('Harrier requires the saved web_search_query prompt')
        if prompts.get('document') or settings.get('default_prompt_name'):
            raise ValueError('Harrier documents must remain unprefixed')
    if candidate == 'voyage':
        if prompts != {'query': 'Represent the query for retrieving supporting documents: ',
                       'document': 'Represent the document for retrieval: '}:
            raise ValueError('Voyage query/document prompts changed')
        config = json.loads((snapshot / 'config.json').read_text())
        if config.get('auto_map') != {'AutoModel': 'modeling_qwen3_bidirectional.Qwen3BidirectionalModel'}:
            raise ValueError('Voyage must use the reviewed bidirectional model')
    return {**profile, 'saved_prompts': prompts}


def validate_vectors(vectors, *, count, dimension):
    if len(vectors) != count or any(len(row) != dimension for row in vectors):
        raise ValueError('embedding count or dimension changed')
    if any(not math.isfinite(v) for row in vectors for v in row):
        raise ValueError('non-finite embedding')
    error = max((abs(math.sqrt(sum(v * v for v in row)) - 1) for row in vectors), default=0.)
    if error > NORM_TOLERANCE:
        raise ValueError(f'embedding norm error {error} exceeds {NORM_TOLERANCE}')
    return {'count': count, 'dimension': dimension, 'max_norm_error': error,
            'norm_tolerance': NORM_TOLERANCE}


def run(args):
    profile = validate_snapshot(args.candidate, args.snapshot)
    gate, settings = open_gate(args.permit, args.nonce, stdin=args.permit_stdin)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'guard_nonce': args.nonce, 'candidate': profile,
              'settings': asdict(settings), 'memory_policy': 'observe',
              'cases_sha256': hashlib.sha256(json.dumps(CASES).encode()).hexdigest(),
              'responsiveness': 'unvalidated', 'retrieval_quality': 'unmeasured',
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  Path(__file__), Path(__file__).with_name('compare_models.py'),
                  Path(__file__).with_name('trial_guard.py'))}}
    model = None
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('probe requires CUDA')
        gate.configure_cuda(torch)
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        from compare_models import experimental_factories
        report['versions'] = {name: importlib.metadata.version(name) for name in (
            'torch', 'sentence-transformers', 'transformers', 'bitsandbytes')}
        factory, _ = experimental_factories(True, False, query_prompt=profile['query_prompt'], gate=gate,
                                             trust_remote_code=bool(profile.get('reviewed_code')))
        started = time.perf_counter()
        model = factory({'embedder': {'model': str(args.snapshot), 'quantize': False,
                                      'batch_size': 8, 'max_seq_length': 2048},
                         'paths': {'model_cache': str(args.output / 'unused-cache')}})
        report['load_s'] = time.perf_counter() - started
        gate.wait()
        parameters = list(model.model.parameters())
        report['parameter_devices'] = sorted({str(p.device) for p in parameters})
        report['parameter_dtypes'] = sorted({str(p.dtype) for p in parameters})
        if not parameters or any(p.device.type != 'cuda' for p in parameters):
            raise ValueError('parameter offload invalidates the probe')
        if any(p.is_floating_point() and p.dtype != torch.bfloat16 for p in parameters):
            raise ValueError('native BF16 precision contract changed')
        del parameters
        if model.dimension != profile['dimension'] or model.model.max_seq_length != 2048:
            raise ValueError('dimension or context cap changed')
        report['modules'] = str(model.model)
        if model.embed([]) != []:
            raise ValueError('empty document list must produce an empty list')
        started = time.perf_counter()
        report['documents'] = validate_vectors(model.embed([text for _, text in CASES]),
                                                count=len(CASES), dimension=profile['dimension'])
        report['queries'] = {}
        for name, text in CASES:
            report['queries'][name] = validate_vectors([model.embed_query(text)],
                                                        count=1, dimension=profile['dimension'])
        gate.wait()
        report['encoding_wall_s'] = time.perf_counter() - started
        report['forward_s'], report['pacing_s'] = gate.forward_seconds, gate.idle_seconds
        report['monitor_wait_s'] = gate.monitor_wait_seconds
        report['forwards'], report['input_shapes'] = gate.forwards, gate.input_shapes
        report['cuda'] = {'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                          'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
        import psutil
        host_memory = psutil.Process().memory_info()
        report['host'] = {'private_gib': host_memory.private / 2**30,
                          'peak_process_commit_gib': host_memory.peak_pagefile / 2**30}
        if not gate.input_shapes or max(shape[-1] for shape in gate.input_shapes) != 2048:
            raise ValueError('long input did not exercise the 2048-token cap')
        started = time.perf_counter()
        gate.close()
        model = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        report['release_s'] = time.perf_counter() - started
        report['cuda']['allocated_after_release_gib'] = torch.cuda.memory_allocated() / 2**30
        gate.wait()
        report['complete'] = True
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        gate.close()
        model = None
        write_json(args.output / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=list(PROFILES), required=True)
    for name in ('snapshot', 'output', 'permit'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--permit-stdin', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
