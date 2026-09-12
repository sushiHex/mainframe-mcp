"""Observe and supervise one paced embedding probe on a shared Windows GPU."""

import argparse
import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from trial_guard import TrialSettings, WindowsResources, supervise, write_json


class VramLease:
    """Use the installed vram-mcp over MCP; claims remain cooperative."""

    def __init__(self, gib, output):
        self.gib, self.output = gib, output
        self.stack = AsyncExitStack()
        self.claim_id = None

    async def call(self, name, arguments):
        response = await asyncio.wait_for(self.session.call_tool(name, arguments), timeout=10)
        if response.isError:
            raise RuntimeError(f'vram-mcp {name} failed')
        data = response.structuredContent
        if data is None:
            data = json.loads(next(c.text for c in response.content if c.type == 'text'))
        if not isinstance(data, dict) or 'error' in data:
            raise RuntimeError(f'invalid vram-mcp {name} response')
        return data

    async def __aenter__(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        command = shutil.which('vram-mcp')
        if command is None:
            raise RuntimeError('vram-mcp must be installed and available on PATH')
        try:
            log = self.stack.enter_context((self.output / 'vram-mcp.log').open('w', encoding='utf-8'))
            read, write = await self.stack.enter_async_context(
                stdio_client(StdioServerParameters(command=command), errlog=log))
            self.session = await self.stack.enter_async_context(ClientSession(read, write))
            await asyncio.wait_for(self.session.initialize(), timeout=10)
            # Preserve coordination evidence without treating other claims as a
            # blanket admission cutoff; this lease does not evict any workload.
            write_json(self.output / 'prior-claims.json', await self.call('list_claims', {}))
            await self.reserve()
            return self
        except BaseException:
            await self.__aexit__(*sys.exc_info())
            raise

    async def reserve(self):
        result = await self.call('reserve', {'gb': self.gib, 'owner': 'mainframe-embedding-trial',
                                           'purpose': 'one paced local model evaluation',
                                           'pid': os.getpid(), 'ttl_seconds': 60})
        if result.get('ok') is not True or not result.get('claim_id'):
            raise RuntimeError('VRAM reservation refused')
        previous, self.claim_id = self.claim_id, result['claim_id']
        with (self.output / 'lease-events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'claim_id': self.claim_id, 'replaces': previous,
                                     'expires_at': result.get('expires_at')}) + '\n')

    async def renew(self):
        result = await self.call('renew', {'claim_id': self.claim_id, 'ttl_seconds': 60})
        if result.get('ok') is False:
            # The ledger uses wall-clock expiry. A missing/expired advisory
            # claim can be replaced before the supervisor publishes a fresh
            # permit; losing it need not discard completed retrieval work.
            await self.reserve()
        elif result.get('ok') is not True:
            raise RuntimeError('VRAM reservation renewal failed')

    async def __aexit__(self, *exc):
        try:
            if self.claim_id:
                result = await self.call('release', {'claim_id': self.claim_id})
                # False means the ledger already has no such claim. Both
                # outcomes leave this owner's reservation absent.
                if not isinstance(result.get('ok'), bool):
                    raise RuntimeError('VRAM reservation release not confirmed; 60-second TTL remains')
        finally:
            await self.stack.aclose()


def prepare_output(path):
    path = path.resolve()
    parent = path
    while not parent.exists():
        parent = parent.parent
    result = subprocess.run(['git', '-C', str(parent), 'rev-parse', '--show-toplevel'],
                            capture_output=True, text=True)
    if result.returncode == 0:
        raise ValueError('trial output must be outside Git worktrees')
    path.mkdir(parents=True, exist_ok=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--candidate', choices=['nemotron', 'harrier', 'voyage'])
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--manifest', type=Path, help='run retrieval instead of the short probe')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--index-manifest', type=Path,
                        help='original manifest for reuse of a legacy completed index')
    parser.add_argument('--label')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    comparison = args.manifest is not None
    if comparison and not all((args.config, args.index, args.label)):
        parser.error('retrieval requires --config, --index and --label')
    if not args.preflight_only and not comparison and (not args.candidate or not args.snapshot):
        parser.error('--candidate and --snapshot are required for a probe')
    # CUDA/NVML numbering must agree. Do not silently monitor another GPU.
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '0'):
        parser.error('probes require the default physical GPU; unset CUDA_VISIBLE_DEVICES')
    output = prepare_output(args.output)
    settings = TrialSettings()
    resources = None
    try:
        resources = WindowsResources()
        if args.preflight_only:
            sample = resources()
            sample.validate()
            report = {'status': 'observed', 'reason': None,
                      'settings': asdict(settings), 'memory_policy': 'observe', 'last_sample': asdict(sample),
                      'responsiveness': 'unvalidated', 'worker_started': False}
            write_json(output / 'guard-report.json', report)
        else:
            from embedding_probe import validate_snapshot
            from trial_worker import OwnedWorker
            if comparison:
                from compare_models import load_manifest
                load_manifest(args.manifest)
                config = json.loads(args.config.read_text(encoding='utf-8'))
                if args.candidate:
                    validate_snapshot(args.candidate, Path(config['embedder']['model']))
            else:
                validate_snapshot(args.candidate, args.snapshot)

            def launch(permit, nonce):
                common = ['--output', output / 'worker', '--permit', permit, '--nonce', nonce, '--permit-stdin']
                if comparison:
                    script = 'compare_models.py'
                    arguments = ['--manifest', args.manifest.resolve(), '--config', args.config.resolve(),
                                 '--index', args.index.resolve(), '--label', args.label]
                    if args.index_manifest:
                        arguments += ['--index-manifest', args.index_manifest.resolve()]
                    if args.candidate:
                        arguments += ['--candidate', args.candidate, '--native-embedder']
                else:
                    script = 'embedding_probe.py'
                    arguments = ['--candidate', args.candidate, '--snapshot', args.snapshot.resolve()]
                return OwnedWorker(args.python, Path(__file__).with_name(script), arguments + common, output)

            # Advisory capacity claim, not an enforced allocation ceiling.
            report = asyncio.run(supervise(settings, output, resources, launch,
                                          VramLease(15. if comparison else 7., output)))
    except Exception as error:
        report = {'status': 'refused', 'reason': f'{type(error).__name__}: {error}',
                  'responsiveness': 'unvalidated', 'worker_started': False}
        write_json(output / 'guard-report.json', report)
    finally:
        if resources is not None:
            resources.close()
    print(json.dumps(report, indent=2))
    return 0 if report['status'] in ('observed', 'completed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
