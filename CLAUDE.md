# Mainframe MCP

Mainframe indexes local Markdown and provides hybrid semantic/keyword search
with reranking. Files are authoritative; the index is disposable. The public
repository is the development home. Read `AGENTS.md`, `CONTRIBUTING.md`, and
`docs/PUBLIC_AUDIT.md` before contributing or publishing.

## Versions and structure

- `src/mainframe_mcp/`: default v1 stdio server, including consolidation, NLI
  contradiction checks, and optional RAPTOR summaries.
- `src/mainframe/`: optional v2 daemon with capture, indexing, search, HTTP MCP,
  REST, CLI, and a stdio forwarding adapter. See `docs/V2.md`.
- v1 owns `.lancedb`; v2 owns `index.lancedb`. Never point either implementation
  at the other's index. v2 builds a new index on first startup.
- v2 consolidation is not implemented. Its capture/receipt primitives do not
  imply a complete ambient consolidation loop.

| Path | Responsibility |
|---|---|
| `src/mainframe/app.py` | Composition and common operation entry points |
| `src/mainframe/core/` | Models, chunks, pipeline, store, search, citations |
| `src/mainframe/memory/` | Lanes, atomic captures, receipts |
| `src/mainframe/service/` | Daemon, discovery, watcher, scheduler, REST |
| `src/mainframe/adapters/` | CLI, MCP mounts, stdio forwarding |
| `tests/`, `tests/v2/` | GPU-free regression suites |
| `configs/`, `eval/` | Hardware presets and evaluation tools |

## Commands

```bash
python -m pip install -e ".[test]"
python -m pytest tests/ -q
python -m pytest tests/v2/test_search.py -q
python -m pip wheel . --no-deps -w dist
python -u -m mainframe_mcp.server
mainframe serve
```

Set `MAINFRAME_CONFIG` explicitly for isolated work. Scope
`paths.include_projects` before starting v2; an empty list includes every
eligible project under `paths.repos_dir`. Keep state, logs, captures, and
model caches outside public Git.

## Design constraints

- `Mainframe.submit` owns mutations. The daemon attaches one writer queue;
  callbacks finish on that writer before results are published. Preserve
  reentrant behavior and recovery of consumed work after failures.
- Keep the daemon lock until its writer exits. Rebuild runs offline under the
  same lock, after `mainframe down`. Never expose rebuild as an online job.
- `Models.invoke` owns loading, use, and release. Callers return results and
  must not retain model references. Recognized device faults get one reload
  and retry; failed loads back off.
- Release partial primary-load allocations before the INT8 fallback. Windows
  error 1455 is host commit exhaustion; preserve its diagnostics and backoff,
  and never classify it as a CUDA-context fault that needs a model reload.
- Health readers use metadata without importing or loading models. Keep
  blocking work off the HTTP event loop.
- Watcher callbacks filter paths cheaply and record invalidations. Lane
  discovery handles scope and containment; rescans recover missed events.
  Index-generated activity must not trigger rescans.
- Index fingerprints describe stored embeddings and chunks. Drift blocks
  writes; incompatible embedding changes also block search until rebuilding.
- Citation lines require a matching file hash and a verified passage.
  Ambiguous matches must never receive guessed source spans.
- Preserve native reranker prompt tokens, truncation limits, and INT8 batch
  order. Batch composition can affect quantized scores. Retrieval changes
  require evaluation, even when automated behavior tests pass.
- Projected store reads must avoid loading vector columns. Read errors must
  not masquerade as an empty ledger or trigger accidental deletion.
- Scrub captures and provenance before writing files or events. Scrubbing
  does not replace publication review.

## Validation and publication

The preview distribution is `2.0.0a1`; `mainframe_mcp.__version__` supplies both
runtime namespaces and build metadata. Keep them consistent. Follow
`docs/PREVIEW_RELEASE.md` for release artifacts, migration, and rollback;
release preparation produces an audited draft prerelease, not a default switch.

Tests use deterministic fake models and temporary LanceDB stores; they must
not download models or require CUDA. Real watcher timing tests use `slow` and
run by default; exclude them only during iteration with `-m "not slow"`.
CI's `package-install` gate separately checks built wheels on Windows/Linux
with Python 3.10/3.14. Run `scripts/check_install.py --wheel-dir dist` against a
directory containing one wheel; it installs dependencies into a fresh CPU-only
environment and imports outside the checkout.

`python -u eval/evaluate.py --daemon` scores v2 without loading another model
set. Default v2 evaluation also uses a running daemon; offline evaluation
uses the configured state directory. `eval/evaluate_v1.py` retains the v1
harness. See `eval/README.md`.

Use a synthetic corpus and separate state directory for live smoke tests.
`mainframe validate --output ~/mainframe-validation/run-01` owns a fresh
synthetic scope and daemon; add `--duration 1800` for a soak. It refuses existing
output directories and paths inside Git. Keep its config, logs, and JSON report
private. Unit tests exercise the same validation flow with fake models.
Validation binds each child to a fresh nonce checked in discovery and health;
do not assume a Windows venv launcher's PID equals the daemon interpreter's PID.
Check available memory before loading models; never run two heavy GPU jobs
concurrently. Use unbuffered Python (`-u`) for long checks and offline model
loading when selected weights are already cached. Do not claim retrieval
quality improvements without measurements.

Before every public push, review exact commits, complete files, history,
metadata, and publication text, including tests and docs. Follow
`docs/PUBLIC_AUDIT.md`, resolve findings, and record the audited SHA. Keep
private corpora and experiment records separate. Never merge private Git
ancestry or revive snapshot publishing.
