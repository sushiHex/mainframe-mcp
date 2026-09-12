# mainframe-mcp

A **100% local, GPU-accelerated semantic knowledge base** for [Claude Code](https://claude.com/claude-code) and any other [MCP](https://modelcontextprotocol.io/)-compatible agent. It indexes Markdown across your repositories (`research/`, `docs/`, and root files like `CLAUDE.md` / `AGENTS.md`), provides hybrid search with source citations, and captures session knowledge for later recall.

Everything runs on your own GPU. Nothing leaves your machine at query time.

## Architecture

![mainframe-mcp architecture](docs/architecture.svg)

*Ingest builds a disposable dual index; queries combine vector and keyword retrieval with native reranking. The diagram also shows legacy v1 consolidation, which is separate from the default daemon's capture and recall workflow. Files are the source of truth.*

## What it does

- **Hybrid retrieval, reranked** — vector nearest-neighbor (Harrier 0.6B, native BF16) + Tantivy full-text search, merged and re-scored by a cross-encoder reranker. Markdown-header-aware 256-token chunking, model-owned query/document encoding, section-heading injection at rerank time.
- **Native Qwen3-Reranker backend** — the reranker scores candidates via the model's yes/no logits with a task instruction (the model-card method), not a bolted-on classification head. On our eval this beat bge-reranker-v2-m3 by **+26% composite / +33% hit@1**.
- **Legacy v1 ambient memory (mem0-inspired)** — agents write immutable session-capture files (secret-scrubbed, gitignored lane); a consolidation pass merges them into curated memory notes where **newer facts supersede stale ones**, with NLI-backed contradiction surfacing and anti-bloat guardrails. Keep private notes in your private knowledge corpus. Files are the source of truth — the vector index is derived and disposable.
- **Ops-hardened** — read-only search hot path, atomic manifest writes with crash recovery, index compaction + FTS refresh after ingest batches, poison-file-resilient syncs, path-traversal validation, and a layered secret-scrub denylist on every write path.
- **Measured, not vibed** — an eval harness with a regression gate (`eval/evaluate.py --min-score`), an experiment-log discipline (`eval/results.template.md`), and a GPU-free test suite (fakes + real LanceDB) that runs in CI.

**The default is the v2 daemon with Harrier 0.6B in BF16 and Qwen3-Reranker-4B
in INT8.** The reranker projects only the final token and budgets documents
inside the existing 2,048-token context, reserving the assistant scoring suffix.
All clients share one model stack. The older Qwen embedding presets now select
Harrier; no experimental option or extra dependency group is needed.

The original 47-question comparison measured **7.80 versus 14.33 GiB** peak
CUDA allocation for Harrier versus Qwen 8B. With the context correction,
Harrier finds all **20/20 top-three passages and 5/5 critical answers** in the
known challenge. First-place passage matches decreased from 16 to 15; the
47-question working sample changed from 27 to 26, retaining all 37 top-three
matches. These measured tradeoffs were accepted for default promotion.
See [model comparisons](docs/MODEL_COMPARISON.md) and
[context regression results](docs/RERANKER_CONTEXT.md) for the original scopes.

An installed, unpaced trial across 18 documents peaked at **6.886 GiB CUDA
allocation** and passed search, private question recording, and restart reuse.
The operator reported smooth interaction; displayed-frame timing was not
measured. Memory use depends on scope and query length. Keep a
[private question journal](docs/V2.md#everyday-harrier-evaluation) for ordinary work.

## Requirements

- Python 3.10+ (developed on 3.14). **3.11–3.13 have the broadest binary-wheel coverage**; on very new interpreters some deps (lancedb, pydantic-core) may not have Linux wheels yet — pin to 3.13 there if `pip install` fails building from source.
- NVIDIA GPU with BF16 support and headroom for the measured ~7–8 GiB Harrier search stack (developed on an RTX 3090 24 GB); measure your workload and leave room for the desktop.
- PyTorch with CUDA — on Windows the default pip wheel is CPU-only:
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```

## Install

Current public `main` uses Harrier and the v2 daemon by default. The original
`v2.0.0a1` release artifact predates this promotion; install from the current
checkout or a wheel built from it. Use an environment with CUDA-enabled PyTorch:

```bash
python -m pip install -e .
# For contributors:
python -m pip install -e ".[test]"
```

Sentence Transformers 5.4.1+ and Transformers 5.7.0+ are core dependencies.
Copy `config.example.json` to `~/.claude/mainframe/config.json`, set
`paths.include_projects`, and run:

```bash
mainframe serve
# In another terminal:
mainframe search "How does native index identity work?" --detailed
mainframe status
mainframe down
```

`MAINFRAME_CONFIG` selects another configuration file. Restart after changing
it. The daemon loads models lazily; `service.prewarm: true` opts into eager
loading. Model weights download from Hugging Face on first use.

### Connect an MCP host

Use the daemon's Streamable HTTP endpoint `http://127.0.0.1:7433/mcp` or
`/mcp/ro` for search and status only. For stdio hosts, `mainframe-mcp` forwards
requests to that daemon without loading another model stack:

```bash
claude mcp add mainframe -- mainframe-mcp
# Equivalent module entry point:
python -u -m mainframe.adapters.mcp_stdio_shim
```

Start the daemon separately and use the same configuration for every client.
`mainframe-mcp` now uses this v2 transport. The explicit
`python -m mainframe_mcp.server` entry point remains the legacy v1 implementation,
which needs its own legacy model configuration and does not support Harrier's
native encoding. The [migration guide](docs/PREVIEW_RELEASE.md) describes the
capability differences and index transition.

## Shared daemon

The `mainframe` command runs one background service for multiple MCP
clients. It watches files, reconciles missed changes, and exposes HTTP MCP at
`http://127.0.0.1:7433/mcp`; `/mcp/ro` exposes only search and status. Stdio-only
clients can use `python -u -m mainframe.adapters.mcp_stdio_shim`.

Configure `paths.include_projects` before running `mainframe serve`: an empty
list includes all eligible projects under `paths.repos_dir`. The daemon builds
its own `index.lancedb`, separate from v1's `.lancedb`. Index-setting drift is
reported explicitly; rebuilding requires stopping the daemon first.

v2 includes model recovery, immutable captures, and verified source citations.
Its memory workflow supports capture, indexing, and search. Legacy v1 retains
consolidation, contradiction detection, and RAPTOR; those are not daemon features.
Both configuration loaders accept shared files containing v2 project-filter
lists; v1 preserves those lists without treating them as filesystem paths.
See [the v2 guide](docs/V2.md) for setup, commands, architecture, and recovery.

Harrier uses a pinned revision, model-owned query/document routes, BF16,
and a recorded index identity. The [Harrier configuration](configs/v2-harrier.json)
shows these defaults explicitly. Existing Qwen vectors cannot be reused:
stop the daemon and run `mainframe rebuild` with the new configuration, or
point it at an already validated Harrier index. Native inference is unpaced.
See the [encoding contract](docs/V2.md#native-embedding-contract).

### Validate a v2 deployment

Start with one explicitly included project and a separate state directory.
Use `/healthz` for frequent liveness probes and `mainframe status` for index,
queue, and model diagnostics. Browser requests must come from the daemon's
own origin; configure `service.token` to authenticate local clients as well.

Before GPU work, check available memory and coordinate capacity with other
clients. If you use `vram-mcp`, reserve capacity for Mainframe's non-Ollama
models, renew the reservation while running, and release it after shutdown.
Reservations are cooperative and cannot prevent unrelated GPU allocations.
They also do not limit GPU compute or protect desktop responsiveness. Use
small-model trials interactively and schedule sustained large-model sweeps
for a dedicated window. Before another shared-desktop sweep, establish memory
monitoring, pacing, and responsiveness checks using the
[GPU evaluation guidance](eval/README.md#shared-desktop-gpu-runs).

On Windows, error 1455 means system commit capacity is exhausted. Check
**Task Manager > Memory > Committed** and page file capacity as well as VRAM.
v2 reports this guidance in failed model status and logs; a failed primary
embedding load releases partial allocations before trying the INT8 fallback.
See [startup memory diagnostics](docs/V2.md#startup-memory-diagnostics).

The GPU profiles in `configs/` now select the same measured Harrier/Qwen
reranker stack. `cpu-only` and `gpu-minimal` remain explicit alternative
profiles with different models and retrieval behavior.

Run the [retrieval gate](eval/README.md) against the same indexed corpus,
queries, and model settings as your baseline. Then exercise repeated searches
and file changes during a scoped soak, recording health latency, queue depth,
rescan counts, and errors. Keep private queries and raw logs outside public Git.
Synthetic smoke success establishes operation; it does not establish retrieval
quality or a production latency guarantee.
Validation reports recorded model-load failures even when a concurrent status
snapshot still shows the model loading.

Run the repeatable live checks with the installed `mainframe` command:

```bash
mainframe validate --output ~/mainframe-validation/smoke-01
mainframe validate --output ~/mainframe-validation/soak-01 --duration 1800
```

Each run requires a **new directory outside Git**. It creates a synthetic corpus,
separate state, and an authenticated daemon on an unused port. Your model,
chunking, and search settings apply; indexing scope is replaced with the scratch
project, and contextual API calls, consolidation, and NLI are disabled. Models
use the configured cache and may download missing weights.

Checks cover authentication, browser origins, search/citations, watcher changes,
capture filtering, restart reuse, rescan bounds, clean shutdown, and v1 isolation.
The runner identifies its daemon with a fresh launch nonce, including in Windows
virtual environments where the Python launcher and interpreter have different PIDs.
`--duration` adds an observation window; mandatory checks can extend the total
runtime. `report.json` records timings, health probes, counts, and failures;
`config.json` contains a generated token and daemon logs may contain local paths.
Keep the entire output private and audit any summary before sharing it. See
[the v2 guide](docs/V2.md#repeatable-live-validation) for report details.

## v1 MCP tools

| Tool | Purpose |
|------|---------|
| `search` | Hybrid semantic+keyword search, reranked. Returns structured JSON (`{results, confidence, guidance}`); each result carries `rerank_score` (higher = better — the field to trust), char **and 1-based line** citation spans, and a truncation flag. |
| `capture_memory` | Write an immutable, secret-scrubbed session note under `<repo>/research/sessions/` and index it. |
| `consolidate_sessions` | Merge pending session captures into one curated `research/memory/` note (newer facts win), surface contradictions, archive the raw logs. |
| `sync_index` | Incremental sync: ingest new/modified knowledge files, prune deleted ones, compact the index. |
| `status` / `reload_config` | Health (models, VRAM, index compaction, pending captures) and hot config reload. |
| `ingest_file` / `delete_file` / `list_files` | The rest of the lifecycle. |
| `build_raptor` | Optional RAPTOR tier-1 summary clusters (off by default — hurts precision on our corpus). |

Query technique matters: **specific technical terms** (proper nouns, function names, exact config keys) rerank ~0.99; natural-language questions rerank ~0.05. See `docs/MAINFRAME_QUERY.md`.

## Default model stack

| Role | Model | Encoding |
|------|-------|----------|
| Embedder | microsoft/harrier-oss-v1-0.6b | Native BF16, 1,024 dimensions; pinned revision |
| Reranker | Qwen/Qwen3-Reranker-4B | INT8, final-token yes/no projection, 2,048-token budget |

The daemon does not load consolidation or NLI models. The existing legacy v1
features remain separate. Changes to an embedding model require a matching
encoding contract and an offline index rebuild; changing only a model name
does not make an existing index compatible.

## Privacy stance

- Serving is **fully local** — no network calls in the query or ingest path.
- Model weights download from Hugging Face on first run; that is the only default network activity.
- One **opt-in** exception: `contextual.enabled` sends document content to the Anthropic API at *ingest time only* (Anthropic's contextual-retrieval method, via Haiku with prompt caching). It is off by default and degrades to plain chunks on any failure.
- Session captures are secret-scrubbed (regex denylist: cloud keys, tokens, private keys, connection strings, env-style assignments) before touching disk, and the raw capture lane is designed to be gitignored.

## Development

```bash
python -m pytest tests/           # GPU-free suite (fakes + real LanceDB) — no GPU, no models, no network
python -u eval/evaluate_v1.py --live # v1: evaluate your existing index
python -u eval/evaluate.py --daemon # v2: evaluate through the running daemon
```

CI runs the GPU-free suite on Linux/Python 3.11 and checks clean wheel installs
on Windows and Linux with Python 3.10 and 3.14. Installation checks use CPU-only
PyTorch, import every shipped runtime module outside the checkout, load both
versions' configuration, and verify CLI entry points without downloading models.
Each matrix job builds a source distribution, builds its wheel, and verifies
that installed runtime versions match package metadata. The stable
`package-install` check requires every matrix job to pass.

To repeat the installation check locally (downloads dependencies):

```bash
python -m pip wheel . --no-deps --wheel-dir dist
python scripts/check_install.py --wheel-dir dist
```

Use a directory containing exactly one wheel. The checker creates and removes
its own temporary virtual environment; source-checkout imports cannot satisfy it.

The eval measures against your local corpus and GPU — **it cannot validate a PR by itself** (see the ground rules in [`eval/results.template.md`](eval/results.template.md)); retrieval-affecting changes get re-measured on the maintainer's corpus before merge.

Log every experiment in `eval/results.md` following `eval/results.template.md` — including the failures (on our corpus that graveyard includes RAPTOR-in-index, RRF fusion, doc-prefix embedding, seq-cls reranker conversions, and per-category instructions; retrieval leaderboards inverted against local measurement four separate times). Numbers are corpus-specific: treat the harness as the method and measure on your own corpus.

## Contributing

This public repository is the primary development home. PRs from humans and coding agents merge into public `main` after review and validation. **Read [CONTRIBUTING.md](CONTRIBUTING.md) and complete the [public push audit](docs/PUBLIC_AUDIT.md) before pushing.** Keep non-public data in the private repository. Tests are GPU-free by design; retrieval-affecting changes get measured on the eval harness before landing.

## License

MIT — see [LICENSE](LICENSE).
