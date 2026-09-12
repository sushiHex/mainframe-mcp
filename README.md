# mainframe-mcp

A **100% local, GPU-accelerated semantic knowledge base** for [Claude Code](https://claude.com/claude-code) and any other [MCP](https://modelcontextprotocol.io/)-compatible agent. It indexes the markdown knowledge scattered across your repos (`research/`, `docs/`, and root context files like `CLAUDE.md` / `AGENTS.md`), and gives coding agents high-precision hybrid search over it — plus an ambient memory pipeline that captures session knowledge and consolidates it into curated notes over time.

Everything runs on your own GPU. Nothing leaves your machine at query time.

## Architecture

![mainframe-mcp architecture](docs/architecture.svg)

*Three loops on one machine: ingest builds a disposable dual index; a read-only query path retrieves hybrid and reranks with the model's own logits; an ambient-memory loop consolidates immutable session captures into curated notes where newer facts win, then re-indexes them. Files are the source of truth; the vector index is derived and disposable.*

## What it does

- **Hybrid retrieval, reranked** — vector nearest-neighbor (Qwen3-Embedding-8B, INT8) + Tantivy full-text search, merged and re-scored by a cross-encoder reranker. Markdown-header-aware 256-token chunking, adaptive per-query-type instruct prefixes, section-heading injection at rerank time.
- **Native Qwen3-Reranker backend** — the reranker scores candidates via the model's yes/no logits with a task instruction (the model-card method), not a bolted-on classification head. On our eval this beat bge-reranker-v2-m3 by **+26% composite / +33% hit@1**.
- **Ambient memory (mem0-inspired)** — agents write immutable session-capture files (secret-scrubbed, gitignored lane); a consolidation pass merges them into curated memory notes where **newer facts supersede stale ones**, with NLI-backed contradiction surfacing and anti-bloat guardrails. Keep private notes in your private knowledge corpus. Files are the source of truth — the vector index is derived and disposable.
- **Ops-hardened** — read-only search hot path, atomic manifest writes with crash recovery, index compaction + FTS refresh after ingest batches, poison-file-resilient syncs, path-traversal validation, and a layered secret-scrub denylist on every write path.
- **Measured, not vibed** — an eval harness with a regression gate (`eval/evaluate.py --min-score`), an experiment-log discipline (`eval/results.template.md`), and a GPU-free test suite (fakes + real LanceDB) that runs in CI.

The final-token projection experiment preserved the full attention context
and all 120 measured rankings while
reducing the full Qwen stack's measured query-phase CUDA peak to **13.70 GiB**.
Nemotron with that optimized reranker completed the same comparison at
**8.20 GiB**: Mainframe passage matches stayed unchanged, while SciFact top-three
recall fell from 90% to 89%. See the [measured comparison](docs/MODEL_COMPARISON.md)
for scope, timings, and retrieval tradeoffs. Model and precision defaults remain unchanged.
Harrier completed the same set at **6.97 GiB**, retaining Mainframe passage
matches and improving SciFact top-three matches to **91/100**. It is the
selected opt-in candidate: Voyage reached **6.71 GiB** allocation
with the same aggregate hit rates, but slightly higher reservation and one lost
first-place match. On a separate working sample, production native Harrier
retained all 34 baseline top-three expected-passage matches and found three
more, using **7.80 versus 14.33 GiB** peak CUDA allocation. That check used
47 existing labels across 64 files; it is not an unseen-query evaluation.

In the original 20-question, source-grounded challenge frozen before either run,
both models found the expected document for every query and the labeled passage
for 19. Harrier lost no baseline passage. Both missed one critical passage at
reranking; [the results](docs/MODEL_COMPARISON.md#fresh-passage-based-challenge)
retain that limitation and the test's authorship and scope.

The [paced evaluation runner](eval/README.md#paced-retrieval-comparison) owns
worker cleanup and a cooperative `vram-mcp` reservation, observes host/GPU memory,
and separates active work, deliberate pacing, and monitoring waits. It imposes
no estimated-memory admission cutoff, allocation cap, or fixed trial deadline.
Expired cooperative claims are replaced before further work is permitted.
The [adoption plan](docs/EMBEDDING_TRIAL.md) separates trial support from a
default-model change. A normal, unpaced daemon trial completed on the 64-file
copy with **7.03 GiB** peak CUDA allocation; the operator reported smooth
interaction during searches. Displayed-frame timing remains unmeasured.

Fresh questions can reuse a completed evaluation index when its corpus and
encoding identity match. The [evaluation guide](eval/README.md) documents
verification of older indexes through their original manifest.

Native Qwen reranking now budgets documents in tokens, replacing the former
3,000-character cutoff. Long paragraphs and preserved code blocks can use the
existing 2,048-token context, with room reserved for the assistant scoring
suffix. Ordinary prompt tokens, candidate order, and INT8 batches are preserved;
scores can change in batches containing longer documents. Existing indexes and
citations remain valid. See [context handling](docs/RERANKER_CONTEXT.md).
For ordinary Harrier use, start with an explicit project scope and keep a
[private question journal](docs/V2.md#everyday-harrier-evaluation) that distinguishes
real user questions from scripted probes and known regression cases.

## Requirements

- Python 3.10+ (developed on 3.14). **3.11–3.13 have the broadest binary-wheel coverage**; on very new interpreters some deps (lancedb, pydantic-core) may not have Linux wheels yet — pin to 3.13 there if `pip install` fails building from source.
- NVIDIA GPU with ~16 GB VRAM for the full stack (developed on an RTX 3090 24 GB); smaller presets in `configs/`
- PyTorch with CUDA — on Windows the default pip wheel is CPU-only:
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```

## Install

The **2.0.0a1 preview** includes the v1 server and opt-in v2 daemon in one
distribution. See [preview installation and rollback](docs/PREVIEW_RELEASE.md)
for isolated environments, audited artifacts, migration, and capability limits.
The preview version does not switch existing MCP registrations to v2.
Source distributions include configuration examples, guides, and the complete
GPU-free test harness; wheels contain the runtime packages.

```bash
pip install -e .            # core: lancedb, sentence-transformers, transformers, mcp, tantivy
pip install -e .[test]      # + pytest for the GPU-free suite
```

Configuration is optional — sensible defaults apply with no config file. To customize, copy `config.example.json` to `~/.claude/mainframe/config.json` (or start from a preset in `configs/`); most knobs hot-reload via the `reload_config` tool, model changes need a restart. Model weights download from Hugging Face on first run (~15 GB total for the full stack) and stay resident.

Register with Claude Code (use the interpreter of the environment you installed into — replace `python` with its full path if you use venvs/conda):

```bash
claude mcp add mainframe -- python -u -m mainframe_mcp.server
```

### Using with other MCP hosts

Nothing about the server is Claude-specific — it speaks plain MCP over stdio. Point any host at `python -u -m mainframe_mcp.server` and tune it with environment variables:

| Env var | Effect |
|---------|--------|
| `MAINFRAME_CONFIG` | Path to the config file (portable override; default `~/.claude/mainframe/config.json`) |
| `MAINFRAME_PREWARM=1` | Load models at startup instead of lazily on the first tool call. **Off by default** — an always-on gateway sharing a GPU should not claim ~13 GB of VRAM merely because a client connected. Set this for a dedicated, interactive setup that wants instant first-tool response. |
| `MAINFRAME_READ_ONLY=1` | Expose only `search` / `list_files` / `status` (no delete / consolidate / bulk-index / RAPTOR). Smaller tool schema, no accidental mutation — ideal when injecting Mainframe into ordinary conversations. Run maintenance from a separate, full-access invocation. |

Example (a generic gateway config; adjust to your host's schema):

```yaml
mcp_servers:
  mainframe:
    command: "python"
    args: ["-u", "-m", "mainframe_mcp.server"]
    env:
      MAINFRAME_CONFIG: "/path/to/mainframe/config.json"
      MAINFRAME_READ_ONLY: "1"      # recall-only in everyday chat
      # MAINFRAME_PREWARM: "1"      # opt in only for a dedicated instance
    timeout: 180
    connect_timeout: 60
```

## v2 daemon preview

The optional `mainframe` command runs one background service for multiple MCP
clients. It watches files, reconciles missed changes, and exposes HTTP MCP at
`http://127.0.0.1:7433/mcp`; `/mcp/ro` exposes only search and status. Stdio-only
clients can use `python -u -m mainframe.adapters.mcp_stdio_shim`.

Configure `paths.include_projects` before running `mainframe serve`: an empty
list includes all eligible projects under `paths.repos_dir`. The daemon builds
its own `index.lancedb`, separate from v1's `.lancedb`. Index-setting drift is
reported explicitly; rebuilding requires stopping the daemon first.

v2 includes model recovery, immutable captures, and verified source citations.
Its memory workflow currently supports capture, indexing, and search. v1 remains
the default and retains consolidation, contradiction detection, and RAPTOR.
Both configuration loaders accept shared files containing v2 project-filter
lists; v1 preserves those lists without treating them as filesystem paths.
See [the v2 guide](docs/V2.md) for setup, commands, architecture, and recovery.

The experimental [Harrier preset](configs/v2-harrier.json) uses the
[native encoding option](docs/V2.md#native-embedding-contract):
model-owned query/document routes in BF16, a pinned revision, and a separately
identified index. Install `.[native]` in an isolated environment; v1 refuses
native configurations. Use a separate state directory and explicit project
scope; keep the old configuration and index for rollback. Normal daemon
inference is unpaced; this preset does not change the default models.

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

For a shared GPU, [gpu-shared.json](configs/gpu-shared.json) selects the 0.6B
Qwen3 embedder and native reranker. Copy it to a private configuration, add an
explicit project scope and separate state, and follow the [scoped trial guide](docs/V2.md#scoped-project-trial).
It disables consolidation, NLI, and contextual calls. Smaller models need their
own index and retrieval evaluation; this profile makes no quality-equivalence claim.

Run the [retrieval gate](eval/README.md) against the same indexed corpus,
queries, and model settings as your baseline. Then exercise repeated searches
and file changes during a scoped soak, recording health latency, queue depth,
rescan counts, and errors. Keep private queries and raw logs outside public Git.
Synthetic smoke success establishes operation; it does not establish retrieval
quality or a production latency guarantee.

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

## Model stack (default `configs/gpu-max.json`)

| Role | Model | VRAM |
|------|-------|------|
| Embedder | Qwen/Qwen3-Embedding-8B (INT8, 4096-dim) | ~9 GB |
| Reranker | Qwen/Qwen3-Reranker-4B (INT8, native logit scoring) | ~4.5 GB |
| Consolidator | Qwen/Qwen2.5-3B-Instruct (4-bit, lazy-loaded) | ~2.1 GB |
| Contradiction NLI | DeBERTa-v3-large-MNLI (lazy-loaded) | ~0.8 GB |

Swap any of them via config or env (`MAINFRAME_RERANKER_MODEL`, etc.). `BAAI/bge-reranker-v2-m3` remains a leaner, faster reranker fallback (~1.2 GB, ~3× lower search latency, lower quality).

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
