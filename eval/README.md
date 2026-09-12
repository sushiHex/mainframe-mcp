# Retrieval Evaluation

Evaluate retrieval against your own corpus and expected answers. The GPU-free
pytest suite checks behavior; this harness measures retrieval quality.

## Select the implementation

```bash
# v2: running daemon, or configured index when it is stopped
python -u eval/evaluate.py --live

# v2: require the daemon and avoid loading models here
python -u eval/evaluate.py --daemon

# v1: original stdio server's index
python -u eval/evaluate_v1.py --live
```

Use the same configuration and query set for comparisons. Set `MAINFRAME_CONFIG`
for the v2 deployment being evaluated. The legacy v1 harness expects the default
`~/.claude/mainframe` state directory. Run comparison workers sequentially and
record unrelated model allocations in the resource telemetry.

## Shared desktop GPU runs

Treat desktop responsiveness as an acceptance criterion for live GPU work.
Use the default Harrier profile for interactive trials; schedule sustained
large-model sweeps for a dedicated window. Neither profile has a desktop
responsiveness guarantee. Follow this procedure before the next GPU run:

- Check host commit headroom and total GPU memory, including model-loading
  peaks and other applications. Coordinate through `vram-mcp` when available;
  reservations do not enforce allocation limits or GPU execution shares.
- Observe memory use and pace work on an active desktop. Insert idle intervals
  between unchanged batches. Honor user-specified limits; do not invent memory
  admission cutoffs or allocator caps from unmeasured startup estimates.
- Observe desktop responsiveness throughout loading, indexing, and queries,
  alongside host commit, GPU memory, and engine utilization. Low CPU usage,
  successful queries, or a single healthy memory sample do not establish
  acceptable desktop performance.
- Preserve INT8 batch composition when adding pacing. Smaller batches can
  change scores and require retrieval evaluation. Record pacing separately
  from inference latency and keep it consistent across comparisons.

Use `run_trial.py` to supervise sustained comparisons as well as short probes.
The earlier external monitor's emergency commit threshold was insufficient
to protect desktop responsiveness. The current supervisor measures resources
and paces work; it does not infer responsiveness from memory thresholds. Do not
change machine-wide settings or terminate unrelated workloads to make a benchmark fit.

## Guarded embedding probe

`run_trial.py` implements resource observation, short encoding probes, and paced retrieval
of the [trial plan](../docs/EMBEDDING_TRIAL.md). It currently supports Windows
with one physical NVIDIA GPU. Install `.[eval]` in the supervisor environment
and make `vram-mcp` available on PATH. Keep the model environment isolated;
the tested dependency pair is Sentence Transformers 5.4.1 / Transformers 5.7.0.

```powershell
# Read-only host/GPU measurement; no model imports, worker, or reservation.
python eval/run_trial.py --preflight-only --output D:/scratch/preflight-01

# One pinned, already-downloaded model; output must be new and outside Git.
python eval/run_trial.py --candidate nemotron --snapshot D:/scratch/models/c0c9fea93ea424587517f2c59e20db9f1d6bf615 --python D:/scratch/venv/Scripts/python.exe --output D:/scratch/nemotron-probe-01
```

The probe observes memory without imposing estimated-headroom cutoffs,
PyTorch/Windows allocation caps, or a trial-duration ceiling. Its 7 GiB
`vram-mcp` claim is advisory coordination, not an allocation limit. Prior claims
are recorded; the runner does not evict other workloads. `--preflight-only`
reports `observed` and the actual memory counters, without deciding whether
a model fits. The model load and probe establish that experimentally.

The supervisor samples every 250 ms and sends heartbeats over the worker's
inherited pipe. A dedicated reader retains the latest complete message; no
repeated filesystem opens are needed for admission. A permit older than one second prevents
another forward until a fresh permit arrives. This pauses scheduling without
discarding a run over a delayed heartbeat; monitoring-reader failures and owner
exit still terminate the worker. The worker waits at least three times the previous forward's
synchronized duration, with a 250 ms minimum; batch contents/order stay intact.
Cancellation, failed telemetry, a reservation API error, or a worker error
stops the owned worker. A Windows Job Object contains
its descendants and terminates them if the supervisor exits unexpectedly.
Claims expire after 60 seconds unless renewed. Create an empty `cancel` file
inside the run directory to stop the trial.
If a claim has expired or disappeared, the supervisor acquires and records a
replacement before publishing another permit. An expired advisory claim alone
does not invalidate completed work; an unconfirmed replacement still stops it.
The initial `permit.json` is diagnostic; live heartbeats use the pipe. Reader
EOF, malformed messages, or a changed nonce fail the worker. Pipe publication
keeps only the latest pending heartbeat and cannot block the resource monitor.

The synthetic fixture checks eight query/document inputs, including empty,
Unicode, code, and a long input exercising the 2,048-token cap. Checks cover
dimensions, finite values, normalization (absolute norm tolerance 0.01), BF16,
and CUDA placement. Harrier uses its saved `web_search_query` prompt. Voyage
uses its reviewed, hash-checked custom module at the pinned revision; modified
or additional Python files are refused before model imports. See the
[source review](../docs/EMBEDDING_TRIAL.md#voyage-custom-code-review).

Inspect `guard-report.json`, `resources.jsonl`, and `worker/report.json`.
Exit 2 means refusal or abort; only an authenticated complete worker report
can produce `completed`. That status is an encoding/resource result: retrieval
quality and visible desktop presentation remain separate, unvalidated gates.
The [Nemotron embedder-only probe passed](../docs/MODEL_COMPARISON.md#paced-nemotron-embedding-probe),
with nine measured forwards and successful release. Its host-headroom minimum
was 2.03 GiB; completion does not imply that the desktop had comfortable headroom.

## Paced retrieval comparison

Add a frozen manifest, full private configuration, scratch index, and label to
run the real v2 indexing/search pipeline under the same supervisor:

```powershell
python eval/run_trial.py --candidate nemotron --manifest D:/scratch/manifest.json --config D:/scratch/nemotron.json --index D:/scratch/index-nemotron --label nemotron-paced --python D:/scratch/venv/Scripts/python.exe --output D:/scratch/nemotron-paced
python eval/summarize_comparison.py D:/scratch/reference/report.json D:/scratch/nemotron-paced/worker/report.json > D:/scratch/paired.json
```

Use `--candidate harrier` or `--candidate voyage` with its pinned local model
in the configuration. Omit `--candidate` to exercise the configured production
embedder: either an explicitly configured legacy control or the
[native encoding contract](../docs/V2.md#native-embedding-contract). For native
Harrier, use the pinned revision, `quantize: false`, and saved
`query_prompt: "web_search_query"`; do not add experimental adapter flags.
`embedding_implementation` distinguishes production from experimental runs. Keep
the same Qwen3 Reranker 4B INT8 configuration across embedding comparisons.
Retrieval claims 15 GiB cooperatively; this is never an allocation ceiling.
The index marker includes the corpus hash, embedding configuration, model
revision, native prompts, precision, and normalization. Incomplete indexes
cannot be reused. Query changes do not invalidate a completed corpus index.
For a legacy marker, supply `--index-manifest` with the exact original manifest;
the runner verifies its hash, corpus, and encoding without rewriting the marker.

Both embedding and reranking forwards are paced without changing batch order.
Failures abort before production recovery can alter the experiment. Query
reports record `forward_s`, `pacing_s`, `monitor_wait_s`, `seconds` (elapsed),
and `active_s` (elapsed minus pacing and monitoring waits, including CPU overhead). Pacing can affect
GPU clocks; active time is not a prediction of unpaced latency. Compare
identically paced runs when making performance claims. Reports also include
source hashes, dependency versions, load/index/release timings, and memory peaks.
Use the maximum across load/index, `warmup_memory`, and `warm_memory` for the
recorded lifecycle peak; older reports without `warmup_memory` omit that phase.
The summary tool requires complete reports with identical corpus/query identity
and shows paired bootstrap intervals plus every gained/lost query.

## v2 modes

| Option | Behavior |
|---|---|
| Default / `--live` | Uses the running daemon; otherwise opens the configured v2 index, or builds a temporary index if absent |
| `--daemon` | Requires the daemon; scores HTTP search with full chunk text |
| `--db PATH` | Opens an existing populated v2 index; refuses v1 schemas |
| `--rebuild` | Replaces `eval/.eval-lancedb` with a deterministic rebuild |
| `--corpus-manifest PATH` | Restricts a rebuild to existing non-session files in a v1 manifest |
| `--dump-candidates PATH` | Saves raw candidate identities for deterministic comparisons |
| `--min-score NUMBER` | Exits 1 below the supplied composite-score baseline |

`--db` and `--rebuild` refuse while the configured daemon is running. Explicit
`--in-process` and candidate dumps load their own models; stop other GPU-heavy
work first. Compare discovery with a v1 manifest using
`python -u eval/corpus_identity.py path/to/.manifest.json`. The gate fails when
a still-existing manifest file is missing from the v2 scan; new files and
content changes are reported separately.

## Queries, results, and measurement

Copy `eval/test_queries.example.json` to gitignored `eval/test_queries.json`
and supply queries, expected files, and optional expected text. The example
set demonstrates the format; its scores are not a corpus baseline.

Runs save local JSON under `eval/results/`. Review with
`python eval/history.py --best` or `python eval/history.py --last 5`.

The composite is **40% MRR + 30% hit@1 + 30% text_match@1**. Both v2 paths
use the same top-three rank window. Daemon `hit_at_5` is `null` because ranks
four and five are not requested. Text matching uses complete chunks.

Change one variable at a time and record corpus identity, configuration,
models, timing, and repeated runs against the same baseline. Follow
`results.template.md`; keep corpus-specific logs in gitignored `results.md`.
Query sets, candidate dumps, and results can expose private paths and project
details. Audit any proposed publication before uploading it.

## Compare experimental local models

Follow the [small-embedding trial plan](../docs/EMBEDDING_TRIAL.md) for the
Nemotron/Harrier/Voyage shortlist, resource-control prerequisites, acceptance
criteria, and conditional production implementation. Use `run_trial.py` for
supervised probes and retrieval comparisons on a shared Windows desktop.

`compare_models.py` evaluates a frozen corpus through v2's real chunker, store,
model registry, reranking, and search-result shaping. It does not start a daemon
or read your live configuration. Supply a complete configuration JSON and keep
all inputs, indexes, and output outside Git. Download and pin model snapshots
first; run with `HF_HUB_OFFLINE=1` for a reproducible comparison.

The manifest contains `files` with unique `id`, absolute `path`, and content
`sha256`, and `queries` with unique `id`, `set`, `query`, and a nonempty
`relevant` list of document IDs. Optional `expected_text` checks the answer
passage. Freeze the queries before running any model. Changed document hashes
or missing relevance labels fail before GPU initialization. Different document
IDs cannot point to the same canonical source path.

```bash
python eval/compare_models.py --manifest /scratch/manifest.json \
  --config /scratch/baseline.json --index /scratch/index-small \
  --output /scratch/baseline-before --label baseline-before
```

Each output directory must be new. Index reuse requires matching canonical file
paths, IDs, content hashes, and embedding configuration; questions and file order
may change. Reports retain the full query-manifest hash separately. A failed
partial build requires a new index. For an older completed index, also pass
`--index-manifest /scratch/original-manifest.json` to prove its corpus identity.
Choose a separate index when
changing embedders. Rerun the original baseline after candidates and compare
candidate identities and ranks, not only aggregate scores.

Experimental adapters are opt-in: `--bf16-reranker` loads a trained CrossEncoder
in BF16 with SDPA; `--native-embedder` uses Sentence Transformers' model-owned
query/document prompts and pooling in BF16 and requires `embedder.quantize=false`.
They require CUDA and reject model parameter offload. These flags belong to the
evaluation harness, not production configuration. The tested environment uses
Sentence Transformers 5.4.1 and Transformers 5.7.0; use a separate environment
for these experiments.

Reports separate loading, indexing, five-query warmup, and warm search. Latency
includes embedding, hybrid retrieval, reranking, and result shaping in-process;
it excludes HTTP/MCP transport. CUDA allocated/reserved peaks cover PyTorch's
allocator, not total driver memory. Monitor host commit separately. Scores use
the API's three chunk slots: duplicate documents occupy ranks but earn relevance
credit only once. This is not a deduplicated BEIR leaderboard score. Record
hardware, dependency versions, snapshot revisions, and corpus selection alongside
the report; a scoped benchmark does not establish private-corpus quality.
