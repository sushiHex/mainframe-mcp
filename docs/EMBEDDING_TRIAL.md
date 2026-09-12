# Small Embedding Trial and Adoption Plan

Harrier has been promoted to the default stack on public main after the
[context correction](RERANKER_CONTEXT.md). The plan and results below are the
historical embedding-adoption record; keeping the former Qwen embedder is no
longer required. All Qwen 4B INT8 reranker references below describe the fixed
historical control. The later [reranker comparison](RERANKER_COMPARISON.md)
selects Qwen 4B NF4 as the preferred shared-GPU configuration after its
production scoring replay and installed synthetic daemon smoke passed.

Prepared September 11, 2026. **Status at original comparison: Nemotron completed the optimized, paced
120-query comparison; Harrier and Voyage also completed it with lower memory
and no aggregate quality regression against Qwen. Harrier is the selected
private-trial finalist. Its production native encoder also completed the
47-query working check with no aggregate regression and 45.6% lower peak CUDA
allocation than Qwen. Experimental v2 support is implemented and the 15-minute
daemon soak passed. A fresh 20-question challenge showed no Harrier regression,
but both models missed one critical passage at reranking. The normal, unpaced
daemon trial passed with operator-reported smooth interaction; displayed-frame
timing remains unvalidated.** Earlier attempts
were refused by conservative policy before any model loaded. Those refusals
did not measure Nemotron's requirements. Use [the probe commands](../eval/README.md#guarded-embedding-probe)
to measure actual loading and encoding. See the [measured probe results](MODEL_COMPARISON.md#paced-nemotron-embedding-probe).
The sustained comparison now uses the same supervisor. Windows denied the
PresentMon tracing request, so visible presentation measurement remains pending;
the live soak and interaction verdicts are recorded in the measured results.
Keep model defaults and the live index unchanged during evaluation.

Follow-up: the [shared token-budget fix](RERANKER_CONTEXT.md#measured-follow-up-2026-09-11)
recovers the hidden passage and gives Harrier all five critical matches, with
first-place ranking tradeoffs. The comparisons above retain their original
reranker baseline; the challenge is now a known regression set.

Delayed heartbeats pause further forwards until the permit is fresh, with waits
recorded separately. No estimated-memory admission gates, allocation caps, or
fixed trial deadlines are imposed.
Live heartbeats use the inherited worker pipe, avoiding Windows permit-file
sharing races; the owner retains worker and reservation cleanup responsibility.
An expired or missing advisory claim is replaced and recorded before the next
permit. Reservation API errors and an unconfirmed replacement still stop work.

## Decision to make

Find the smallest practical embedding stack that preserves useful retrieval
and desktop responsiveness. Harrier is the private-trial finalist; Voyage's
smaller allocation peak came with slightly higher reservation and a first-place
relevance loss that Harrier avoided. Both preserve aggregate public hit rates.
Published benchmark scores alone cannot establish a winner on this corpus.

The completed embedding comparison used Qwen3 Reranker 4B INT8 throughout.
Changing the reranker, output dimension, quantization, and embedder together
would have obscured which change helped. Rerankers were evaluated separately
after this comparison.

## Candidates and footprint evidence

The figures below are **checkpoint tensor payload**, calculated from the
publisher's Hugging Face safetensors metadata as parameter count times dtype
bytes divided by 2^30. They exclude activations, allocator caches, CUDA context,
the reranker, and desktop allocations. They are not measured runtime VRAM.

| Candidate | BF16 tensor payload (GiB) | Trial priority |
|---|---:|---|
| [Nemotron 3 Embed 1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | 2.125 | Local reference candidate; 1.141B parameters |
| [Harrier OSS v1 0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | 1.110 | First challenger; 596M parameters, MIT |
| [Voyage 4 Nano](https://huggingface.co/voyageai/voyage-4-nano) | 0.645 | Second challenger; 346M parameters, Apache-2.0 |
| [Harrier OSS v1 270M](https://huggingface.co/microsoft/harrier-oss-v1-270m) | 0.499 | Reserve for a tighter budget |
| [Granite Embedding 311M Multilingual R2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) | 0.581 | Reserve with a different encoder architecture |

Microsoft reports multilingual MTEB v2 scores of 69.0 and 66.5 for the two
Harriers. NVIDIA reports 71.04 on **MMTEB Retrieval** for Nemotron; these are
different aggregates and must not be ranked against one another. Voyage's
published asymmetric results using hosted document embeddings do not establish
quality for a fully local Nano/Nano index. Test local query and document encoding.

The first three snapshots are pinned for the proposed comparison:

| Model | Revision |
|---|---|
| Nemotron 1B | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` |
| Harrier 0.6B | `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee` |
| Voyage Nano | `67fabc9bef010dabc5f6024aa1b1b6b93410426f` |

Other credible options were checked:

- [Jina v5 text small/nano](https://huggingface.co/jinaai/jina-embeddings-v5-text-small)
  offer strong compact-model results. Their published license is CC-BY-NC-4.0;
  keep them optional for a compatible use case. Retrieval requires the trained
  task adapter or the publisher's merged retrieval checkpoint. Base-model
  parameter metadata alone can omit adapter weights.
- [Perplexity embed v1 0.6B](https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b)
  publishes about 2.22 GiB of FP32 weight tensors. Its INT8 output vectors do
  not mean INT8 model weights. A BF16 trial would be a separately validated
  precision change. Normalize its output correctly for the existing float-vector
  store; do not silently treat integer outputs as the current embedding contract.
- [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma) is a smaller
  on-device option, with a 2K context limit and gated checkpoint access. Its
  [card](https://huggingface.co/google/embeddinggemma-300m) supports BF16/FP32,
  not FP16 activations. Keep it as an optional compact baseline.
- [Nomic v2 MoE](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) has
  475M total parameters despite 305M active parameters, and a 512-token limit.
  Its memory estimate must count all experts; truncation would add a confound.
- [Nemotron NVFP4](https://huggingface.co/blog/nvidia/nemotron-3-embed-wins-rteb)
  targets Blackwell. Its advertised acceleration does not apply to the RTX 3090.

Keep the first wave to Nemotron, Harrier 0.6B, and Voyage Nano. Expand only if
the results leave a useful quality/memory tradeoff unresolved. Smaller vectors
primarily reduce index storage; they do not remove transformer weights.

## Stage 0: reuse evidence before spending GPU time

The CPU-only preflight rechecked all 1,030 document hashes and all 120 query IDs
in the existing complete reports. Original and optimized full-Qwen reports
match in candidate lists, returned results, and metrics for every query.

Nemotron preserved each of the 20 Mainframe queries' measured quality metrics.
On SciFact, three queries lost reciprocal rank and three gained. Top-three
relevance lost two queries and gained one: the net change was 90/100 to 89/100.
Both lost queries were already missing from the candidate pool, so reranking
alone cannot recover them. Keep this regression review visible.

These results established the historical quality references. The subsequent
complete paced run reproduced every Nemotron candidate list, result, score,
and metric with final-token reranking at 8.205 GiB peak CUDA allocation.
Raw per-query review and input hashes remain outside public Git.

## Stage 1: prove the resource controls without models

The short probe now has one evaluation supervisor that owns one worker, its telemetry, and its
reservation. A batch gate controls when the worker may submit its next forward
pass. Keep this machinery under `eval/`; reuse the existing model factory hooks.
It does not add a production scheduler or a machine-wide service. The sustained
comparison now uses the same supervisor for embedding and fixed-reranker forwards,
with unchanged batches and separate active, pacing, and elapsed timings.

Current probe policy:

| Control | Policy |
|---|---|
| Allocations | No PyTorch or Windows Job Object memory cap |
| GPU and host memory | Measure actual use; no estimated-headroom admission cutoff |
| Duration | No fixed trial deadline; cancellation remains available |
| Pacing | After each synchronized forward pass, idle for at least three times its duration, with a 250 ms minimum |
| Telemetry | Sample resource state every 250 ms; missing or stale telemetry prevents further work |

The probe records `memory_policy: observe` in both reports. Its 7 GiB
cooperative claim is an estimate for coordination and never caps allocations.
The process group ensures descendant cleanup if the supervisor crashes;
see Microsoft's [Job Object ownership contract](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject).

Pacing and telemetry do not establish desktop responsiveness. Measure the
actual workload; do not substitute an invented startup allowance for evidence
that a model fits or fails. Honor explicit operator limits when supplied.

On cancellation, worker failure, or lost monitoring, stop the owned worker and
release its claim. The supervisor covers loading as well as inference. Actual
OOMs fail the experiment without retrying at different INT8 batch sizes.
Memory readings alone do not terminate the trial. Reserve through `vram-mcp`,
but treat its claims as coordination.

GPU-free tests cover observation without memory/duration cutoffs, uncapped CUDA
and Windows worker allocation, stale telemetry, cancellation, crash cleanup,
reservation release, and unchanged batch inputs/order under pacing. Verify that
an interrupted run cannot produce a passing report or a reusable partial index.

Measure actual presentation timing from a visible 60 Hz probe alongside normal
typing and scrolling. Proposed acceptance: p95 frame interval at most 33.3 ms
and at most twice the healthy idle baseline, with no gaps above 100 ms.
Reported interaction lag fails the trial even if counters pass. Unavailable
presentation evidence means responsiveness is unvalidated; an HTTP health
probe or CPU heartbeat cannot substitute for it.

## Stage 2: short contract and resource probes

Run one candidate at a time, first embedder-only, then with the fixed reranker.
Use a frozen eight-query smoke fixture covering short questions, code/config
terms, empty-input handling, Unicode, and near-2,048-token text. Check finite
vectors, expected dimensions, normalization within a recorded precision tolerance,
correct query/document routing, and all parameters on CUDA.

Nemotron must retain its saved prompts, bidirectional attention, and mean pooling.
Harrier must select its saved `web_search_query` prompt and leave documents
unprefixed. Voyage must use its saved query/document routes and pooling; review
and pin its custom loading code before enabling it. Use SDPA on this host and
hold dependency versions fixed. A generic `encode_query` call is insufficient
when the required prompt has a different saved name.

Record loading, indexing, querying, and release separately. Use unchanged INT8
reranker batches and fail the experiment on OOM recovery or parameter offload.
Passing this smoke permits the next stage; it does not establish retrieval quality.

## Stage 3: controlled retrieval and desktop trial

1. Freeze source hashes, model revisions, environment versions, prompts, resource policy,
   pacing, corpus, and query order. Keep 256-token chunks, 35% overlap, candidate
   pool 20, context cap 2,048, and three returned chunks.
2. Run the existing 120-query set against separate embedding indexes. Reuse an
   index only when its complete embedding identity and corpus hashes match.
   Use optimized Qwen3 Reranker 4B INT8 in every new comparison.
3. Review each candidate before starting the next. Compare candidate recall,
   MRR@3, hit@1/3, and answer-passage hits per query and per dataset. Preserve
   negative results and report paired uncertainty; equal means do not imply parity.
4. Repeat only the finalist to check stability. For comparable full-Qwen timing
   and memory, use a dedicated window and record actual resource use. Existing
   full-Qwen results remain labeled historical until that control is available.
5. Before adoption, freeze a separate working-corpus holdout with labeled expected
   sources and critical queries. Evaluate Qwen and the finalist on that holdout,
   then run a paced 15-minute scoped soak with searches and file changes.

The completed working check used 47 existing labels on 64 frozen files, with
4,819 chunks and matched Qwen/Harrier configurations. It retained all baseline
top-three expected passages and found three more. These tuning labels are not
an unseen-query holdout; stronger independent queries remain future adoption
evidence. A subsequent 20-question source-grounded challenge retained all 19
baseline passage matches and every per-query metric; both models missed the
same fifth critical passage despite retrieving it into their candidates. Its
frozen labels remain unchanged. The daemon soak uses a separate synthetic scope to test lifecycle
behavior: 87 searches, three mutation/capture cycles, zero restart re-embedding,
and 455 successful health checks. It passed with clean owned-worker cleanup.
See [the full measured scope](MODEL_COMPARISON.md#production-native-harrier-on-a-working-sample).

Record active inference time, pacing time, end-to-end latency, and throughput
separately. Record PyTorch allocated/reserved peaks, total GPU use, host private
commit, system headroom, and presentation timing. A successful query is not a
resource pass. A run interrupted for pressure remains incomplete.

## Acceptance and opt-in implementation

Use separate verdicts for **quality**, **resources**, and **comparability**.
The conservative quality gate requires no aggregate regression on the frozen
sets and review of every changed answer; critical working-corpus queries must
retain their expected evidence. A small-loss tradeoff remains a review decision,
not an automatic pass. Do not adjust tolerances after seeing a candidate's results.

Target at least 25% lower full-stack peak CUDA allocation than optimized full
Qwen, without worse active p95 latency, while meeting the resource and desktop
gates. Nemotron's public allocation reduction is 40.1%; native Harrier reduced
the private comparison's allocation by 45.6%. Harrier's observed active p95 was
lower, but background activity was uncontrolled, including a GPU-free test run
during its early queries. Controlled latency and desktop presentation evidence
remain necessary for a default change. A challenger
tops Nemotron only if comparable measurements show lower runtime memory with
acceptable quality and latency. If a reference or telemetry is missing, report
that comparison as inconclusive.

The following v2 support is implemented for opt-in trials. Availability of the
experimental profile is not a declaration that every adoption check passed:

1. **Explicit encoding contract.** Add an opt-in model-owned encoding policy to
   `src/mainframe/core/embedder.py`, with validated dtype and pinned revision.
   Reuse Sentence Transformers' query/document routes in BF16,
   with no inherited Qwen category prefixes or manual last-token fallback.
   Preserve the existing `embed`, `embed_query`, `dimension`, and `info` interface
   and `Models.invoke` ownership. Unsupported settings fail before allocation.
   Harrier selects its saved `web_search_query` prompt. Custom architecture
   loading remains confined to the reviewed evaluation adapter.
2. **Index identity.** Extend `Store` fingerprint checks to cover the new encoding
   policy, revision, precision, prompt behavior, normalization, and context cap.
   A changed embedding space blocks both reads and writes even at equal vector
   dimensions. Preserve untouched legacy defaults' fingerprint behavior; native
   encoding requires a newly built, fully identified index. Never fill missing
   native identity fields by assuming compatibility.
3. **Opt-in rollout.** Add a scoped preset and dependency compatibility check.
   Test routing, normalization, failed-load cleanup, drift, offline rebuild,
   clean installation, and the full GPU-free suite. Repeat the finalist trial
   through production code before switching a private deployment. Update README,
   v2 docs, and measured results; retain existing default models.

Build the candidate index offline in a separate state directory. Keep the old
configuration and index intact. Rollback stops the candidate, releases its
models, and restarts the old configuration against its matching index. Never
load both stacks to perform the switch. Audit exact outgoing commits and all
publication artifacts before any public push.

## Voyage custom code review

Reviewed the complete `modeling_qwen3_bidirectional.py` at revision
`67fabc9bef010dabc5f6024aa1b1b6b93410426f` before execution. It imports Torch and
Transformers, constructs a Qwen3 backbone plus a learned projection, disables
causal attention, and builds the padding-aware bidirectional mask. It contains
no filesystem, network, subprocess, or dynamic-evaluation operations. Saved
Sentence Transformers modules provide mean pooling and normalization, with
distinct query/document prompts. This is a loading review, not a quality result.

The trial checks the sole Python file's SHA-256
`f4340347ce92a764e6ce6dc76fb4e412a6ab876dd33e61a9ed4de7595c697728`,
the local revision, `auto_map`, and saved prompts before enabling custom code.
Execution remains offline in the isolated worker. The upstream implementation
is used unchanged; library compatibility must still pass the live probe.
Source: [pinned model repository](https://huggingface.co/voyageai/voyage-4-nano/tree/67fabc9bef010dabc5f6024aa1b1b6b93410426f).
