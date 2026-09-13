# Local Model Comparison

Measured on September 10–11, 2026.

Harrier native BF16 is the default embedder on public main. The later
[reranker comparison](RERANKER_COMPARISON.md) selects the pinned Qwen3-Reranker-0.6B BF16 `<Document>` contract from a
separate source-grounded evaluation. The older control measurements remain
historical. The tables below retain their historical INT8 reranker baselines
and experimental status at measurement time.

## Scope and method

This evaluation is based on the v2 retrieval code at public commit
`d44994313e95af97e63e770971d058906a498aa7` and
[`eval/compare_models.py`](../eval/compare_models.py). It measures model choices
in scratch indexes, with the working changes described below. Each report
records source hashes. The initial tables use a frozen public benchmark;
the separate working-corpus check is reported later.

The frozen corpus contains 1,024 documents from [BEIR SciFact](https://github.com/beir-cellar/beir)
and six eligible Mainframe documents at that commit, producing 1,621 chunks.
For SciFact, sort test query IDs by SHA256 of
`mainframe-model-eval-v1/query/<id>` and select the first 100. Include every
labeled relevant document, then fill to 1,024 by sorting remaining document IDs
by SHA256 of `mainframe-model-eval-v1/doc/<id>`. Twenty Mainframe questions and
expected answer snippets were written and frozen before scoring any model.
Raw manifests, queries, indexes, and reports remain outside Git.

All runs use 256-token chunks, 35% overlap, a candidate pool of 20, heading
injection, and three returned chunks. Embedding context is capped at 2,048
tokens. These historical trials preserve their recorded Qwen prompts, INT8 batching, and document clamps.
Ettin uses its trained CrossEncoder modules in BF16 with SDPA; Nemotron uses
model-owned query/document prompts, attention, and mean pooling in BF16.
Experimental adapters are confined to the harness. The private Harrier check
uses the production native encoder instead.
Stack measurements cover embedding and reranking; contextual enrichment,
consolidation, and NLI were disabled.

The host is Windows with an RTX 3090 24 GiB, driver 596.49, and Python 3.14.3.
The isolated environment uses PyTorch 2.11.0+cu128, Sentence Transformers 5.4.1,
Transformers 5.7.0, bitsandbytes 0.49.2, and LanceDB 0.30.0. It reuses the host's
CUDA dependencies through a virtual environment with system packages enabled.
Downloads finish before evaluation; model loading then runs offline. All model
parameters must be on CUDA for a run to qualify.

Warm latency follows five warmup queries and includes embedding, hybrid search,
reranking, and result shaping, excluding HTTP/MCP transport. P95 uses linear
interpolation. Memory figures below are PyTorch CUDA allocator peaks, which
exclude driver/context allocations. Host private bytes and commit headroom are
sampled separately (one-second historical samples; 250 ms in paced trials).
Fresh-process loading is not a
controlled cold-disk benchmark. Background GPU allocation varied between runs;
timings are observations on this shared desktop, not universal throughput claims.

Historical allocator peaks cover the recorded load/index and measured-query
phases. The harness formerly reset the warmup peak without retaining it;
the current runner records `warmup_memory` separately before that reset.
The older tables are not guaranteed whole-lifecycle allocator maxima. Paced
runs' host/GPU telemetry covers the worker lifetime, including warmup.

The first full Qwen 8B/4B run reported 25.156 GiB of peak allocator reservation
on a nominal 24 GiB GPU and 39.022 GiB of sampled process private memory.
Minimum sampled system commit headroom was 0.082 GiB. Reservation is not a
measurement of physical residency; these baseline timings include this host's
memory-management behavior. Spot checks reported no driver spill, but do not
establish its absence throughout the run. Starting with the historical full
baseline repeat, an external Windows monitor used a stop condition of less than
2 GiB commit headroom for three consecutive one-second samples. Subsequent
paced trials removed that estimated-headroom cutoff and observe actual use.
No page-file setting was changed by the evaluation runner.

## Rerankers with Qwen3 Embedding 0.6B

All four runs reuse one index. Each candidate received the same ordered
candidate list for all 120 queries. Restoring Qwen reproduced every candidate
list and all 120 top-three chunk rankings exactly.

| Reranker | SciFact MRR@3 | SciFact hit@1 / hit@3 | Mainframe answer@1 / answer@3 | SciFact p50 / p95 (s) | Peak allocated / reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| Qwen3 0.6B INT8, before | 0.8217 | 79% / 87% | 14/20 / 18/20 | 1.322 / 1.520 | 4.388 / 9.189 |
| Ettin 150M BF16 | 0.7983 | 74% / 87% | 16/20 / 19/20 | 0.420 / 0.520 | 1.967 / 2.684 |
| Ettin 400M BF16 | 0.8150 | 78% / 86% | 17/20 / 19/20 | 0.509 / 0.610 | 2.705 / 3.744 |
| Qwen3 0.6B INT8, after | 0.8217 | 79% / 87% | 14/20 / 18/20 | 1.260 / 1.467 | 4.381 / 9.184 |

Both Ettin models reduce latency and memory on this workload. Neither meets a
strict no-observed-regression gate across both query sets: SciFact declines
while Mainframe answer matching improves. These are tradeoffs, not established
universal quality upgrades. Paired query bootstrap intervals for the SciFact
MRR differences include zero: 150M −0.0667 to +0.0183; 400M −0.0550 to +0.0433
(10,000 resamples, seed 20260910).

MRR and hit rates score document relevance in the API's actual three chunk
slots. Repeated documents consume ranks but receive relevance credit once.
Mainframe answer matches additionally require the expected text in a relevant
document's returned chunk. These twenty checks are literal passage-match
proxies, not a human assessment of every valid answer or paraphrase. Scores
are not comparable with deduplicated BEIR
leaderboards, the repository's composite gate, or another corpus/harness. The
composite gate's own verdict on the default pairing is recorded in
[COMPACT_RERANKERS.md](COMPACT_RERANKERS.md#composite-gate-on-the-maintainer-corpus).

## Embedders with Qwen3 Reranker 4B INT8

Each embedder builds its own index from the same frozen documents. The
reranker and search settings stay fixed.

| Embedder | SciFact MRR@3 | SciFact hit@1 / hit@3 | Mainframe answer@1 / answer@3 | SciFact p50 / p95 (s) | Peak allocated / reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| Qwen3 8B INT8, before | 0.8367 | 79% / 90% | 17/20 / 19/20 | 5.620 / 6.806 | 15.304 / 25.156 |
| Nemotron 3 Embed 1B BF16 | 0.8367 | 80% / 89% | 17/20 / 19/20 | 3.613 / 4.472 | 9.814 / 14.459 |
| Qwen3 8B INT8, after | 0.8367 | 79% / 90% | 17/20 / 19/20 | 5.654 / 6.301 | 15.297 / 25.109 |

The full baseline repeat reproduced all 120 candidate lists, top-three chunk
rankings, and reported scores exactly. Its sampled peak private memory was
36.575 GiB, compared with Nemotron's 23.433 GiB. Minimum sampled commit
headroom was 0.170 GiB and 0.257 GiB respectively; transient headroom remained
tight. The monitor's stop condition did not trigger during the full repeat.

Nemotron matches MRR and Mainframe passage matches while reducing allocated
CUDA peak by 36% and median SciFact search latency by 36% against the first
full baseline. Indexing took 48 seconds versus approximately 382 seconds,
excluding model loading. One fewer SciFact query has a relevant top-three
result, while one more has a relevant first result. The paired SciFact MRR
bootstrap interval is −0.025 to +0.025, with three improved and three regressed
queries. Equal aggregate MRR does not mean every answer is preserved.

This is a promising candidate for a corpus-specific trial, not an automatic
default replacement. The current daemon cannot reproduce this adapter by
changing only `embedder.model`: the model-owned query/document encoding route
must also be supported. Use the evaluation harness for this comparison.

## Reduce memory with the existing Qwen models

The native Qwen reranker formerly projected every token into its vocabulary,
then kept only the final token's yes/no logits. Both v1 and v2 now request
[`logits_to_keep=1`](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen3).
Attention still sees the entire prompt. Instructions, clamps, candidate order,
INT8 batches, and score conversion stay the same.

The isolated small-model control preserved all 120 candidate lists, rankings,
and four-decimal reported scores exactly. Compared with the repeated small
baseline, allocated CUDA peak fell from 4.381 to 2.523 GiB, reserved peak from
9.184 to 2.896 GiB, and sampled process private peak from 13.719 to 7.547 GiB.
SciFact p50/p95 changed from 1.260/1.467 to 1.191/1.413 seconds. This is a clear
memory reduction; the smaller latency difference should not be generalized
beyond this workload.

A final run exercised the changed production v2 code directly with the full
Qwen 8B/4B INT8 stack and the existing frozen index:

| Full Qwen stack | SciFact p50 / p95 (s) | Peak allocated / reserved (GiB) | Sampled peak private memory (GiB) |
|---|---:|---:|---:|
| Original code, repeated baseline | 5.654 / 6.301 | 15.297 / 25.109 | 36.575 |
| Final-token projection | 2.146 / 2.809 | 13.699 / 14.695 | 33.742 |

All 120 candidate lists, top-three rankings, and reported scores matched
exactly. SciFact MRR remained 0.8367 and Mainframe answer@1 remained 17/20.
Minimum sampled commit headroom was 3.914 GiB; the stop condition did not fire.
This code change is adopted in both native Qwen backends without changing
model names or quantization defaults. GPU measurements cover v2; both backends
have GPU-free regression coverage for scores and INT8 input order.

The model-choice tables above use the original reranker implementation.
This final control isolates the code change. Establish a new baseline before
comparing model choices against the optimized implementation.

An additional FP16 control with the same small Qwen models and final-token
projection was rejected as a replacement for INT8. SciFact p50/p95 improved
to 0.679/0.894 seconds, but MRR fell to 0.8150 and Mainframe document hit@1
fell from 85% to 75%. Peak allocated/reserved memory was 4.315/4.902 GiB.
Quantization and batch composition can affect retrieval; faster inference
does not establish quality parity. At the time of this measurement, the default
remained INT8. The later [reranker comparison](RERANKER_COMPARISON.md) records
the later Qwen 0.6B BF16 selection and its evaluation limits. A reranker change
reorders existing candidates and does not require re-embedding.

## Paced Nemotron embedding probe

On September 11, the isolated Nemotron BF16 probe completed with no estimated
memory admission cutoff, allocator cap, Windows job memory cap, or fixed trial
deadline. This tests the embedder alone; the full reranker stack and retrieval
quality are separate evaluations.

All eight synthetic document/query cases passed finite-value, 2,048-dimension,
normalization, BF16, and CUDA-placement checks. The long case exercised a real
2,048-token input. The document batch stayed at eight; eight individual queries
followed, for nine observed transformer forwards. Maximum norm error was 0.00334
against the declared 0.01 tolerance.

| Measurement | Observed value |
|---|---:|
| Peak CUDA allocated / reserved | 3.978 / 5.291 GiB |
| Peak worker process commit | 9.297 GiB |
| Minimum whole-GPU free memory | 15.704 GiB |
| Minimum system commit headroom | 2.031 GiB |
| Model loading after library imports | 4.371 s |
| Encoding wall time / synchronized forward time / pacing | 7.471 / 1.953 / 5.381 s |
| Model release | 0.352 s |

The worker exited and its cooperative VRAM claim was released. Host headroom
includes other applications and is not the worker's private-memory measurement.
No presentation timing was collected, so desktop responsiveness remains
unvalidated. These embedder-only figures are not comparable to the earlier
full-stack peaks as a percentage saving.

Live runs exposed two instrumentation issues: Sentence Transformers 5.4 calls
the Hugging Face `forward` method directly, bypassing hooks there, and its inputs
are mapping objects rather than plain dictionaries. Pacing now observes the
enclosing Sentence Transformers module and records mapping input shapes. A
Windows file-sharing race during permit replacement also has a bounded retry.
Earlier incomplete reports are retained outside Git; only the complete rerun
is summarized here. No OOM fallback, smaller-batch retry, or CPU offload was used.

## Compact challenger probes

The same eight-document/eight-query BF16 fixture passed for both challengers,
with nine paced forwards each, native prompts/pooling, 2,048-token truncation,
finite normalized vectors, CUDA placement, and successful release.

| Embedder | Peak CUDA allocated / reserved (GiB) | Peak process commit (GiB) | Forward / pacing time (s) |
|---|---:|---:|---:|
| Nemotron 1B | 3.978 / 5.291 | 9.297 | 1.953 / 5.381 |
| Harrier 0.6B | 1.803 / 2.230 | 6.193 | 1.911 / 4.764 |
| Voyage Nano | 1.842 / 2.012 | 6.010 | 0.960 / 3.479 |

These are embedding-only probes, not retrieval-quality or full-stack results.
Harrier used 1,024-dimensional vectors; Nemotron and Voyage used 2,048. Voyage's
smaller weights did not produce the lowest allocated peak on the long-input
fixture: activations and pooling/projection also consume memory. Timing comes
from sequential shared-desktop runs and is not a general speed guarantee.

The first optimized, paced Nemotron retrieval run stopped after 117 queries
when a 1.58-second telemetry gap exceeded the one-second permit age. Completed
queries matched the historical run, but the run remains incomplete. Workers now
pause new forwards until fresh telemetry arrives; broken permits, cancellation,
monitor-reader failure, and owner exit still stop them. Monitoring waits are
reported separately from pacing. The full rerun reuses only the completed,
identity-checked scratch index; the interrupted report is retained locally.
The next attempt exposed a Windows permit-file sharing failure. Live heartbeats
now use the owned parent/worker pipe, with separate tests for delivery, clean
interpreter exit, broken-pipe cleanup, and end-to-end supervisor completion.

## Optimized, paced Nemotron retrieval

The complete supervised rerun finished all 120 queries with the optimized
Qwen3 Reranker 4B INT8. Every candidate list, returned ranking, reported score,
and quality metric matched the historical Nemotron run exactly. SciFact MRR@3
remained 0.8367, hit@1/3 was 80%/89%, and Mainframe answer@1/3 was 17/20 and
19/20. Against optimized Qwen, the paired SciFact MRR difference was zero
(95% bootstrap interval −0.025 to +0.025, 10,000 resamples, seed 42).
Top-three relevance gained one query and lost two; both losses occurred in
candidate retrieval. This remains a quality tradeoff.

| Measurement | Observed value |
|---|---:|
| Peak CUDA allocated / reserved | 8.205 / 8.813 GiB |
| Peak worker process commit | 20.676 GiB |
| Minimum whole-GPU free memory | 10.190 GiB |
| Minimum system commit headroom | 28.958 GiB |
| SciFact active p50 / p95 | 2.607 / 3.275 s |
| SciFact elapsed p50 / p95 | 10.420 / 12.998 s |
| Query pacing / monitoring waits, all 120 queries | 907.851 / 2.311 s |

Allocated peak was **40.1% lower** than the optimized Qwen reference's
13.699 GiB. The Qwen timing reference was unpaced, so these timings do not
establish a matched speed comparison. Active time excludes deliberate pacing
and monitor waits but includes CPU work; pacing can also change GPU clocks.
The final run reused its completed, identity-checked index. Its near-zero
index opening time is not a fresh indexing measurement.

The earlier fresh paced build produced 1,621 rows in 327.091 seconds, including
261.974 seconds of pacing and 56.587 seconds of synchronized embedding forwards.
Its later query interruption does not erase that indexing measurement, but the
interrupted run remains incomplete. Final model release left 0.008 GiB allocated;
the worker exited and its reservation was released. Desktop presentation was
not measured. Raw reports, failures, and source archives remain outside Git.

## Optimized, paced Harrier retrieval

Harrier 0.6B completed the same 120 queries with its saved `web_search_query`
prompt and the optimized Qwen3 Reranker 4B INT8. Its peak CUDA allocation was
**6.970 GiB**, 15.1% below paced Nemotron and 49.1% below optimized Qwen.
Peak reservation was 7.777 GiB and peak process commit was 19.692 GiB.

| Metric | Qwen reference | Nemotron | Harrier |
|---|---:|---:|---:|
| SciFact hit@1 / hit@3 | 79% / 90% | 80% / 89% | 81% / 91% |
| SciFact MRR@3 | 0.8367 | 0.8367 | 0.8467 |
| Mainframe answer@1 / answer@3 | 17/20 / 19/20 | 17/20 / 19/20 | 17/20 / 19/20 |
| SciFact active p50 / p95, paced (s) | Unpaced reference | 2.607 / 3.275 | 2.263 / 3.017 |

Every Mainframe query retained its measured quality metrics. No SciFact
top-one or top-three relevance match was lost against Qwen; one top-three
match and two top-one matches were gained. Two existing relevant answers
moved from second to third place. Candidate recall stayed at 98%, with two
gains and two losses on queries whose final top-three output remained misses.
The paired MRR delta was +0.0100 (95% bootstrap interval −0.0033 to +0.0267).
Harrier has no observed aggregate quality regression on these frozen sets;
this does not establish unseen-corpus superiority or identical answers.

The review covered all 59 changed query/ranked-source records and all ten
metric changes. Every previously returned relevant SciFact passage remained
verbatim within the new top three. Three Mainframe queries lost supplementary
passages from an expected source; their first result retained the actual token,
private-storage, or read-only endpoint guidance. Unlabeled distractors were
reviewed by title, not individually adjudicated for scientific relevance.

Elapsed SciFact p50/p95 was 8.899/11.469 seconds. The 120 queries included
816.645 seconds of pacing and no stale-telemetry waits. Mainframe active
p50/p95 was 2.160/2.646 seconds, slightly higher than Nemotron's 2.073/2.459;
latency improvements were not uniform across groups. Runs were sequential on
a shared desktop, with uncontrolled background workloads.

An earlier Harrier attempt stopped at 80 queries after its cooperative claim
expired or disappeared. The rerun reproduced all 80 candidate lists, results,
scores, and metrics exactly, then completed the full set. The supervisor now
replaces missing claims before issuing another permit and treats an already
absent claim as released. It does not evict other workloads. The complete
rerun reused only the finished index; model release left 0.008 GiB allocated.
Minimum observed GPU free memory was 8.299 GiB and system commit headroom was
23.124 GiB. Desktop responsiveness remains unvalidated.

## Optimized, paced Voyage retrieval and finalist choice

Voyage Nano also completed all 120 queries using its reviewed, pinned custom
module. SciFact hit@1/3 was 81%/91%, MRR@3 was 0.8483, and nDCG@3 was 0.8458.
Every Mainframe query retained its measured metrics. Against Qwen, Voyage
gained three first-place relevance matches and lost one; it gained one
top-three match with none lost. Two queries lost reciprocal rank. Its paired
MRR delta was +0.0117 (95% bootstrap interval −0.0083 to +0.0333).

| Recorded measurement | Nemotron | Harrier | Voyage |
|---|---:|---:|---:|
| CUDA allocated / reserved peak (GiB) | 8.205 / 8.813 | 6.970 / 7.777 | 6.714 / 7.854 |
| Process commit peak (GiB) | 20.676 | 19.692 | 20.176 |
| SciFact active p50 / p95 (s) | 2.607 / 3.275 | 2.263 / 3.017 | 2.536 / 3.404 |
| SciFact elapsed p50 / p95 (s) | 10.420 / 12.998 | 8.899 / 11.469 | 9.975 / 13.229 |

Voyage's fresh index completed in 324.189 seconds including pacing. Its query
phase included 889.879 seconds of deliberate pacing and no monitor waits.
Minimum whole-GPU free memory was 6.289 GiB and system commit headroom was
15.484 GiB. Model release left 0.008 GiB allocated and the supervisor completed
cleanly. Background allocations varied, so the latency differences are
observations rather than a controlled throughput ranking.

**Take Harrier into the private trial.** Voyage saved 0.256 GiB (3.7%) of
allocated peak relative to Harrier, but reserved 0.076 GiB more. Both achieved
the same aggregate hit rates, while Harrier retained every Qwen top-one and
top-three match on this set. Harrier also uses the standard native loader;
Voyage requires its reviewed custom architecture. This is a practical finalist
choice, not proof that Harrier is universally better. Keep Voyage available in
the isolated evaluation harness. The production native encoder and experimental
Harrier preset were then checked against Qwen on a separate working sample.

## Production native Harrier on a working sample

Both fresh, paced runs completed 47 existing labeled queries across 64 files
and 4,819 chunks. The scope included every supported labeled source plus frozen
distractors. Three stale labels lacking their expected source text were excluded
before either run; all remaining expected snippets occur in the configured
chunks. This is an existing tuning set, not a new unseen-query holdout or a
full-corpus evaluation. Raw queries, source details, and records stay private.

Harrier used the production native encoder, pinned BF16 revision, saved query
prompt, and an index with its native identity recorded in LanceDB. Chunking,
context, batches, Qwen3 Reranker 4B INT8, and search settings matched the control.

| Measurement | Qwen 8B INT8 | Harrier 0.6B BF16 |
|---|---:|---:|
| Document hit@1 / hit@3 | 28/47 / 39/47 | 30/47 / 41/47 |
| Expected passage@1 / passage@3 | 23/47 / 34/47 | 27/47 / 37/47 |
| Candidate hit | 44/47 | 45/47 |
| MRR@3 / nDCG@3 | 0.7021 / 0.7186 | 0.7411 / 0.7614 |
| CUDA allocated / reserved peak (GiB) | 14.329 / 15.035 | 7.800 / 8.756 |
| Worker process commit peak (GiB) | 29.148 | 20.754 |
| Fresh index, including pacing (s) | 1,729.455 | 257.154 |
| Index forwards / pacing (s) | 432.270 / 1,291.132 | 56.871 / 195.904 |
| Active query p50 / p95 (s) | 2.603 / 3.258 | 2.150 / 3.079 |
| Elapsed query p50 / p95 (s) | 9.643 / 12.488 | 8.898 / 11.981 |

Both builds performed 630 embedding forwards on identical chunks. Harrier used
**45.6% less peak CUDA allocation**, 28.8% less peak worker commit, and completed
the paced index 6.7 times faster. These allocator peaks include load, index,
warmup, and measured queries. Both released to 0.008 GiB allocated and exited
cleanly, with no monitoring waits, parameter offload, or recovery retries.
Background applications varied; the GPU-free suite also overlapped the start
of Harrier queries. Latencies are observations, not a controlled causal speed
comparison or a prediction of unpaced production latency.

Every aggregate quality metric improved. No baseline candidate, top-three
document, or top-three expected-passage match was lost. Two first-place sources
moved lower; one expected passage moved from first to third. The review covered
all 23 changed ranked-source records and all eight metric changes, including
the full returned text of those demotions and newly found sources. One gain is
a title-only keyword match: these broad labels do not prove answer completeness.
Paired MRR delta was +0.0390 (95% bootstrap interval −0.0284 to +0.1170).

The observed aggregate quality and memory checks pass. Keep stronger unseen
queries and actual desktop presentation as outstanding adoption evidence.
The 15-minute synthetic daemon soak passed with real production native models:
87 searches, three create/edit/rename/delete and capture-isolation cycles,
restart reuse with zero re-embedded rows, and clean shutdown. All 455 health
checks succeeded (p95 4.8 ms, maximum 13.6 ms). Both sessions loaded and released
their model pair; release left 0.016 GiB allocated before the owned worker
exited. Its cooperative reservation was released. This small synthetic scope
tests lifecycle behavior, not retrieval quality across the full working corpus
or visible desktop presentation.

## Fresh passage-based challenge

Twenty new questions and exact supporting excerpts were frozen before either
model ran, including five critical checks. The questions were authored by the
implementation agent from 15 documents in the same 64-file corpus; they were
not used in the earlier model comparisons. This is a fresh, source-grounded
challenge, not an independent user-authored holdout or a full-corpus assessment.
All excerpts were verified to occur in configured chunks before evaluation.

Both runs reused their completed indexes, with zero document-embedding forwards.
The runner now separates corpus identity from query-manifest identity. Legacy
reuse requires the exact original manifest and a matching corpus/encoding;
historical completion markers remain untouched. Both reports retain the new
full manifest hash. Pinned models, query order, five warmups, context, chunking,
candidate pool, and pacing matched. No test suite ran during either comparison;
ordinary background applications remained uncontrolled.

| Measurement | Qwen 8B INT8 | Harrier 0.6B BF16 |
|---|---:|---:|
| Document hit@1 / hit@3 | 20/20 / 20/20 | 20/20 / 20/20 |
| Expected passage@1 / passage@3 | 16/20 / 19/20 | 16/20 / 19/20 |
| Candidate hit | 20/20 | 20/20 |
| Critical expected passage@3 | 4/5 | 4/5 |
| Active query p50 / p95 (s) | 3.094 / 3.988 | 2.694 / 3.837 |
| Elapsed query p50 / p95 (s) | 12.165 / 15.193 | 10.568 / 14.347 |
| Load/warmup/query CUDA allocated / reserved peak (GiB) | 13.546 / 14.418 | 7.023 / 7.896 |
| Worker process commit peak (GiB) | 29.148 | 19.694 |

Every query's scored metrics matched. Review covered all eight changed ranked
passage records, including the complete added/removed text; differences were
in supplemental evidence or ordering while labeled answer evidence survived.
Both missed the same critical excerpt despite retrieving its chunk into the
20 candidates. Its 4,719-character chunk places the answer at character 3,950,
beyond the unchanged reranker's 3,000-character document clamp; the scorer does
not see that evidence. This shared truncation limitation is not a Harrier
regression. Labels and denominators
remain unchanged. The no-regression check passes; the stricter requirement that
all critical passages appear does not. Ceiling document scores on this small
source-grounded set do not establish general equivalence.

## Normal daemon desktop trial

A separate private copy of the same 64 files exercised the unmodified daemon
and its default production model factories, without forward hooks or inference
pacing. It built 4,819 chunks in 83.061 seconds, reused every row on the next
rescan, served 25 searches, and indexed then removed a new file through the
normal writer. Twenty searches retained full result records; five exercised
the file-change phase. No job/model failure or device-recovery event occurred.
All 432 health checks succeeded (p95 6.3 ms, maximum 284.8 ms).

Peak CUDA allocation/reservation was 7.034/8.279 GiB. Minimum observed whole-GPU
free memory was 10.628 GiB, maximum system commit 61.016 GiB, and GPU utilization
reached 99%. These are observed values, not admission limits. The worker owned
its daemon and visible interaction window; shutdown released its models to
0.016 GiB allocated, then exited and released its cooperative reservation.

The operator reported **smooth** interaction at idle, during active searches,
and after the trial, with 46 text/scroll input events recorded by the probe.
Typed text was not stored. This supports usability for this scoped session.
Windows denied PresentMon's ETW tracing request; permissions and machine settings
were not changed. WPF rendering callbacks were recorded, but are not measurements
of displayed frames. No frame-presentation threshold or broad desktop guarantee
is claimed. See [PresentMon's capture documentation](https://github.com/GameTechDev/PresentMon/blob/main/README-ConsoleApplication.md).

## Validation and decision

The original model comparisons in this document use the character-clamped reranker.
The subsequent [token-budget regression check](RERANKER_CONTEXT.md)
restores the hidden critical passage without changing indexes. It records
first-place ranking tradeoffs separately; the original measurements and labels
here are preserved.

The integrated public checkout's GPU-free suite passed: **534 passed, 1 skipped**. The projection
regression tests failed against both original backends, and the JSON-setting
test failed against the original configuration parser before their fixes.
Complete public comparisons and the paired 47-query working checks retain
separate warmups. Their workers stopped and released their VRAM reservations.

An earlier wheel with the same 54 runtime modules and native extra passed installation in a fresh CPU-only
environment: all 54 runtime modules imported from that environment, both CLI
entry points loaded, native configuration identity passed, and v1 refused the
native configuration without initializing CUDA. That installation resolved
PyTorch 2.14.0+cpu, Sentence Transformers 6.0.1, Transformers 5.17.0, and
packaging 26.3. This checks installation/import compatibility; the GPU results
above use the separately recorded trial versions.

Harrier native BF16 is now the standard v2 embedder. The token-budget follow-up
recovers all five critical answers and all 20 top-three passage matches on the
known challenge. Promotion accepts the documented first-place tradeoffs; that
reused challenge is no longer a fresh holdout. Voyage passes the aggregate
public check but offers a smaller incremental memory saving. These conclusions
use the historical INT8 reranker. The later
[reranker comparison](RERANKER_COMPARISON.md) selects Qwen3-Reranker-0.6B
BF16 from a later, separate evaluation. The historical INT8 baselines in this
document remain historical.

The live synthetic daemon soak and normal daemon trial passed. The operator
reported smooth interaction during the validation session; displayed-frame
timing and independent user-authored holdout evidence remain incomplete. These
limits constrain the claims, rather than requiring retention of the former
default. Local installation now uses Harrier and its matching index. Raw
records, downloaded weights, and machine configuration remain outside Git.

## Model provenance

| Model | Pinned revision |
|---|---|
| [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418` |
| [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | `e61197ed45024b0ed8a2d74b80b4d909f1255473` |
| [Ettin Reranker 150M](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1) | `025501c4e0f9bbeb4c5b198318e0089ff061cc14` |
| [Ettin Reranker 400M](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1) | `5dca36282a5d85f368d2544002513a29159b4c9e` |
| [Qwen3 Embedding 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` |
| [Qwen3 Reranker 4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` |
| [Nemotron 3 Embed 1B BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` |
| [Harrier OSS v1 0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee` |
| [Voyage 4 Nano](https://huggingface.co/voyageai/voyage-4-nano) | `67fabc9bef010dabc5f6024aa1b1b6b93410426f` |
