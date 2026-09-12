# Compact reranker comparison

Status: **Qwen3 Reranker 4B NF4 is the selected shared-GPU configuration.** Its
production scoring contract and installed synthetic whole-stack smoke passed.
This is a measured resource/quality tradeoff, not a claim of quality parity or
universal improvement.

## Method

This comparison isolates reranking behind Harrier 0.6B. The frozen input has
187 queries and the same 20 heading-rendered candidates per query: 100 SciFact
document-label queries, 20 public Mainframe queries, 47 working queries, and 20
known challenges. The latter three groups have passage labels. All queries stay
in the denominators; four relevant documents and four labeled passages are
absent from their candidate pools.

Every model used a 2,048-token limit and batches of eight. The Qwen control used
the accepted cached INT8 path. Qwen 0.6B, Ettin, and Nemotron used BF16 and SDPA.
Qwen preserved its assistant suffix and ranked FP32 `P(yes)`; R3 used its native
agent-skill instruction and `yes logit - no logit`; Ettin and Nemotron used
their released formats and raw logits. The NF4 run used the production `<Doc>`
prompt, saved order, double quantization, BF16 compute, SDPA, and
`use_cache=False`. A preregistered Qwen 0.6B follow-up changed only `<Doc>` to
the model card's `<Document>` marker.

Models ran alone through an owned, paced supervisor on an RTX 3090 after five
warmups. The control reproduced its saved results. Runtime versions were Python
3.14.3, PyTorch 2.11.0+cu128, Transformers 5.7.0, and Sentence Transformers
5.4.1.

## Retrieval quality

The compact table shows the top-three measures most relevant to this decision.
SciFact has document labels; the other groups have explicit passage labels.
Counts retain every query in the denominator, including cases whose labeled
document or passage is absent from the candidate pool.

| Model | SciFact doc@3 /100 | Mainframe passage@3 /20 | Working passage@3 /47 | Challenge passage@3 /20 | Critical passage@3 /5 |
|---|---:|---:|---:|---:|---:|
| Qwen 4B INT8 control | 91/100 | 19/20 | 37/47 | 20/20 | 5/5 |
| Ettin 400M BF16 | 87/100 | 19/20 | 37/47 | 18/20 | 3/5 |
| Ettin 1B BF16 | 87/100 | 18/20 | 38/47 | 18/20 | 3/5 |
| Nemotron 1B BF16 | 92/100 | 18/20 | 34/47 | 19/20 | 5/5 |
| Qwen 0.6B BF16 | 87/100 | 17/20 | 37/47 | 18/20 | 5/5 |
| R3 0.6B BF16 | 83/100 | 16/20 | 35/47 | 16/20 | 5/5 |
| Qwen 4B NF4 | 88/100 | 19/20 | 37/47 | 19/20 | 4/5 |
| Qwen 0.6B BF16, `<Document>` marker | 88/100 | 18/20 | 36/47 | 19/20 | 5/5 |

<details>
<summary>Detailed retrieval metrics</summary>

Document hits are `matches / all group queries`. MRR@3 and nDCG@3 range from
zero to one. Passage hits are shown only when explicit passage labels exist.

| Model | Group | N | Document hit@1 | Document hit@3 | MRR@3 | nDCG@3 | Passage hit@1 | Passage hit@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen 4B INT8 control | SciFact public | 100 | 81/100 | 91/100 | 0.8467 | 0.8468 | — | — |
| Qwen 4B INT8 control | Mainframe public | 20 | 18/20 | 20/20 | 0.9417 | 0.9565 | 17/20 | 19/20 |
| Qwen 4B INT8 control | Working sample | 47 | 29/47 | 41/47 | 0.7305 | 0.7535 | 26/47 | 37/47 |
| Qwen 4B INT8 control | Known challenge | 20 | 20/20 | 20/20 | 1.0000 | 1.0000 | 15/20 | 20/20 |
| Ettin 400M BF16 | SciFact public | 100 | 77/100 | 87/100 | 0.8133 | 0.8137 | — | — |
| Ettin 400M BF16 | Mainframe public | 20 | 18/20 | 20/20 | 0.9500 | 0.9631 | 17/20 | 19/20 |
| Ettin 400M BF16 | Working sample | 47 | 27/47 | 39/47 | 0.6950 | 0.7103 | 25/47 | 37/47 |
| Ettin 400M BF16 | Known challenge | 20 | 20/20 | 20/20 | 1.0000 | 1.0000 | 14/20 | 18/20 |
| Ettin 1B BF16 | SciFact public | 100 | 80/100 | 87/100 | 0.8300 | 0.8230 | — | — |
| Ettin 1B BF16 | Mainframe public | 20 | 16/20 | 20/20 | 0.8917 | 0.9196 | 15/20 | 18/20 |
| Ettin 1B BF16 | Working sample | 47 | 27/47 | 41/47 | 0.7057 | 0.7288 | 25/47 | 38/47 |
| Ettin 1B BF16 | Known challenge | 20 | 20/20 | 20/20 | 1.0000 | 1.0000 | 15/20 | 18/20 |
| Nemotron 1B BF16 | SciFact public | 100 | 80/100 | 92/100 | 0.8517 | 0.8539 | — | — |
| Nemotron 1B BF16 | Mainframe public | 20 | 19/20 | 20/20 | 0.9667 | 0.9750 | 18/20 | 18/20 |
| Nemotron 1B BF16 | Working sample | 47 | 31/47 | 38/47 | 0.7340 | 0.7338 | 25/47 | 34/47 |
| Nemotron 1B BF16 | Known challenge | 20 | 20/20 | 20/20 | 1.0000 | 1.0000 | 17/20 | 19/20 |
| Qwen 0.6B BF16 | SciFact public | 100 | 75/100 | 87/100 | 0.8017 | 0.8035 | — | — |
| Qwen 0.6B BF16 | Mainframe public | 20 | 16/20 | 19/20 | 0.8750 | 0.8946 | 15/20 | 17/20 |
| Qwen 0.6B BF16 | Working sample | 47 | 29/47 | 40/47 | 0.7199 | 0.7401 | 25/47 | 37/47 |
| Qwen 0.6B BF16 | Known challenge | 20 | 20/20 | 20/20 | 1.0000 | 1.0000 | 14/20 | 18/20 |
| R3 0.6B BF16 | SciFact public | 100 | 78/100 | 83/100 | 0.8017 | 0.7833 | — | — |
| R3 0.6B BF16 | Mainframe public | 20 | 16/20 | 18/20 | 0.8500 | 0.8631 | 14/20 | 16/20 |
| R3 0.6B BF16 | Working sample | 47 | 27/47 | 38/47 | 0.6738 | 0.6972 | 23/47 | 35/47 |
| R3 0.6B BF16 | Known challenge | 20 | 20/20 | 20/20 | 1.0000 | 1.0000 | 11/20 | 16/20 |
| Qwen 4B NF4 | SciFact public | 100 | 77/100 | 88/100 | 0.8150 | 0.8126 | — | — |
| Qwen 4B NF4 | Mainframe public | 20 | 18/20 | 20/20 | 0.9417 | 0.9565 | 17/20 | 19/20 |
| Qwen 4B NF4 | Working sample | 47 | 30/47 | 41/47 | 0.7447 | 0.7611 | 26/47 | 37/47 |
| Qwen 4B NF4 | Known challenge | 20 | 20/20 | 20/20 | 1.0000 | 1.0000 | 17/20 | 19/20 |
| Qwen 0.6B BF16, `<Document>` marker | SciFact public | 100 | 73/100 | 88/100 | 0.7967 | 0.7962 | — | — |
| Qwen 0.6B BF16, `<Document>` marker | Mainframe public | 20 | 18/20 | 19/20 | 0.9250 | 0.9315 | 16/20 | 18/20 |
| Qwen 0.6B BF16, `<Document>` marker | Working sample | 47 | 29/47 | 39/47 | 0.7163 | 0.7306 | 24/47 | 36/47 |
| Qwen 0.6B BF16, `<Document>` marker | Known challenge | 20 | 20/20 | 20/20 | 1.0000 | 1.0000 | 14/20 | 19/20 |

</details>

The compact models were not demonstrated upgrades. Ettin missed two critical
passages; Nemotron led SciFact at 92/100 but fell to 34/47 working passages; and
R3 had fewer top-three passage matches than Qwen 0.6B in every passage-labeled
group. Qwen 0.6B was the smallest Qwen tested, but paired review found two real
answer losses. Its `<Document>` follow-up recovered some counts yet retained a
root-cause mechanism loss, so it remains a footprint-first fallback.

NF4 reduced SciFact document hit@3 from 91/100 to 88/100 and total passage hit@3
from 76 to 75. Its critical miss was benign on review. Of three SciFact losses,
one dropped subject-specific evidence, one weakened scope/provenance, and one
returned an alternate paper that still refuted the claim. The decision accepts
these losses for the shared-GPU memory benefit; it does not call NF4 a quality
upgrade.

## CUDA memory and latency

Loaded and peak values are PyTorch allocator measurements from the reranker
worker. Reserved allocator cache is separate from allocated tensor memory.
They are not total GPU use for the desktop or embedding stack. Active latency
subtracts pacing and monitor waits.

| Model | Loaded allocated (GiB) | Loaded reserved (GiB) | Peak allocated (GiB) | Peak reserved (GiB) | Active p50 (s) | Active p95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 4B INT8 control | 4.113 | 4.275 | 7.951 | 9.283 | 2.274 | 4.525 |
| Ettin 400M BF16 | 0.737 | 0.830 | 1.286 | 1.840 | 0.196 | 0.486 |
| Ettin 1B BF16 | 1.941 | 2.051 | 2.833 | 4.010 | 0.325 | 1.065 |
| Nemotron 1B BF16 | 2.303 | 2.305 | 3.343 | 4.744 | 0.366 | 1.078 |
| Qwen 0.6B BF16 | 1.110 | 1.133 | 1.621 | 2.170 | 0.367 | 1.075 |
| R3 0.6B BF16 | 1.110 | 1.133 | 1.621 | 2.170 | 0.369 | 1.075 |
| Qwen 4B NF4 | 2.493 | 2.531 | 3.738 | 6.154 | 1.902 | 4.408 |
| Qwen 0.6B BF16, `<Document>` marker | 1.110 | 1.133 | 1.621 | 2.170 | 0.343 | 0.998 |

<details>
<summary>Model initialization and synchronized forward timings</summary>

Initialization includes model loading but filesystem cache state was not
controlled.

| Model | Initialization (s) | Forward p50 (s) | Forward p95 (s) |
|---|---:|---:|---:|
| Qwen 4B INT8 control | 11.506 | 2.258 | 4.493 |
| Ettin 400M BF16 | 1.355 | 0.179 | 0.458 |
| Ettin 1B BF16 | 3.071 | 0.308 | 1.036 |
| Nemotron 1B BF16 | 7.496 | 0.353 | 1.050 |
| Qwen 0.6B BF16 | 20.973 | 0.352 | 1.043 |
| R3 0.6B BF16 | 15.924 | 0.352 | 1.038 |
| Qwen 4B NF4 | 37.789 | 1.887 | 4.378 |
| Qwen 0.6B BF16, `<Document>` marker | 14.703 | 0.330 | 0.956 |

</details>

A separate no-cache probe covered 23 queries, including the largest batches.
All 460 raw scores, full rankings, and metrics exactly matched the cached
control. Peak allocation/reservation was 5.701/7.016 GiB, compared with
7.951/9.283 GiB for the full 187-query cached control. Because the probe was a
subset rather than a matched full-run memory experiment, those peaks do not
quantify a full-run saving. Paired active time was 75.610 seconds without cache
and 73.569 seconds with cache, so no speed improvement is claimed.

The existing warning surface uses a rounded top-one Qwen score below 0.3. The
historical INT8 control produced 11 warnings and NF4 produced 12: three newly
appeared and two disappeared. This is an operating-surface change, not evidence
that either score distribution is calibrated confidence.

## Reproducibility and license gate

Model cards are the primary sources for architecture, intended scoring, and
weight-license metadata. Public default candidates require commercially usable
weights. A model being license-eligible does not mean it has passed the quality
review.

| Model | Pinned revision | Weight license | Default license eligibility | Status |
|---|---|---|---|---|
| [Harrier OSS v1 0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) | `f9b9dc8d367d443f2479d27aa5d8d2850c0774ee` | MIT | Eligible | Fixed first stage |
| [Qwen3 Reranker 4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | Apache-2.0 | Eligible | Control complete |
| [Ettin Reranker 400M v1](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1) | `5dca36282a5d85f368d2544002513a29159b4c9e` | Apache-2.0 | Eligible | Complete; not a demonstrated upgrade |
| [Ettin Reranker 1B v1](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1) | `7d20e9baad17016fdf5549c08f69a2d7ca3e60c3` | Apache-2.0 | Eligible | Complete; not a demonstrated upgrade |
| [Llama Nemotron Reranker 1B v2](https://huggingface.co/nvidia/llama-nemotron-rerank-1b-v2) | `828765652b05bd439c9789d2a6d093db1caa1443` | [OpenMDW-1.1](https://openmdw.ai/license/1-1/) and Llama 3.2 terms | Eligible under stated terms | Complete; held for comparison |
| [Qwen3 Reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | Apache-2.0 | Eligible | Complete; footprint-first fallback with a reviewed mechanism loss |
| [R3 Reranker 0.6B](https://huggingface.co/tencent/R3-rerank-0.6b) | `6d0c3e486b8653578fd3f0e3c8e7871a0ce01953` | Apache-2.0 | Eligible | Complete; not a demonstrated upgrade |
| Qwen3 Reranker 4B NF4 | `22e683669bc0f0bd69640a1354a6d0aebcfeede5` | Apache-2.0 upstream weights | Eligible | Selected; production and installed synthetic validation passed |
| [Jina Reranker v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) | `e8a93f33f0b22108f8c2364f8484ce3422552fbc` | CC BY-NC 4.0 | Ineligible | Research only; no GPU trial |

The CPU token audit covered all 3,740 pairs and confirmed suffix preservation.
Three Qwen pairs, nine Ettin pairs, and three Nemotron pairs reached the limit.
R3 remains a transfer test because its model card describes agent-skill routing
with its own embedder.

A replay imported `mainframe.core.reranker.Reranker` from the audited worktree.
All 3,740 scores and 187 full rankings matched the NF4 result exactly; maximum
absolute score difference was zero, and ownership, nonce, cleanup, and source
integrity checks passed. It measured 2.493/2.531 GiB allocated/reserved after
initialization, 3.160/3.928 GiB at the five-query warmup peak, then
3.736/6.154 GiB during the 187 queries after resetting peak counters. These
scoped values add no speed claim.

An installed-wheel smoke ran real Harrier and Qwen models, normal unpaced daemon
inference, and a fresh two-document synthetic corpus. Both phases reported the
expected pins, `nf4-bf16`, `qwen3-logit`, and batch eight. Authentication, 76
searches, mutation/capture, zero-upsert restart, 44 failure-free health checks,
exit, and model release passed. Startup peaked at 3.756/3.848 GiB
allocated/reserved; restart peaked at 3.682/3.727 GiB, and both released to
0.016/0.041 GiB. This verifies integration and cleanup, not quality, latency, or
larger-corpus memory.

## Limits and current decision

SciFact and the public Mainframe questions were reused; the working and challenge
sets are known development evidence, not independent holdouts. Results apply to
Harrier's frozen top-20 pools, not other embedders, an entire corpus, or unseen
normal use. Raw scores are not calibrated confidence.

NF4 is the preferred shared-GPU configuration because it preserves the reviewed
local answers while materially reducing loaded allocation. The strongest
attributable saving is the loaded reranker allocation, 4.113 to 2.493 GiB,
about 39%. Its full-run peak reduction also includes the cache and BF16 compute
changes; the cache-off INT8 evidence is only a targeted 23-query subset and is
not a matched full-run control.

Production and installed synthetic validation passed. The genuine SciFact loss
still rules out a quality-parity claim; broader claims need independent normal
use evidence.
