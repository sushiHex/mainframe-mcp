# Experiment Log — template

Append-only log of retrieval experiments. The discipline matters more than the
format: **record negative results** (a rejected candidate saves the next person
a GPU afternoon), never compare scores across harness changes, and gate
regressions in CI.

Copy this file to `eval/results.md` and log every run. (The maintainer's own
log is kept private — it describes a private corpus.)

## Ground rules

1. **One baseline per harness configuration.** Any change to the candidate
   pool, query set, corpus size, or scoring weights starts a new baseline —
   note it loudly and never compare across the break.
2. **Negative results are results.** Log rejected models/params with the
   numbers and WHY (wrong architecture? corpus mismatch? latency?).
3. **Re-run the baseline after reverting an experiment** to prove the revert
   is bit-faithful before trusting subsequent numbers.
4. **Wire the gate:** `python eval/evaluate.py --live --min-score <baseline>`
   exits 1 on regression — run it in CI or before promoting any change.

## Entry format

```markdown
## Session <YYYY-MM-DD><letter>: <one-line what-and-verdict>

<1-3 sentences: hypothesis, what changed, verdict.>

| config | hit@1 | hit@3 | MRR | txt@1 | score | s/query |
|--------|-------|-------|-----|-------|-------|---------|
| baseline (<name>) | | | | | | |
| candidate         | | | | | | |

Decision: <promoted / rejected / config-gated>. <Costs: VRAM, latency, load
time.> <New gate number if the baseline moved.>
```

## Example entry (fictional numbers)

## Session 2026-01-15a: swapped reranker X for Y — REJECTED

Hypothesis: Y's larger head converts the candidate pool better. It didn't —
leaderboard rank inverted on this corpus (third such inversion; always eval
locally, never trust the leaderboard).

| config | hit@1 | hit@3 | MRR | txt@1 | score | s/query |
|--------|-------|-------|-----|-------|-------|---------|
| baseline (X) | 0.40 | 0.64 | 0.51 | 0.32 | 0.4187 | 8.0 |
| candidate (Y) | 0.30 | 0.55 | 0.41 | 0.25 | 0.3320 | 3.9 |

Decision: rejected. Baseline reproduction after revert: 0.4187 exact. Gate
stays `--min-score 0.41`.
