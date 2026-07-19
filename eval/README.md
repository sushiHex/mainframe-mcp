# Mainframe MCP Evaluation Framework

## Overview

Automated parameter optimization for the Mainframe RAG system. Uses autoresearch pattern: hypothesize → modify → test → evaluate → keep or discard → repeat.

## Quick Start

```bash
# Run a single eval against the live index
python eval/evaluate.py --live

# View results history
python eval/history.py --best
```

## Test Corpus

The eval runs against YOUR live index (or a temp rebuild of it) with a
corpus-specific query set: `eval/test_queries.json` (gitignored — it inevitably
describes your private corpus). `eval/test_queries.example.json` documents the
format. Record experiments in `eval/results.md` following
`eval/results.template.md`.

## Eval Modes

| Mode | Command | Speed | What it tests |
|---|---|---|---|
| `--live` | Uses existing index | ~40s | Search params, tier boosts, reranking |
| `--rebuild` | Rebuilds temp index | ~15min | Chunking params + search params |

## Autoresearch Sessions

Each session tests a specific category of parameters. Open a new Claude Code session in this directory, provide the session instructions, let it iterate.

### Session Types

**Session A: Search & Ranking (fast, ~40s/run)**
```
Read eval/sessions/search-ranking.md and optimize.
```
Parameters: FETCH_MULTIPLIER, RERANK_TOP_K

**Session B: Tier Boost Weights (fast, ~40s/run)**
```
Read eval/sessions/tier-boosts.md and optimize.
```
Parameters: LIBRARY_BOOST, PROJECT_BOOST, DOCS_BOOST, RESEARCH_BOOST, ARCHIVE_BOOST

**Session C: Model Swaps (slow, ~15min/run)**
```
Read eval/sessions/model-swap.md and optimize.
```
Swaps embedding model and/or reranker, rebuilds index, evaluates.

**Session D: Chunking (slow, ~15min/run)**
```
Read eval/sessions/chunking.md and optimize.
```
Parameters: CHUNK_SIZE, OVERLAP_RATIO, MIN_SECTION_TOKENS

## Results

Each run auto-saves to `eval/results/YYYYMMDD-HHMMSS.json`.

```bash
python eval/history.py              # all runs
python eval/history.py --best       # top 10 by score
python eval/history.py --last 5     # most recent 5
```

## Scoring

Composite: 40% MRR + 30% hit@1 + 30% text_match@1. Higher is better.

| Metric | What it measures |
|---|---|
| hit@1 | Correct file in the #1 result |
| hit@3 | Correct file in top 3 |
| hit@5 | Correct file in top 5 |
| MRR | Mean reciprocal rank (how high is the correct result) |
| text_match@1 | Expected keyword in the #1 result's text |
