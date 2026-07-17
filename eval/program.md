# Autoresearch: Mainframe RAG Parameter Optimization

_The maintainer's agent-driven tuning runbook — usable as a template: point an agent at this file and eval/evaluate.py against your own corpus._

## Goal
Maximize `val_score` in `eval/evaluate.py` by tuning search and tier boost parameters.

## How to run
```bash
cd /path/to/mainframe-mcp
python eval/evaluate.py --live
```

Uses the existing Mainframe index (218 files, ~15 min to rebuild — do NOT rebuild each run).
Each eval run takes ~30-40s (model load + 30 search queries).
Outputs JSON report to stdout and `val_score: X.XXXX` to stderr. **Higher is better.**

## What you can modify
ONLY the search and tier boost parameters in `eval/evaluate.py` (lines 31-36):

```python
FETCH_MULTIPLIER = 2        # overfetch ratio for reranking (try: 2-8)
RERANK_TOP_K = 3            # results after reranking (try: 3-15)
LIBRARY_BOOST = 0.9         # tier boost for library (try: 0.7-1.0)
PROJECT_BOOST = 0.93        # tier boost for project CLAUDE.md (try: 0.8-1.0)
DOCS_BOOST = 0.95           # tier boost for docs (try: 0.85-1.0)
RESEARCH_BOOST = 0.97       # tier boost for research (try: 0.9-1.05)
ARCHIVE_BOOST = 1.1         # tier boost for archive (try: 1.0-1.5)
```

Do NOT modify CHUNK_SIZE or OVERLAP_RATIO — those require re-indexing (15 min per run).

## Rules
- Only modify the parameter values, not the evaluation logic
- Record every experiment: parameters and resulting val_score
- Try one parameter at a time first, then combinations
- The score is a composite: 40% MRR + 30% hit@1 + 30% text_match@1
- The test corpus is 218 files from 13 real projects (research, docs, CLAUDE.md)
- Test queries covering a spread of project types across the indexed corpus

## What NOT to modify
- The test queries (`test_queries.json`)
- The scoring formula in `evaluate()`
- CHUNK_SIZE or OVERLAP_RATIO (require re-indexing)
- Any file outside `eval/evaluate.py`

## Baseline
Run with default parameters first to establish the baseline score on the full 218-file corpus.
