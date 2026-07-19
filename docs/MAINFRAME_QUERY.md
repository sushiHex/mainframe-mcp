# Mainframe Query Guide

How to search the Mainframe effectively from any Claude Code session.

## CLAUDE.md Template

Include this section (or a subset) in the CLAUDE.md of any project that searches the Mainframe:

```markdown
## Mainframe
- Search before answering from training data: `mcp__mainframe__search("specific technical terms")`
- Use specific nouns and function names, not natural language questions
- One topic per search. Multi-topic queries scatter results.
- Trust `rerank_score` (higher = better). Above 0.85 is strong. Below 0.3 is noise.
- `limit=2-3` for targeted lookups, `limit=5-10` for exploratory coverage
- After adding `research/` or `docs/` files, run `mcp__mainframe__sync_index` to index them
- Cross-project results are expected — knowledge from any repo is available here
```

## Tools

| Tool | Purpose |
|------|---------|
| `mcp__mainframe__search` | Semantic search. Args: `query` (string), `limit` (int, default 3, max 10 — the actual result count; the reranker's candidate pool is floored separately at 20). |
| `mcp__mainframe__sync_index` | Incremental sync: ingest new/modified knowledge files AND prune deleted ones, then compact. Prefer over ingest_projects. |
| `mcp__mainframe__ingest_projects` | DEPRECATED — use sync_index. |
| `mcp__mainframe__ingest_file` | Ingest a single .md file by absolute path (`file_path`). |
| `mcp__mainframe__status` | Index stats, model info, retrieval analytics. |
| `mcp__mainframe__build_raptor` | Build tier-1 summary clusters. Currently hurts precision — skip unless recall matters more. |
| `mcp__mainframe__list_files` | Show all indexed files with retrieval counts. |
| `mcp__mainframe__delete_file` | Remove a file from the index. |

## Reading Results

The `search` tool returns a JSON object: `{"results": [...], "confidence":
"high"|"low"|"none", "guidance": "..."}`. `guidance` is present only on `low`/
`none` and tells you how to rephrase — the whole payload is valid JSON, safe to
parse. Each entry in `results` has:
- `file` — source file path
- `heading` — markdown section heading
- `text` — chunk excerpt (first 500 chars)
- `truncated` — `true` when the chunk was longer than the 500-char preview
- `char_start` / `char_end` — exact character span in the source file (use with `Read` to cite the full section)
- `line_start` / `line_end` — 1-based line span (for line-oriented `read_file(offset=…)` tools); omitted if the source file has moved or changed
- `rerank_score` — 0 to 1, **higher = more relevant** (cross-encoder reranked)
- `score` — raw vector distance (lower = closer, but rerank_score is the one to trust)
- `source_type` — `research`, `docs`, `project`, `library`, `session`, or `archive`

**Use `rerank_score` to judge quality.** Above 0.85 is a strong match. Below 0.3 is noise.

## Writing Effective Queries

### What works best: specific technical terms

```
LanceDB to_pandas columns kwarg projection workaround         → 0.99
bitsandbytes INT8 kernel grid batch seq_length overflow       → 0.99
bge-reranker heading injection cross-encoder config           → 0.99
```

The embedder (Qwen3-8B) excels at matching specific nouns, function names, and technical vocabulary.

### What works well: mixed technical + natural language

```
Zod v4 catchall strict deprecated backward compatible         → 0.82
Docker Dockerfile FastAPI Python production uvicorn           → 0.90
error handling SVG parsing failure invalid path retry          → 0.84
```

### What works but lower precision: natural language questions

```
what happens when the index returns stale results after edits      → 0.10
how do you test that retrieval works across different corpora      → 0.05
```

Natural language finds relevant *documents* but the rerank_score is low because the chunk text doesn't phrase-match the question. The results are still useful — just check more of them.

### What doesn't work

| Pattern | Why |
|---------|-----|
| `chunks NOT overlapping` | Embedder doesn't understand boolean negation |
| `chunking AND reranking AND deployment` | Multi-topic queries scatter attention — one topic per search |
| `"exact phrase match"` | Quotes are ignored — this is semantic search, not FTS |
| Very short queries (`reranker`) | Works but returns broad results — add context words |

## Query Strategies

### Start specific, broaden if needed
```
# Try specific first
mcp__mainframe__search("CrossEncoder predict batch heading injection")

# If too few results, broaden
mcp__mainframe__search("sentence-transformers reranker Python config")

# If still nothing, go natural language
mcp__mainframe__search("how to rerank search results with a cross encoder")
```

### Use limit=2 or limit=3 for targeted lookups
The reranker puts the best result first. For specific facts or code patterns, `limit=2` avoids noise. Use `limit=5-10` for exploratory searches where you want coverage across multiple documents.

### Search per task, not in bulk
Don't try to pre-load everything. Search when you need a specific detail for the section you're writing. The Mainframe is fast — treat it as a lookup table, not a batch export.

### Cross-project results are a feature
The Mainframe indexes all repos under ~/repos/. A search for "Claude Code plan mode" may return results from an `agents/` project. This is useful — it means knowledge from one project is available in another.

## Handling Conflicts

The Mainframe indexes research from multiple rounds and multiple projects. Conflicts happen.

**When two results disagree:**
- Check the `file` path — later rounds generally have corrections
- Look for "CORRECTION" or "DISPUTE" headings in the results
- Round numbers in filenames indicate order: `round15` supersedes `round7`

**When a result says something different from what you believe:**
- Trust the Mainframe result over assumptions — it's grounded in research
- But verify: the result may be from a different project or context
- If the `source_type` is `research`, it's from an Oracle report (high trust)
- If it's `docs`, it's from a project doc (check if it's the right project)

## Common Pitfalls

1. **Reranker disagrees with vector search**: The top result by vector distance may not be top by rerank. Trust `rerank_score`.

2. **Truncated chunks**: `truncated: true` means the 500-char preview was clipped. Use `char_start`/`char_end` with the `Read` tool to pull the exact full section from `file`.

3. **Stale index**: If you just added files to `research/`, run `sync_index` before searching. New files won't appear until ingested.

4. **Abstract queries get low rerank scores**: A search for "best approach" returns relevant documents but with 0.01-0.05 rerank because the chunk text doesn't contain the word "best." The results are still correct — just check them.

5. **Same content, multiple rounds**: A finding researched in round 3 and refined in round 10 appears as two separate chunks. The later round is usually more accurate and more detailed.
