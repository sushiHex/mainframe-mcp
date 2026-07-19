# Mainframe MCP Server

GPU-accelerated semantic knowledge base MCP server for Claude Code. Replaces mcp-local-rag with Qwen3-8B embeddings, native Qwen3-Reranker-4B reranking, and markdown-aware chunking on an RTX 3090.

## Stack
- Python 3.10+, PyTorch CUDA 12.8, sentence-transformers, LanceDB
- RTX 3090 (24GB VRAM): Qwen3-8B Q8 (9GB) + Qwen3-Reranker-4B INT8 (4.5GB; bge-v2-m3 1.2GB fallback) + DeBERTa NLI (0.8GB) + Qwen2.5-3B Q4 (2.1GB) ≈ 16.4GB peak
- NOTE for public-repo readers: that sizing is the maintainer's deployment, not a requirement — `configs/` ships cpu-only/gpu-light/gpu-minimal presets, and the test suite needs no GPU at all. References to `research/…` and `eval/results.md` point at the maintainer's private research lane (excluded from the public mirror; see CONTRIBUTING.md).

## Structure
```
src/mainframe_mcp/
  server.py       — MCP stdio server (search, ingest, list, delete, status)
  chunker.py      — Markdown header-aware recursive 256-tok chunking
  embedder.py     — Qwen3-Embedding-8B on GPU (Instruct query prefix)
  reranker.py     — bge-reranker-v2-m3 cross-encoder (heading injection)
  store.py        — LanceDB with RAPTOR-ready schema, tier boosting
  dedup.py        — SHA-256 exact + cosine >0.90 near-dupe
  manifest.py     — Provenance tracking (hashes, retrieval counts)
  nli.py          — DeBERTa contradiction detection
  consolidator.py — Qwen2.5-3B report merging + cluster summarization
  raptor.py      — RAPTOR tier-1 clustering and summary generation
  watcher.py      — File change detection
  secrets.py      — Regex secret-scrub denylist (shared by capture + hook)
  contextualizer.py — API contextual retrieval at ingest (opt-in, Haiku; the ONE local-rule exception)
  session_capture.py — Zero-VRAM extractive transcript parser (SessionEnd backstop)
docs/
  MAINFRAME_QUERY.md — Query guide template for projects using the Mainframe
eval/
  evaluate.py     — RAG eval harness (autoresearch target)
  test_queries.json — 50 query-answer pairs (gitignored/private; test_queries.example.json ships)
  program.md      — Autoresearch instructions
  results.md      — Experiment log (private; results.template.md ships)
```

## Commands
- Run server: `python -u -m mainframe_mcp.server`
- Run eval: `python -u eval/evaluate.py`
- Install deps: `pip install lancedb sentence-transformers mcp tantivy`
- CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu128`

## Conventions
- All models load once and stay resident in VRAM (embedder + reranker at startup; the consolidator and NLI lazy-load on first `consolidate_sessions`/`build_raptor`, then stay resident)
- LanceDB schema is RAPTOR-ready (tier, parent_id, cluster_id fields — all tier=0 for now)
- Three tiers: library (boosted), oracle (raw reports), archive (consolidated)
- Dedup at ingest: SHA-256 hash + cosine >0.90 check
- Query embedding uses adaptive Instruct prefix — classifier routes code/research/architecture/legal/config queries to specialized prefixes, general queries use config default
- Reranker prepends chunk headings for section context (config: `reranker.heading_inject`)
- Search is hybrid: vector nearest-neighbor + tantivy FTS keyword matching, merged by distance, then weighted by tier boost (and recency if `recency_weight>0` — see Temporal section)
- Do NOT add doc prefix (path/heading) at ingest time — tested in Session 3d, hurts score
- 256-tok chunks are optimal — 384/512 tested, reduce precision without recall gains
- RAPTOR tier-1 summaries: implemented but NOT in live index — improves h@3 recall but hurts composite score due to text-match penalty on generated summaries. Use `build_raptor` MCP tool to enable.

## Ambient capture (Phase 1)
- `capture_memory(content, title, project=None, kind="session", session_id="")` MCP tool: the live agent authors a dense markdown note, the server writes ONE immutable file under `<repo>/research/sessions/` (fallback `~/.claude/mainframe/library/sessions/` when cwd is outside `~/repos`), then reuses `ingest_file` verbatim. 100% local, no extra model load.
- `research/sessions/` is the **raw, gitignored lane** — only curated, human-reviewed `research/memory/` notes are git-tracked (Phase 3). Secrets are scrubbed (`secrets.py`) before write; the gitignore is the primary leak defense.
- **One immutable file per capture — never overwrite a session file.** Overwriting re-triggers the append/dedup data-loss bug (`existing_hashes` is collected before `delete_by_doc`). Filenames carry `<HHMMSS>` + content hash + a collision counter.
- `source_type="session"`, derived by the anchored test `"/research/sessions/" in path_str` placed BEFORE the `/research/` branch (order is load-bearing). Tier boost `session: 1.05` (marginal FTS-tail aid only).
- **`search` excludes `source_type="session"` by default** — pass `include_sessions=true` to include captures. This filter (not the near-inert tier boost) is the real precision defense: it keeps accreted session chunks out of the fixed vector candidate pool.
- Exact-dedup donor set excludes session chunks (`get_existing_hashes(exclude_source_type="session")`) so a session chunk can't evict an identical curated chunk by ingest order.
- `build_raptor` skips `source_type="session"` docs — no tier-1 summaries from low-signal session logs.
- `delete_file` now calls `Manifest.remove()` so a corrected/re-created file re-indexes (no ghost manifest entry).
- Triggers: SKILL `/capture` (primary, in-context) + `SessionEnd` hook `~/.claude/hooks/capture_session.py` (zero-VRAM backstop, writes file; dedups on `*-<session8>*.md` AND skips if the transcript shows capture_memory was already called) + `SessionStart` hook `mainframe_sessionstart.py` (zero-VRAM; nudges the live server to `sync_index` — does NOT load a second model set). All additive in `~/.claude/settings.json`. NOTE for readers of the public repo: these hooks/skills live in the maintainer's personal `~/.claude/`, not in this repo — adapt the pattern for your own setup or call `capture_memory` manually.
- `sync_index` (Phase 2) is the incremental sync tool: ingests new/modified knowledge files AND prunes files deleted from disk (deletion is decided by file non-existence via `watcher.diff_files`, so manually-ingested files aren't wrongly pruned). Superset of `ingest_projects`. Run from the live server only — never concurrently with another LanceDB writer.
- Source type is derived in ONE place: `store.classify_source_type(path_str)`, shared by `ingest_file` and the scanner (research/sessions/ and the library/sessions fallback both → `session`).
- Tests: `tests/` (GPU-free — fake CPU embedder + real LanceDB). Run one: `python -m pytest tests/test_capture_memory.py`.

## Temporal + conflict-aware memory (mem0-inspired; see `research/2026-06-30-mem0-architecture-for-mainframe.md`)
- **Recency ranking (A) — OFF by default (`search.recency_weight=0`, opt-in):** when enabled, `store.search` multiplies the boosted candidate score by a vectorized half-life factor (`recency_halflife_days` 90, clamped >0) so a recent chunk is nudged higher in the **candidate pool** — the cross-encoder reranker is recency-unaware and may reorder, so it is NOT a guarantee (like the near-inert tier boost). **Keys off `authored_at` (file mtime, written at ingest; per-row fallback to `created_at` for old rows)** — re-ingesting an old file no longer makes it look freshly written. Pre-migration tables get the column via LanceDB `add_columns` on first open (backfilled from created_at). Eval-neutral when off (verified 0.3827).
- **Conflict-aware consolidation (B):** `consolidate_sessions` resolves the `research/memory/<slug>.md` path FIRST and, if a note exists, feeds ONLY its curated body (`_canonical_body` strips the title/`_Consolidated…_` preamble/⚠ section/`## Sources`) to `Consolidator.consolidate(reports, topic, existing=...)`, which switches to `UPDATE_CONSOLIDATION_PROMPT` — newer facts SUPERSEDE stale ones, and feeding body-only stops scaffolding/provenance from accreting each round (anti-bloat → counters index dilution). The UPDATE prompt puts NEW reports first and budgets them against a measured skeleton so a large note can't truncate them away. Sources are pruned when the note ingest is `ingested`/`unchanged`/`all_chunks_duplicate` (all mean it's in the index), not on a hard error.
- **Contradiction surfacing (C):** on *re*-consolidation only, `_detect_contradictions` delegates to `nli.find_contradictions_nn`, which NLI-checks each new-report fact against its NEAREST existing-body fact (resident embedder cosine ≥ 0.4 gate → one NLI call per new fact, not O(n·m)); flagged pairs are appended under `CONTRADICTION_HEADING` + counted in the return dict (`nli` status: `ran`/`no_prior_note`/`disabled`/`error`). Annotate, never mutate (files-as-truth). Lazy-loads `nli.py` via `_ensure_nli`; fully optional (any NLI failure degrades, never aborts). `nli.fact_units` splits text into atomic fact lines (strips list markers, drops headings/short lines, evenly samples to a cap of 120).

## Ops hardening (Tier 0 of the 10-POV review — see `research/2026-07-01-mainframe-10-pov-review.md`)
- **Search is read-only.** The old per-query `retrieval_count` `table.update` was the primary LanceDB fragmentation driver (666 fragments / 1349 versions / 737 MB at 36K chunks) and the counter is never used for ranking. Manifest retrieval analytics accumulate in memory, flushed every 25th search.
- **`Store.optimize()`** (compaction + version cleanup + FTS refresh) runs after `sync_index`/`ingest_projects` batches that changed anything, and now also **creates the FTS index if missing** (a brand-new mainframe_dir otherwise ran its whole first session vector-only). Without it nothing ever compacts, and **rows inserted after the FTS index was created are invisible to keyword search** until an optimize pass. Never run concurrently with another writer. **FTS detection matches both 'INVERTED' and 'FTS' index_type spellings** — the INVERTED-only check re-created the index on every process start (26 unpruned generations, +193MB, from eval runs alone; fixed + live DB compacted 881->645MB on 2026-07-02).
- **Manifest writes are atomic** (tmp + `os.replace`, previous version rotated to `.manifest.json.bak`); a corrupt manifest falls back to `.bak` → empty instead of bricking startup.
- **`_ingest_paths` survives poison files** — a raising `ingest_file` (CUDA fault etc.) is logged into `errors` and the batch continues. **Batch ingests build ONE `dedup.DonorCache`** (hash->doc-paths, maintained incrementally: remove-before-delete, add-after-insert) instead of a full projected table scan PER FILE (an 88-file sync scanned 37K rows 88 times before this). Empty/whitespace files return `status="empty"` (a skip, not an error). **`hf_offline_if_cached(config)`** sets `HF_HUB_OFFLINE=1` at server/eval startup when every enabled model is already cached — a hub outage can no longer hang the pre-warm (stays online on first run/model swap; respects an explicit user env).
- **Consolidation output is re-scrubbed** (`scrub_secrets`) before the git-tracked `research/memory/` write — the note is `research`-typed (searchable, committable), so a secret missed at capture time must not be promoted out of the gitignored session lane.
- **`secrets.py`**: the KV pattern matches keyword-SUFFIX identifiers (`AWS_SECRET_ACCESS_KEY=`, `DB_PASSWORD=` — a bare `\b` never fires inside SCREAMING_SNAKE); also Stripe `sk_live_`/`whsec_`, GCP `GOCSPX-`, and `scheme://user:pass@host` URL creds (password-only redaction).
- **`project` args are validated** (`_SAFE_NAME_RE`, no separators/leading dot) and `session_id` is sanitized before becoming a filename component — blocks `../` traversal out of `repos_dir`.

## Agent-UX + eval infra (Tiers 1-3 of the 10-POV review)
- **`search.limit` is an honest result cap** (default 3, ceiling 10, clamped in code — the schema max is advisory) and the reranker's candidate pool is independently floored by `search.candidate_pool` (20). **Pool-sweep verdict (2026-07-01, eval/results.md): do NOT tune the pool — recall@pool hits 0.90 at 100 candidates but bge-reranker can't convert it (score flat, 117× latency). The reranker is the bottleneck; the next lever is the matched-family Qwen3-Reranker swap.**
- **Search results carry `truncated`** (500-char preview clipped) **+ `char_start`/`char_end`** (exact source span for citation). The MCP layer appends a rephrase **hint on empty results** and a **low-confidence warning** when best `rerank_score` < 0.3 — the "specific technical terms / trust rerank_score" wisdom now lives in the tool surface, not just this file.
- **`file_path`** is the canonical param for ingest_file/delete_file (`filePath` still accepted as a legacy alias).
- **`consolidate_sessions` days default is 0 = ALL pending sessions** (a fixed window stranded older captures forever). Re-consolidation returns `note_bloat_warning: true` when the merge grew >2× the prior body (LLM accreting instead of superseding).
- **`eval/evaluate.py` mirrors production**: candidate pool floor (`CANDIDATE_POOL=20`, was 6 — old scores are NOT comparable), recency wiring from config, and a `--min-score` regression gate (exit 1). GPU-free suite runs in CI (`.github/workflows/tests.yml`); test deps via `pip install -e .[test]`.

## Public mirror (github.com/sushiHex/mainframe-mcp)
- MAINTAINER-ONLY (meaningless in a fork): dev happens on `origin` (mainframe-mcp-private, full history). The public repo is a squashed single-commit snapshot: run `bash scripts/publish_mirror.sh` (clean tree required) — it rebuilds the orphan `public` branch, excludes the PRIVATE-BY-POLICY paths, leak-greps, and force-pushes `public:main`.
- **External PRs land on the PUBLIC repo but are never merged there** (its `main` is force-replaced by every mirror refresh): apply the PR as a patch to master here (keep attribution in the commit message), refresh the mirror, close the PR with a landed-in note. CONTRIBUTING.md + the PR template document this for contributors; CI runs a leak-check job on public PRs.
- **Private by policy, tracked in master but never published**: `research/` (ALL research, regardless of audit verdict), `eval/results.md` (real experiment log — `eval/results.template.md` ships instead), `eval/sessions/`, `docs/mainframe-mcp-design.md`. The exclusion list lives in the script — add there, not ad hoc.

## Head-to-toe wave (2026-07-02 — see the 10-POV review doc for provenance)
- **Memory loop V1**: `sync_index` + `status` return `pending_sessions` (per-project un-consolidated capture counts via `session_capture.pending_sessions_by_project`) and a `consolidation_hint` at `memory.consolidate_threshold` (5) — the agent decides, no daemon. `consolidate_sessions` **topic defaults to the project name** (one curated note per project; explicit topic still supported). The capture hooks are VERIFIED FIRING in production (2026-07-16: 130 events in `~/.claude/mainframe/hook.log`, session captures across 12+ repos; the first real capture also exposed two `_FILE_PATH_RE` bugs — URL tails and version-number dirs matched as file paths — both fixed with regression tests).
- **Pre-warm**: `main()` starts model loading in a daemon thread at launch (get_mainframe is lock-guarded), so the ~2min embedder+4B load overlaps the MCP handshake instead of taxing the first search.
- **`status()`** now reports the loaded reranker backend (`Reranker.info`), VRAM, `index_health` (fragments/versions), and pending sessions. **`reload_config`** hot-reloads live-read knobs (chunker/search/tiers); model-section changes come back as `restart_required_for`.
- **Per-category reranker instructions — measured, REJECTED as default (0.3760 vs 0.4187, the 4th model-card-vs-corpus inversion)**: the machinery stays (`reranker.CATEGORY_INSTRUCTIONS`, config `reranker.instructions` overrides, gated by `reranker.category_instructions` = false) for instruction-WORDING experiments; the embedder's query classification is only passed to the reranker when the flag is on. qwen3 scoring also clamps the query (not just docs) and degrades to item-at-a-time on CUDA OOM.
- **`consolidator.build_prompt` is a pure function** (tokenizer injected) — the newer-facts-survive budget math is now unit-tested GPU-free (`tests/test_consolidator_prompt.py`).
- **eval `--rebuild` is manifest-driven** (reads the live manifest's file list, classifies source types) — the old flat `library/*.md` glob could not reproduce the real 664-file corpus, blocking chunking experiments and the contextual-retrieval A/B.
- **Packaging**: `bitsandbytes` is a hard dep (all three default model loads quantize); `anthropic` ships as the `contextual` extra.

## MCP ergonomics for non-Claude hosts (2026-07-17 — Hermes/ChatGPT bake-off feedback)
- **Portable, not Claude-coupled**: `MAINFRAME_CONFIG` env overrides the config path (falls back to `~/.claude/mainframe/config.json`). Server speaks plain MCP/stdio — README documents a generic gateway snippet.
- **Prewarm is OPT-IN** (`MAINFRAME_PREWARM=1`, default LAZY): `main()` only spawns the model-load daemon thread when set, so an always-on gateway sharing the GPU doesn't claim ~13GB on connect. Claude-Code users who want instant first-search set it. `_prewarm_enabled()`/`_env_flag()` are unit-tested.
- **Read-only mode** (`MAINFRAME_READ_ONLY=1`): `list_tools` advertises only `READ_ONLY_TOOLS` (search/list_files/status) and `call_tool` rejects mutations (belt-and-suspenders vs injection). Run maintenance from a separate full-access invocation.
- **Search returns STRUCTURED JSON** `{results, confidence: high|low|none, guidance?}` — no prose appended to the JSON body (was invalid JSON for strict clients). The empty/low-confidence rephrase hints moved into `guidance`.
- **Line-number citations**: `search` results carry 1-based `line_start`/`line_end` (`MainframeMCP._add_line_spans`, each unique file read once, absent on unreadable/moved files) for line-oriented `read_file(offset=…)` tools.
- **Root context files**: `store.ROOT_CONTEXT_FILES` = CLAUDE.md/AGENTS.md/HERMES.md/.hermes.md all classify as `project` tier and are picked up by the scanner; SOUL.md and other identity files are deliberately NOT indexed.
- **Packaging**: `accelerate` (device_map="auto" on all quantized loaders) and `pandas` (store.search runtime import) are now declared runtime deps — a clean install was missing both.
- **Optional `skills/mainframe-retrieval/SKILL.md`** ships the query/capture technique so hosts can keep it out of the permanent tool-schema prompt.

## Retrieval experiments (Tier 1)
- **Reranker backends** (`reranker.py`): `cross-encoder` (default, bge et al.) and `qwen3-logit` — native Qwen3-Reranker scoring (causal LM, P("yes") from last-position logits, model-card prompt scaffold, instruction-aware via `reranker.instruction`, INT8 ~4.5GB for the 4B). Auto-detected from the model name (`detect_backend`; the `*-seq-cls` conversions stay on CrossEncoder), override with `reranker.backend`. A/B via `MAINFRAME_RERANKER_MODEL` env — no re-index needed.
- **Eval baselines (pool-20 harness, 664-file corpus)**: **Qwen3-Reranker-4B native 0.4187 (current default + the `--min-score 0.41` gate)** vs bge 0.3320 (+26%; costs ~8s/query vs ~2-4s, ~4.5GB VRAM vs 1.2GB, ~93s load). Rejected: Qwen3-Reranker-0.6B-seq-cls 0.2327 (the seq-cls conversion was the wrong weapon — the native head + instruction wins); pool widening under bge (recall@pool 0.90 at 100 candidates unconverted — see eval/results.md 2026-07-01). Live deployments opt in by editing `~/.claude/mainframe/config.json`.
- **API contextual retrieval** (`contextualizer.py`, config `contextual.enabled`, OFF by default): at INGEST only, Haiku situates each chunk in its doc (prompt-cached doc prefix) and the 1-2 sentence context is prepended before hash/embed/FTS — the ONE deliberate 100%-local exception (serving stays local). Failures degrade to plain chunks; contexts are scrubbed. Enabling requires API credentials in the server env and a re-ingest to take effect.

## Gotchas
- PyTorch default pip install is CPU-only on Windows — must use `--index-url .../cu128`
- bge-reranker-v2-m3 replaced mxbai-rerank-large-v2 (Session C: +10% score, 60% less VRAM). The `torch.manual_seed(42)` in reranker.py remains as a safety net for any model with missing checkpoint weights.
- LanceDB `table.to_pandas(columns=[...])` RAISES on this version (no `columns` kwarg) — it was silently swallowed by bare `except`, disabling dedup/stats. For projected reads (skip the 4096-dim `vector` column) use `table.to_lance().to_table(columns=[...]).to_pandas()` (see `Store._scan`). A bare `to_pandas()` loads vectors.
- Reranker can disagree with vector search on small corpora — resolves at scale
- **HuggingFace Hub outages hang model loading DESPITE a full local cache** — sentence-transformers HEAD-checks the hub per config file and retries 504s for minutes before falling back. If loads hang or a sync gets killed mid-load, set `HF_HUB_OFFLINE=1` (safe whenever weights are already cached; also faster)

## Running GPU scripts
- Always use `python -u` (unbuffered) for background tasks, otherwise stdout buffers silently and the task appears stuck
- Model loading takes ~21s — this is normal, not a hang
- Never run two GPU-heavy scripts concurrently — all 4 models total ~15GB VRAM on a 24GB card, no room for a second process
- For eval sweeps: load models once, loop over configs in one process (see eval/sweep pattern)
- If a background task shows 0 output for >30s, check with `wc -l` on the output file before killing — it may be buffered
