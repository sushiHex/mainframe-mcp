"""LanceDB vector store with RAPTOR-ready schema and hybrid search.

Supports: tier-boosted search, retrieval counting, dedup checking, FTS hybrid.
"""

import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

logger = logging.getLogger(__name__)

# Source type for ambient session captures — centralized so the ingest
# classifier, the default search exclusion, the dedup donor scope, and the
# RAPTOR skip can't drift apart on a rename.
SESSION_SOURCE_TYPE = "session"

# Root-level agent-context files that carry project conventions/evidence, tiered
# like CLAUDE.md ("project"). SOUL.md and other identity files are deliberately
# NOT here — identity instructions are not ordinary project evidence.
ROOT_CONTEXT_FILES = frozenset({"CLAUDE.md", "AGENTS.md", "HERMES.md", ".hermes.md"})


def classify_source_type(path_str: str, mainframe_dir: str | None = None) -> str:
    """Derive a chunk's source_type from its path. Single source of truth for
    both ingest_file and the bulk scanner so they can't disagree.

    Order is load-bearing: the session lanes (research/sessions/ and the
    library/sessions/ fallback) are checked BEFORE the broader library/research
    branches that their paths also match, so session captures are correctly
    typed "session" and stay subject to the default search exclusion + tier.

    `mainframe_dir` makes the Mainframe's own library/archive lanes robust to a
    relocated MAINFRAME_DIR — without it, the '/mainframe/...' substrings assume
    the dir is literally named "mainframe".
    """
    p = path_str.replace("\\", "/")
    md = (mainframe_dir or "").replace("\\", "/").rstrip("/")

    if "/research/sessions/" in p or "/mainframe/library/sessions/" in p:
        return SESSION_SOURCE_TYPE
    if md and p.startswith(md + "/library/sessions/"):
        return SESSION_SOURCE_TYPE
    if "/mainframe/archive/" in p or (md and p.startswith(md + "/archive/")):
        return "archive"
    if "/mainframe/library/" in p or (md and p.startswith(md + "/library/")):
        return "library"
    if p.rsplit("/", 1)[-1] in ROOT_CONTEXT_FILES:
        return "project"
    if "/research/" in p:
        return "research"
    if "/docs/" in p or "/machines/" in p or "/playbooks/" in p:
        return "docs"
    return "library"


# Tier boost: library results get a relevance boost over oracle/archive
class Store:
    def __init__(self, db_path: Path, embedding_dim: int = 4096, tier_boost: dict | None = None,
                 recency_weight: float = 0.0, recency_halflife_days: float = 90.0):
        # Temporal ranking (mem0-style): nudge more recent chunks up the candidate
        # pool (OFF by default — opt in via search.recency_weight). Keys off
        # authored_at (file mtime) with per-row created_at fallback; the reranker
        # is recency-unaware (candidate-pool nudge only). See
        # research/2026-06-30-mem0-architecture-for-mainframe.md.
        self.reconfigure(tier_boost or {}, recency_weight, recency_halflife_days)
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.db = lancedb.connect(str(db_path))
        self._ensure_table()

    @staticmethod
    def search_kwargs_from_config(config: dict) -> dict:
        """The config -> search-knob extraction (tiers + recency defaults) in
        ONE place — server startup, reload_config, and the test harness all
        feed __init__/reconfigure through this, so the default literals can't
        drift between call sites."""
        return {
            "tier_boost": config.get("tiers", {}),
            "recency_weight": config["search"].get("recency_weight", 0.0),
            "recency_halflife_days": config["search"].get("recency_halflife_days", 90),
        }

    def reconfigure(self, tier_boost: dict, recency_weight: float,
                    recency_halflife_days: float):
        """Apply the live-reloadable search knobs. The single home of the
        halflife clamp (guard 0 AND negative — negative would invert:
        0.5**(age/-hl) = 2**age), shared by __init__ and reload_config so the
        two can't drift."""
        self.tier_boost = tier_boost
        self._recency_weight = recency_weight
        self._recency_halflife = recency_halflife_days if recency_halflife_days and recency_halflife_days > 0 else 90.0

    def _ensure_fts_index(self):
        """Create full-text search index on text column if not present.

        Detection must match BOTH spellings LanceDB has used ('INVERTED' and
        'FTS' — 0.30 reports 'FTS'). Matching only 'INVERTED' made every
        process start re-create the index with replace=True, minting a new
        index generation each launch (observed live: 26 generations, +193MB,
        entirely from eval runs against the production DB)."""
        try:
            indices = self.table.list_indices()
            has_fts = any(
                getattr(idx, "columns", []) == ["text"]
                and str(getattr(idx, "index_type", "")).upper() in ("FTS", "INVERTED")
                for idx in indices
            )
            if not has_fts:
                self.table.create_fts_index("text", replace=True)
                logger.info("Created FTS index on text column")
        except Exception as e:
            logger.debug(f"FTS index check/create failed: {e}")

    def _ensure_table(self):
        """Create the chunks table if it doesn't exist."""
        if "chunks" in self.db.table_names():
            self.table = self.db.open_table("chunks")
            logger.info(f"Opened existing table: {len(self.table)} rows")
            self._ensure_fts_index()
            # Schema migration: pre-authored_at tables get the column added,
            # backfilled from created_at (ingest time — the best available
            # proxy for already-indexed rows).
            if "authored_at" not in [f.name for f in self.table.schema]:
                try:
                    self.table.add_columns({"authored_at": "created_at"})
                    logger.info("Migrated schema: added authored_at (backfilled from created_at)")
                except Exception as e:
                    logger.warning(f"authored_at migration failed (non-fatal): {e}")
        else:
            schema = pa.schema([
                pa.field("chunk_id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("doc_path", pa.string()),
                pa.field("tier", pa.int32()),
                pa.field("parent_id", pa.string()),
                pa.field("cluster_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
                pa.field("heading", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("char_start", pa.int32()),
                pa.field("char_end", pa.int32()),
                pa.field("token_count", pa.int32()),
                pa.field("content_hash", pa.string()),
                pa.field("source_type", pa.string()),
                pa.field("retrieval_count", pa.int32()),
                pa.field("created_at", pa.string()),
                pa.field("authored_at", pa.string()),
                pa.field("pipeline_version", pa.string()),
            ])
            self.table = self.db.create_table("chunks", schema=schema)
            logger.info("Created new chunks table")

    def insert_chunks(
        self,
        chunks: list[dict],
        doc_id: str,
        doc_path: str,
        source_type: str,
        embeddings: list[list[float]],
        content_hashes: list[str],
        authored_at: str = "",
    ):
        """Insert chunked and embedded document into the store.

        `authored_at` is the source file's mtime (ISO) — the authored-time
        signal recency ranking prefers over created_at (ingest time), so
        re-ingesting an old file doesn't make it look freshly written."""
        now = datetime.now().isoformat()
        rows = []
        for i, (chunk, embedding, content_hash) in enumerate(
            zip(chunks, embeddings, content_hashes)
        ):
            rows.append({
                "chunk_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "doc_path": doc_path,
                "tier": 0,  # leaf node
                "parent_id": "",
                "cluster_id": "",
                "text": chunk["text"],
                "vector": embedding,
                "heading": chunk.get("heading", ""),
                "chunk_index": chunk.get("chunk_index", i),
                "char_start": chunk.get("char_start", 0),
                "char_end": chunk.get("char_end", 0),
                "token_count": chunk.get("token_count", 0),
                "content_hash": content_hash,
                "source_type": source_type,
                "retrieval_count": 0,
                "created_at": now,
                "authored_at": authored_at or now,
                "pipeline_version": "1.0.0",
            })

        self.table.add(rows)
        # Refresh table handle to see new data
        self.table = self.db.open_table("chunks")
        logger.info(f"Inserted {len(rows)} chunks for {doc_path}")
        return len(rows)

    def delete_by_doc(self, doc_path: str):
        """Delete all chunks for a document."""
        self.table.delete(f'doc_path = "{doc_path}"')
        logger.info(f"Deleted chunks for {doc_path}")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        tier_filter: Optional[int] = None,
        exclude_source_type: Optional[str] = None,
        query_text: str = "",
    ) -> list[dict]:
        """Hybrid vector + FTS search with tier boosting.

        Merges vector nearest-neighbor results with full-text keyword matches,
        deduplicates by chunk_id, and returns top_k by boosted score.

        `exclude_source_type` drops rows of that source_type from the candidate
        set — applied to the vector WHERE (so curated docs fill the candidate
        pool) and again to the merged frame (so FTS-injected rows are excluded
        too). This is the real precision defense for session captures (§4.2).
        """
        import pandas as pd

        # Vector search. Build one combined WHERE — chained .where() calls would
        # overwrite, not AND.
        conds = []
        if tier_filter is not None:
            conds.append(f"tier = {tier_filter}")
        if exclude_source_type:
            conds.append(f'source_type != "{exclude_source_type}"')

        def _vec_query():
            q = self.table.search(query_embedding).limit(top_k)
            if conds:
                q = q.where(" AND ".join(conds))
            return q.to_pandas()

        try:
            vec_results = _vec_query()
        except Exception as e:
            # A long-lived handle can be pinned to a dataset version whose files
            # another process's optimize() has since cleaned up ("Object ... not
            # found"). Reopen at the latest version and retry once — search must
            # self-heal, not stay broken until restart.
            logger.warning(f"Vector search failed ({e}); reopening table and retrying")
            self.table = self.db.open_table("chunks")
            vec_results = _vec_query()

        # FTS search (keyword boost)
        fts_results = pd.DataFrame()
        if query_text:
            try:
                fts_results = self.table.search(query_text, query_type="fts").limit(top_k).to_pandas()
            except Exception:
                pass  # FTS may not be indexed yet

        # Merge: vector results + any FTS results not already in vector set
        if not fts_results.empty and not vec_results.empty:
            vec_ids = set(vec_results["chunk_id"].tolist())
            new_fts = fts_results[~fts_results["chunk_id"].isin(vec_ids)]
            if not new_fts.empty:
                # FTS results don't have _distance — rank them after all vector results
                # so they only help when vector search completely misses the right doc
                max_dist = vec_results["_distance"].max()
                new_fts = new_fts.copy()
                new_fts["_distance"] = max_dist * 1.01  # just after worst vector match
                vec_results = pd.concat([vec_results, new_fts], ignore_index=True)
        elif vec_results.empty and not fts_results.empty:
            fts_results = fts_results.copy()
            fts_results["_distance"] = 1.0
            vec_results = fts_results

        if vec_results.empty:
            return []

        # Re-apply the exclusion after the merge: FTS results bypass the vector
        # WHERE, so an excluded source_type could otherwise re-enter via keywords.
        if exclude_source_type and "source_type" in vec_results.columns:
            vec_results = vec_results[vec_results["source_type"] != exclude_source_type]
            if vec_results.empty:
                return []

        # Apply tier boost (lower boosted_score = better), then — only when
        # enabled — a half-life recency factor in [1-weight, 1], so a recent chunk
        # edges ahead of an equally-relevant stale one in the candidate pool.
        # Vectorized (one datetime parse pass, not per-row); the reranker is
        # recency-unaware, so this is a candidate-pool nudge, not a guarantee.
        # fillna(1.0) is the PRIMARY path, not a safety net: most source_types
        # aren't in tier_boost (only 'library' by default) -> map -> NaN -> 1.0.
        boost = vec_results["source_type"].map(self.tier_boost).fillna(1.0)
        vec_results["boosted_score"] = vec_results["_distance"] * boost
        if self._recency_weight > 0 and "created_at" in vec_results.columns:
            # Prefer authored_at (file mtime — survives re-ingest) with a
            # per-row NULL fallback to created_at (rows written outside
            # ingest_file, e.g. RAPTOR summaries, omit authored_at).
            src = vec_results["created_at"]
            if "authored_at" in vec_results.columns:
                src = vec_results["authored_at"].fillna(src)
            # Timestamps are written as NAIVE LOCAL time (insert_chunks uses
            # datetime.now().isoformat()), so parse them naive and subtract a naive
            # local now — labeling naive values as UTC (utc=True) would inflate
            # every age by the local UTC offset (~8h on this Pacific machine).
            # Only a mixed naive/tz-aware column (object dtype) falls back to
            # UTC normalization, which is what prevents the naive-minus-aware
            # subtraction crash.
            ts = pd.to_datetime(src, errors="coerce")
            if getattr(ts.dtype, "tz", None) is not None:
                now = pd.Timestamp.now(tz=ts.dtype.tz)
            elif ts.dtype == object:  # mixed naive+aware strings
                ts = pd.to_datetime(src, errors="coerce", utc=True)
                now = pd.Timestamp.now(tz="UTC")
            else:
                now = pd.Timestamp.now()  # naive local vs naive local (Pacific)
            ages = (now - ts).dt.total_seconds() / 86400.0
            factors = 1.0 - self._recency_weight * (0.5 ** (ages.clip(lower=0) / self._recency_halflife))
            # NaN factor can only come from NaT (unparseable created_at) -> neutral.
            vec_results["boosted_score"] *= factors.fillna(1.0)
        vec_results = vec_results.sort_values("boosted_score").head(top_k)

        return vec_results.to_dict("records")

    def _scan(self, columns: list[str]):
        """Read only `columns` for the whole table as a DataFrame.

        Projects via the Lance dataset scanner. NOTE: `table.to_pandas(columns=)`
        RAISES on this LanceDB version, so the metadata reads (dedup hashes,
        stats) must NOT request the 4096-dim `vector` column — projecting here
        avoids deserializing ~16 KB/row of float32 just to read a few strings.
        """
        return self.table.to_lance().to_table(columns=columns).to_pandas()

    def get_existing_hashes(
        self,
        exclude_source_type: Optional[str] = None,
        exclude_doc_path: Optional[str] = None,
    ) -> set[str]:
        """Content hashes usable as exact-dedup donors — a view over
        get_hash_doc_map (ONE scan implementation, not two to keep in sync).

        `exclude_source_type` drops chunks of that source_type from the donor
        set so they cannot cause a later, identical curated chunk to be skipped
        (ingest-order-dependent precision inversion — see plan §6 / risk M1).

        `exclude_doc_path` drops the file currently being re-ingested from its
        own donor set — but a hash shared with ANOTHER doc still counts (the
        other copy survives). Without it, an edited file's UNCHANGED sections
        match their own prior chunks, get skipped as dupes, and are then wiped
        by the subsequent delete_by_doc — silent data loss (review finding #1).
        """
        hash_docs = self.get_hash_doc_map(exclude_source_type=exclude_source_type)
        if exclude_doc_path:
            return {h for h, docs in hash_docs.items() if docs - {exclude_doc_path}}
        return set(hash_docs)

    def get_hash_doc_map(self, exclude_source_type: Optional[str] = None) -> dict[str, set[str]]:
        """content_hash -> {doc_paths} for the whole table, ONE projected scan.
        Feeds dedup.DonorCache so a batch ingest doesn't re-scan per file."""
        try:
            df = self._scan(["content_hash", "source_type", "doc_path"])
            if df.empty:
                return {}
            if exclude_source_type:
                df = df[df["source_type"] != exclude_source_type]
            out: dict[str, set[str]] = {}
            for h, d in zip(df["content_hash"], df["doc_path"]):
                out.setdefault(h, set()).add(d)
            return out
        except Exception as e:
            logger.warning(f"get_hash_doc_map scan failed: {e}")
            return {}

    def get_embeddings_for_dedup(
        self, limit: int = 1000
    ) -> tuple[list[list[float]], list[str]]:
        """Get the most recent `limit` embeddings for near-dupe checking.

        Reads only the last `limit` rows (Lance offset) so we don't deserialize
        the full 4096-dim vector column just to slice the tail.
        """
        try:
            ds = self.table.to_lance()
            n = ds.count_rows()
            if n == 0:
                return [], []
            df = ds.to_table(
                columns=["vector", "chunk_id"], offset=max(0, n - limit)
            ).to_pandas()
            return df["vector"].tolist(), df["chunk_id"].tolist()
        except Exception as e:
            logger.warning(f"get_embeddings_for_dedup scan failed: {e}")
            return [], []

    def optimize(self):
        """Compact fragments, clean up old dataset versions, and fold newly
        inserted rows into the FTS index. Run after ingest batches — never
        concurrently with another writer.

        Without this nothing ever compacts: every add/delete accretes fragments
        and versions unboundedly (observed live: 666 fragments / 1349 versions /
        737 MB for ~36K chunks), and rows added AFTER the FTS index was created
        stay invisible to keyword search until an optimize pass."""
        # A brand-new table has NO FTS index (table.optimize refreshes indexes,
        # it never creates one) — without this, a fresh mainframe_dir runs its
        # entire first server session vector-only until a restart.
        self._ensure_fts_index()
        try:
            try:
                # 1 day (not minutes): cleanup DELETES files of older versions on
                # disk, which crashes any OTHER process whose table handle is
                # still pinned to one (a search-only session never refreshes its
                # handle). A day-wide window keeps long-lived readers safe;
                # store.search also self-heals by reopening on a failed read.
                self.table.optimize(cleanup_older_than=timedelta(days=1))
            except TypeError:  # future LanceDB signature drift
                self.table.optimize()
            logger.info("Index optimized (compaction + version cleanup + FTS refresh)")
        except Exception as e:
            logger.warning(f"Index optimize failed (non-fatal): {e}")

    def index_health(self) -> dict:
        """Fragment/version counts — the compaction signals that were previously
        only observable by inspecting .lancedb on disk (666 fragments went
        unnoticed for months)."""
        try:
            ds = self.table.to_lance()
            return {
                "fragments": len(ds.get_fragments()),
                "versions": len(self.table.list_versions()),
                "rows": ds.count_rows(),
            }
        except Exception as e:
            logger.warning(f"index_health failed: {e}")
            return {"fragments": -1, "versions": -1, "rows": -1}

    def get_stats(self) -> dict:
        """Return index statistics (projected columns only — no vectors)."""
        empty = {"total_chunks": 0, "by_tier": {}, "by_source_type": {}, "unique_docs": 0}
        try:
            df = self._scan(["tier", "source_type", "doc_path"])
            if df.empty:
                return empty
            return {
                "total_chunks": len(df),
                "by_tier": {str(k): int(v) for k, v in df["tier"].value_counts().items()},
                "by_source_type": {str(k): int(v) for k, v in df["source_type"].value_counts().items()},
                "unique_docs": int(df["doc_path"].nunique()),
            }
        except Exception as e:
            logger.warning(f"get_stats scan failed: {e}")
            return empty
