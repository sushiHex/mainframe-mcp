"""LanceDB store — a derived, rebuildable cache of chunk rows.

- One `merge_insert` per batch on the deterministic `chunk_key`; the
  not-matched-by-source delete is scoped to the batch's documents by hex
  `doc_id`, so a shrinking document loses only its stale keys and every
  other document is untouched.
- The index is self-describing: `ledger()` derives {doc_path: file_hash}
  from a projected scan. There is no manifest.
- Readers refresh via `read_consistency_interval`; the cleanup window is
  asserted far larger than that interval so a reader is never pinned to a
  version that compaction deletes.
- A fingerprint answers "what produced these rows?": the FINGERPRINT of the
  settings they were built under, so a changed chunk size or embedder is
  detected instead of silently mixing two regimes into one index. Native
  encoding also persists that identity in the table schema.
- There is no ownership protocol, because there is no collision to arbitrate:
  v2 keeps its rows in `<mainframe_dir>/index.lancedb` while the still-
  installed v1 (`mainframe_mcp`) keeps its own `.lancedb`. Two namespaces are
  simpler than teaching two shipping implementations not to destroy each
  other's `chunks` table.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import lancedb
import pyarrow as pa

from mainframe.core.classify import CAPTURE_LANE
from mainframe.memory.atomic import write_atomic

logger = logging.getLogger(__name__)

TABLE = "chunks"
_HEX32 = re.compile(r"[0-9a-f]{32}")

# v2's index directory under `mainframe_dir`. Named here, read by everything
# that opens the live index (the daemon, the eval harness) so the namespace is
# one value rather than a string repeated per call site.
INDEX_DIRNAME = "index.lancedb"
FORMAT_VERSION = 1

# The fingerprint fields whose change makes the STORED VECTORS incomparable
# to a freshly embedded query, rather than merely making the index
# heterogeneous. See `IndexStaleError` for why that asymmetry exists.
EMBEDDER_FIELD = "embedder_model"
ENCODING_FIELD = "embedding_contract"
_NATIVE_FINGERPRINT = b'mainframe.native_fingerprint'


class IndexStaleError(RuntimeError):
    """These rows were built under settings that are no longer in force.

    Drift used to be computed, reported in `status`, and then permitted — so
    one table could hold rows chunked under two different rules, or vectors
    from two same-dimension embedders, while the marker claimed otherwise.
    Visibility is not a guarantee: the index is STALE, and the only honest
    repair is an offline rebuild.

    The refusal is deliberately ASYMMETRIC. Every WRITE is refused under any
    drift, because a write is what mixes the regimes. A READ is refused only
    when the EMBEDDER or its ENCODING CONTRACT changed, because then the
    stored vectors no longer live in the same space as the query's; a chunk
    size or frontmatter change leaves every row still meaningful, merely
    inconsistent with what a rebuild would produce, and refusing to answer
    would be worse than answering from a heterogeneous index."""


@dataclass
class DocRows:
    doc_id: str
    doc_path: str
    rows: list  # full-schema dicts; empty for a file that chunks to nothing


@dataclass
class BatchResult:
    upserted_rows: int
    docs: int
    deleted_docs: int


class Store:
    def __init__(self, db_path, embedding_dim: int | None = None, tier_boost: dict | None = None,
                 recency_weight: float = 0.0, recency_halflife_days: float = 90.0,
                 read_consistency_seconds: int = 30, cleanup_minutes: int = 120,
                 fingerprint: dict | None = None):
        self.reconfigure(tier_boost or {}, recency_weight, recency_halflife_days)
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self.fingerprint = fingerprint
        self.config_drift: dict = {}
        # The last observed keyword-search failure, or None. Written by `search`
        # (readers) and cleared by `_ensure_indexes` (the writer thread): one
        # small assignment either way, never read-modify-write, so the honest
        # weak guarantee - a reader may observe a repair a moment late - costs
        # nothing and needs no lock on the search path.
        self.fts_error: str | None = None
        self.cleanup = timedelta(minutes=cleanup_minutes)
        refresh = timedelta(seconds=read_consistency_seconds)
        if self.cleanup < refresh * 20:
            raise ValueError(
                f"cleanup_minutes ({cleanup_minutes}) must be far larger than "
                f"read_consistency_seconds ({read_consistency_seconds}) or a reader can be pinned "
                "to a version that compaction deletes")
        self.db = lancedb.connect(str(db_path), read_consistency_interval=refresh)
        self.table = None
        self._ensure_table()

    @staticmethod
    def search_kwargs_from_config(config: dict) -> dict:
        return {
            "tier_boost": config.get("tiers", {}),
            "recency_weight": config["search"].get("recency_weight", 0.0),
            "recency_halflife_days": config["search"].get("recency_halflife_days", 90),
            "read_consistency_seconds": config["index"].get("read_consistency_seconds", 30),
            "cleanup_minutes": config["index"].get("cleanup_minutes", 120),
        }

    @staticmethod
    def fingerprint_from_config(config: dict) -> dict:
        """The settings that decide what a row LOOKS like. Changing any of them
        leaves every existing row untouched while new documents enter a
        different regime, and nothing about the row itself records which regime
        produced it — so the index records them once, in the marker, and
        reports a difference as drift."""
        chunker = config.get("chunker", {})
        fingerprint = {
            "embedder_model": config.get("embedder", {}).get("model", ""),
            "chunk_size": int(chunker.get("chunk_size", 256)),
            "overlap_ratio": float(chunker.get("overlap_ratio", 0.35)),
            "strip_frontmatter": bool(chunker.get("strip_frontmatter", False)),
            "contextual": bool(config.get("contextual", {}).get("enabled", False)),
        }
        # drop_headings decides which ROWS exist at all, so it belongs in the
        # fingerprint too. Stored sorted: the filter's meaning is set-like, so
        # reordering a user's patterns is not a real change. Included only
        # when non-empty so an empty list (today's default) fingerprints
        # IDENTICALLY to a config from before this key existed — upgrading
        # must never force a rebuild on its own.
        drop_headings = sorted(chunker.get("drop_headings") or [])
        if drop_headings:
            fingerprint["drop_headings"] = drop_headings
        from mainframe.core.encoding import native_contract
        contract = native_contract(config)
        if contract:
            fingerprint[ENCODING_FIELD] = contract
        return fingerprint

    def reconfigure(self, tier_boost: dict, recency_weight: float, recency_halflife_days: float):
        self.tier_boost = tier_boost
        self._recency_weight = recency_weight
        self._recency_halflife = (recency_halflife_days
                                  if recency_halflife_days and recency_halflife_days > 0 else 90.0)

    # ---------- schema ----------

    def _schema(self) -> pa.Schema:
        schema = pa.schema([
            pa.field("chunk_key", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("doc_path", pa.string()),
            pa.field("project", pa.string()),
            pa.field("lane", pa.string()),
            pa.field("source_type", pa.string()),
            pa.field("file_hash", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("text", pa.string()),
            pa.field("heading", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("char_start", pa.int32()),
            pa.field("char_end", pa.int32()),
            pa.field("token_count", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
            pa.field("authored_at", pa.string()),
            pa.field("indexed_at", pa.string()),
        ])
        if (self.fingerprint or {}).get(ENCODING_FIELD):
            # Native identity belongs to the table that owns the vectors. Its
            # loss must not turn a sidecar rewrite into silent compatibility.
            schema = schema.with_metadata({_NATIVE_FINGERPRINT: json.dumps(self.fingerprint).encode('utf-8')})
        return schema

    def _table_names(self):
        resp = self.db.list_tables()
        return list(getattr(resp, "tables", resp))

    # ---------- fingerprint marker ----------

    def _marker_path(self) -> Path:
        """Sibling of the database directory, never inside it: LanceDB owns
        everything under `.lancedb` and is free to prune what it does not
        recognize."""
        return self.db_path.with_name(self.db_path.name + ".meta.json")

    def _read_marker(self) -> dict | None:
        embedded = (self.table.schema.metadata or {}).get(_NATIVE_FINGERPRINT) if self.table is not None else None
        if embedded is not None:
            try:
                fingerprint = json.loads(embedded)
                if not isinstance(fingerprint, dict) or not isinstance(fingerprint.get(ENCODING_FIELD), dict):
                    raise ValueError('native identity is incomplete')
            except (ValueError, TypeError):
                fingerprint = {ENCODING_FIELD: 'unreadable native identity'}
            return {'format_version': FORMAT_VERSION, 'fingerprint': fingerprint}
        try:
            data = json.loads(self._marker_path().read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            # A damaged marker records nothing about the rows, so it is treated
            # as absent: the current settings become the baseline again. It
            # cannot be evidence of anything else — nobody else writes here.
            logger.warning(f"index marker unreadable ({e}); re-establishing the fingerprint baseline")
            return None
        return data if isinstance(data, dict) else None

    def _write_marker(self):
        write_atomic(self._marker_path(), json.dumps(
            {"format_version": FORMAT_VERSION, "fingerprint": self.fingerprint or {}}, indent=2) + "\n")

    def _note_drift(self, marker: dict | None):
        stored = (marker or {}).get("fingerprint") or {}
        if self.fingerprint is None:
            if ENCODING_FIELD in stored:
                self.config_drift = {ENCODING_FIELD: [stored[ENCODING_FIELD], None]}
            return                          # this caller did not ask (eval harness, offline tools)
        if not stored:
            if ENCODING_FIELD in self.fingerprint:
                self.config_drift = {ENCODING_FIELD: [None, self.fingerprint[ENCODING_FIELD]]}
                return
            # Nothing was ever recorded for these rows (an index built before
            # the marker existed, or a damaged one): an unrecorded past cannot
            # be reported as drift, so the current settings become the baseline.
            self._write_marker()
            return
        fields = set(self.fingerprint)
        if ENCODING_FIELD in stored:
            fields.add(ENCODING_FIELD)  # A native -> legacy change must also refuse reads.
        self.config_drift = {k: [stored.get(k), self.fingerprint.get(k)] for k in fields
                             if stored.get(k) != self.fingerprint.get(k)}
        for field, (was, now) in self.config_drift.items():
            # The marker keeps the OLD values until a rebuild actually
            # re-creates the rows: it describes what produced them, not what
            # the config says today.
            logger.warning("index config drift: %s was %r when this index was built, is %r now — "
                           "the index is stale until `mainframe rebuild`", field, was, now)

    def refuse_if_stale(self, operation: str):
        """Raise unless this index was built under the settings now in force.

        One predicate (`config_drift`), one raiser, two call sites. The mutation
        itself calls it, because that is where the invariant lives; the pipeline
        calls it FIRST, before it embeds anything, because a refusal that
        arrives after a GPU pass costs that pass every debounce window."""
        if not self.config_drift:
            return
        detail = "; ".join(f"{k}: {was!r} -> {now!r}" for k, (was, now) in sorted(self.config_drift.items()))
        raise IndexStaleError(
            f"{operation} refused: this index was built under different settings ({detail}). "
            "Stop the daemon and run `mainframe rebuild`, or restore the previous settings.")

    def _create_table(self, dim: int):
        self.embedding_dim = dim
        self.table = self.db.create_table(TABLE, schema=self._schema())
        self.config_drift = {}
        self._write_marker()
        logger.info(f"Created chunks table (dim={dim})")

    def _ensure_table(self):
        if TABLE in self._table_names():
            self.table = self.db.open_table(TABLE)
            on_disk = self.table.schema.field("vector").type.list_size
            if self.embedding_dim is not None and self.embedding_dim != on_disk:
                logger.warning("index vector dim %d != configured %d — run a rebuild",
                                on_disk, self.embedding_dim)
            self.embedding_dim = on_disk    # the index is truth
            self._note_drift(self._read_marker())
        elif self.embedding_dim is not None:
            self._create_table(self.embedding_dim)
        else:
            self.table = None   # deferred to the first upsert (the daemon learns dim from the embedder)

    def _has_index(self, column: str) -> bool:
        try:
            return any(list(getattr(i, "columns", [])) == [column] for i in self.table.list_indices())
        except Exception:
            return False

    def _ensure_indexes(self) -> bool:
        """FTS on text + BTREE on chunk_key, created lazily once rows exist
        (an empty table cannot be indexed). Idempotent.

        An index that EXISTS is not an index that ANSWERS. After a large delete
        the FTS postings can still reference a fragment the dataset no longer
        has, and every query matching those rows fails with "the input to a take
        operation specified fragment id N but this fragment does not exist" -
        while `_has_index("text")` keeps reporting the index as present, so this
        method used to leave it broken forever and `optimize` returned ok beside
        it. Recreating it rebuilds the postings from the current dataset, which
        is what actually clears the dangling reference.

        The trigger is an OBSERVED failure (`fts_error`), not a probe: a
        synthetic query only fails if it happens to match rows in the missing
        fragment, so a probe reports health it has not established. The real
        workload is the only honest detector, and `search` records what it sees.
        """
        if self.table is None:
            return True
        try:
            if self.table.count_rows() == 0:
                return True
            if not self._has_index("text"):
                self.table.create_fts_index("text", replace=True)
                logger.info("Created FTS index on text")
            elif self.fts_error is not None:
                self.table.create_fts_index("text", replace=True)
                logger.info(f"Recreated FTS index on text after a failure: {self.fts_error}")
                self.fts_error = None
            if not self._has_index("chunk_key"):
                self.table.create_scalar_index("chunk_key", index_type="BTREE")
                logger.info("Created BTREE index on chunk_key")
            return True
        except Exception as e:
            logger.warning(f"index check/create failed (non-fatal): {e}")
            return False

    # ---------- writes ----------

    @staticmethod
    def _in(ids) -> str:
        ids = list(ids)
        for i in ids:
            if not _HEX32.fullmatch(i):
                raise ValueError(f"not a 32-hex doc_id: {i!r}")
        return "doc_id IN (" + ", ".join(f"'{i}'" for i in ids) + ")"

    def upsert_batch(self, docs: list, deleted_doc_ids=()) -> BatchResult:
        """ONE merge for the batch. `docs` must contain only documents whose
        every stage succeeded (the caller leaves failed files out, so their
        previous rows survive). A DocRows with no rows means "this document is
        now empty": its old rows are deleted. A doc_id that appears in both
        `docs` and `deleted_doc_ids` was just upserted, so the stale delete
        request for it is dropped — the upsert wins."""
        self.refuse_if_stale("indexing")
        rows = [r for d in docs for r in d.rows]
        if self.table is None:
            if not rows:
                return BatchResult(0, len(docs), 0)
            self._create_table(len(rows[0]["vector"]))
        if rows and len(rows[0]["vector"]) != self.embedding_dim:
            raise ValueError(
                f"embedding dim {len(rows[0]['vector'])} != index dim {self.embedding_dim}; "
                "run `mainframe rebuild`")
        ids = [d.doc_id for d in docs]
        if rows:
            (self.table.merge_insert("chunk_key")
                 .when_matched_update_all()
                 .when_not_matched_insert_all()
                 .when_not_matched_by_source_delete(self._in(ids))
                 .execute(pa.Table.from_pylist(rows, schema=self.table.schema)))
        elif ids:
            self.table.delete(self._in(ids))
        deleted = set(deleted_doc_ids) - {d.doc_id for d in docs}
        if deleted:
            self.table.delete(self._in(deleted))
        logger.info(f"upsert_batch: {len(rows)} rows across {len(docs)} docs, "
                    f"{len(deleted)} docs deleted")
        return BatchResult(len(rows), len(docs), len(deleted))

    def optimize(self) -> bool:
        """Compact and clean up versions older than the cleanup window, then
        ensure the FTS/BTREE indexes exist. Returns True only when both steps
        succeeded (or there is no table yet)."""
        if self.table is None:
            return True
        ok = True
        try:
            self.table.optimize(cleanup_older_than=self.cleanup)
        except TypeError:  # signature drift
            try:
                self.table.optimize()
            except Exception as e:
                logger.warning(f"optimize fallback (no-arg) failed (non-fatal): {e}")
                ok = False
        except Exception as e:
            logger.warning(f"optimize failed (non-fatal): {e}")
            ok = False
        idx = self._ensure_indexes()
        return ok and idx

    def drop(self):
        """A rebuild is the ONE moment the fingerprint may move: every row is
        about to be produced again under the current settings, so the marker is
        rewritten here and the drift cleared — otherwise the very next
        `upsert_batch` would refuse the rebuild it is performing.

        The table is NOT re-created. Re-creating it pinned the OLD vector
        dimension, which is the one drift a rebuild most obviously has to
        survive: production opens with `embedding_dim=None` and learns the dim
        from disk, so swapping to a differently-sized embedder emptied the
        index and then died on the first batch with `embedding dim 128 !=
        index dim 64` — telling the operator to run the rebuild they were
        already running. Forgetting the dim too lets the first batch create the
        table at the dimension the new embedder actually produces."""
        if TABLE in self._table_names():
            self.db.drop_table(TABLE)
        self.table = None
        self.embedding_dim = None
        self.config_drift = {}
        self._write_marker()

    # ---------- reads ----------
    #
    # Readers SHARE the writer's `self.table` handle across threads (search runs
    # on asyncio.to_thread workers while merge/optimize run on the mutation
    # thread), so every read method snapshots the attribute ONCE into a local
    # and uses that: cheap hygiene against a handle that could be replaced
    # mid-call and raise an uncaught AttributeError.

    def _scan(self, columns: list, table=None):
        """Projected read (never the vector column unless asked)."""
        tbl = self.table if table is None else table
        return tbl.to_lance().to_table(columns=columns).to_pandas()

    def count_rows(self) -> int:
        tbl = self.table
        if tbl is None:
            return 0
        return tbl.count_rows()

    def ledger(self) -> dict:
        """{doc_path: file_hash} for every indexed document.

        RAISES on a failed scan. `{}` is a real answer — "this index holds
        nothing" — and a caller that cannot tell it from "I could not look"
        prunes against a ledger it never read. Only the no-table and empty-table
        cases return it."""
        tbl = self.table
        if tbl is None:
            return {}
        try:
            if tbl.count_rows() == 0:
                return {}
            df = self._scan(["doc_path", "file_hash"], table=tbl)
            return dict(zip(df["doc_path"], df["file_hash"]))
        except Exception as e:
            logger.warning(f"ledger scan failed: {e}")
            raise

    def search(self, query_embedding, top_k: int = 20, include_captures: bool = False,
               query_text: str = "") -> list:
        """Hybrid vector + FTS candidates, tier-boosted, stably ordered."""
        import pandas as pd

        if {EMBEDDER_FIELD, ENCODING_FIELD}.intersection(self.config_drift):
            self.refuse_if_stale("search")
        tbl = self.table
        if tbl is None:
            return []

        where = None if include_captures else f"lane != '{CAPTURE_LANE}'"

        def _vec():
            q = tbl.search(query_embedding).limit(top_k)
            if where:
                q = q.where(where)
            return q.to_pandas()

        def _fts():
            return tbl.search(query_text, query_type="fts").limit(top_k).to_pandas()

        try:
            vec = _vec()
        except Exception as e:
            logger.warning(f"vector search failed ({e}); checkout_latest and retry")
            tbl.checkout_latest()
            vec = _vec()

        fts = pd.DataFrame()
        if query_text:
            try:
                fts = _fts()
            except Exception as e:
                logger.debug(f"FTS search failed ({e}); checkout_latest and retry")
                try:
                    tbl.checkout_latest()
                    fts = _fts()
                except Exception as e2:
                    # A caught FTS error is a DEGRADED ANSWER, not a debug
                    # detail: this query is about to be answered vector-only and
                    # nothing in the response says so. Warn once per outage -
                    # per query would be one line per search, forever - and
                    # leave the reason where `optimize` and `status` can see it.
                    if self.fts_error is None:
                        logger.warning(f"keyword search is failing; answering vector-only: {e2}")
                    self.fts_error = str(e2)

        if not fts.empty and not vec.empty:
            new = fts[~fts["chunk_key"].isin(set(vec["chunk_key"]))]
            if not new.empty:
                new = new.copy()
                new["_distance"] = float(vec["_distance"].max()) * 1.01
                vec = pd.concat([vec, new], ignore_index=True)
        elif vec.empty and not fts.empty:
            vec = fts.copy()
            vec["_distance"] = 1.0
        if vec.empty:
            return []
        if where and "lane" in vec.columns:
            vec = vec[vec["lane"] != CAPTURE_LANE]
            if vec.empty:
                return []

        boost = vec["source_type"].map(self.tier_boost).fillna(1.0)
        vec = vec.copy()
        vec["boosted_score"] = vec["_distance"] * boost
        if self._recency_weight > 0 and "authored_at" in vec.columns:
            ts = pd.to_datetime(vec["authored_at"], errors="coerce")
            now = pd.Timestamp.now(tz=ts.dt.tz) if getattr(ts.dtype, "tz", None) else pd.Timestamp.now()
            ages = (now - ts).dt.total_seconds() / 86400.0
            factors = 1.0 - self._recency_weight * (0.5 ** (ages.clip(lower=0) / self._recency_halflife))
            vec["boosted_score"] *= factors.fillna(1.0)
        vec = vec.sort_values(["boosted_score", "doc_path", "chunk_index"], kind="mergesort").head(top_k)
        vec = vec.drop(columns=["vector"], errors="ignore")
        return vec.to_dict("records")

    def explain_key_lookup(self, key: str):
        """The query plan for a chunk_key point lookup (None if unsupported)."""
        tbl = self.table
        if tbl is None:
            return None
        try:
            return tbl.search().where(f"chunk_key = '{key}'").limit(1).explain_plan()
        except Exception:
            return None

    @property
    def keyword_search(self) -> dict:
        """Whether the keyword half of hybrid search is answering.

        `ok` is OBSERVATIONAL: it means no FTS query has failed since the last
        repair, not that one has been verified to work. Verification is not
        available - a synthetic probe passes on an index whose real queries
        fail, because it only breaks if it happens to match the missing rows.
        Claiming health from such a probe is the defect this reports, not a fix
        for it, so the weaker true statement is the one published.
        """
        return {"ok": self.fts_error is None, "error": self.fts_error}

    def index_health(self) -> dict:
        tbl = self.table
        if tbl is None:
            return {"fragments": 0, "versions": 0, "rows": 0, "indexes": [],
                    "keyword_search": self.keyword_search}
        try:
            ds = tbl.to_lance()
            indexes = [{"columns": list(getattr(i, "columns", [])), "type": str(getattr(i, "index_type", ""))}
                       for i in tbl.list_indices()]
            return {"fragments": len(ds.get_fragments()), "versions": len(tbl.list_versions()),
                    "rows": ds.count_rows(), "indexes": indexes,
                    "keyword_search": self.keyword_search}
        except Exception as e:
            logger.warning(f"index_health failed: {e}")
            return {"fragments": -1, "versions": -1, "rows": -1, "indexes": [],
                    "keyword_search": self.keyword_search}

    def stats(self) -> dict:
        empty = {"total_chunks": 0, "by_lane": {}, "by_source_type": {}, "unique_docs": 0}
        tbl = self.table
        if tbl is None:
            return empty
        try:
            if tbl.count_rows() == 0:
                return empty
            df = self._scan(["lane", "source_type", "doc_path"], table=tbl)
            return {"total_chunks": len(df),
                    "by_lane": {str(k): int(v) for k, v in df["lane"].value_counts().items()},
                    "by_source_type": {str(k): int(v) for k, v in df["source_type"].value_counts().items()},
                    "unique_docs": int(df["doc_path"].nunique())}
        except Exception as e:
            logger.warning(f"stats failed: {e}")
            return empty
