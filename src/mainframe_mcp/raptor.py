"""RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Clusters tier-0 chunks by semantic similarity, generates tier-1 summary chunks
via the consolidator, embeds them, and inserts into the store.

Approach:
1. Group chunks by doc_path (summaries are per-document)
2. Within each doc, cluster chunks using agglomerative clustering on embeddings
3. For each cluster of 3+ chunks, generate a summary via Llama-3.2-3B
4. Embed the summary and insert as tier-1 chunk linked to source chunks
"""

import logging
import uuid
from datetime import datetime

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from mainframe_mcp.store import SESSION_SOURCE_TYPE

logger = logging.getLogger(__name__)


def cluster_chunks(
    embeddings: list[list[float]],
    chunk_ids: list[str],
    distance_threshold: float = 0.8,
    min_cluster_size: int = 3,
) -> list[list[int]]:
    """Cluster chunk indices by cosine similarity using agglomerative clustering.

    Returns list of clusters, each a list of indices into the input arrays.
    Only returns clusters with >= min_cluster_size members.
    """
    n = len(embeddings)
    if n < min_cluster_size:
        return []

    emb_matrix = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine distance
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_matrix = emb_matrix / norms

    # Cosine distance = 1 - cosine_similarity
    distances = pdist(emb_matrix, metric="cosine")

    # Agglomerative clustering with distance threshold
    Z = linkage(distances, method="average")
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # Group indices by cluster label
    from collections import defaultdict
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)

    # Filter to clusters meeting min size
    return [indices for indices in groups.values() if len(indices) >= min_cluster_size]


def build_raptor_summaries(
    store,
    embedder,
    consolidator,
    distance_threshold: float = 0.8,
    min_cluster_size: int = 3,
    max_cluster_size: int = 15,
) -> dict:
    """Build tier-1 RAPTOR summaries for all documents in the store.

    Returns stats about what was built.
    """
    # Delete existing tier-1 chunks (rebuild from scratch)
    try:
        store.table.delete("tier = 1")
        logger.info("Deleted existing tier-1 summaries")
    except Exception:
        pass  # No tier-1 chunks yet

    # Load all tier-0 chunks grouped by doc_path
    df = store.table.to_pandas()
    tier0 = df[df["tier"] == 0]
    logger.info(f"Building RAPTOR summaries for {tier0['doc_path'].nunique()} docs, {len(tier0)} chunks")

    total_clusters = 0
    total_summaries = 0
    skipped_small = 0

    skipped_session = 0
    for doc_path, doc_chunks in tier0.groupby("doc_path"):
        # Skip session captures: don't spend consolidator inference summarizing
        # low-signal session logs or inject tier-1 "session" summaries into the
        # corpus (plan §6 / risk H1').
        if doc_chunks["source_type"].iloc[0] == SESSION_SOURCE_TYPE:
            skipped_session += 1
            continue
        if len(doc_chunks) < min_cluster_size:
            skipped_small += 1
            continue

        # Get embeddings and chunk IDs for this doc
        chunk_embeddings = doc_chunks["vector"].tolist()
        chunk_ids = doc_chunks["chunk_id"].tolist()
        chunk_texts = doc_chunks["text"].tolist()
        chunk_headings = doc_chunks["heading"].tolist()
        source_type = doc_chunks["source_type"].iloc[0]
        doc_id = doc_chunks["doc_id"].iloc[0]

        # Cluster
        clusters = cluster_chunks(
            chunk_embeddings, chunk_ids,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
        )

        if not clusters:
            continue

        total_clusters += len(clusters)

        for cluster_indices in clusters:
            # Cap cluster size
            if len(cluster_indices) > max_cluster_size:
                cluster_indices = cluster_indices[:max_cluster_size]

            # Gather texts for this cluster
            cluster_texts = [chunk_texts[i] for i in cluster_indices]
            cluster_headings = [chunk_headings[i] for i in cluster_indices]
            cluster_chunk_ids = [chunk_ids[i] for i in cluster_indices]

            # Generate summary
            try:
                summary_text = consolidator.summarize_cluster(
                    cluster_texts, cluster_headings, doc_path
                )
            except Exception as e:
                logger.warning(f"Summary generation failed for {doc_path}: {e}")
                continue

            if not summary_text or len(summary_text) < 20:
                continue

            # Embed the summary
            summary_embedding = embedder.embed([summary_text])[0]

            # Create a combined heading from cluster headings
            unique_headings = list(dict.fromkeys(h for h in cluster_headings if h and h != "(no heading)"))
            summary_heading = " / ".join(unique_headings[:3]) or "(cluster summary)"

            # Insert as tier-1 chunk
            cluster_id = str(uuid.uuid4())[:12]
            now = datetime.now().isoformat()

            row = {
                "chunk_id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "doc_path": doc_path,
                "tier": 1,
                "parent_id": ",".join(cluster_chunk_ids),  # links to source chunks
                "cluster_id": cluster_id,
                "text": summary_text,
                "vector": summary_embedding,
                "heading": summary_heading,
                "chunk_index": -1,  # not a positional chunk
                "char_start": 0,
                "char_end": 0,
                "token_count": len(summary_text.split()),
                "content_hash": "",
                "source_type": source_type,
                "retrieval_count": 0,
                "created_at": now,
                "pipeline_version": "1.0.0-raptor",
            }
            store.table.add([row])
            total_summaries += 1

        logger.info(f"  {doc_path}: {len(clusters)} clusters, summaries generated")

    # Refresh table handle
    store.table = store.db.open_table("chunks")

    # Rebuild FTS index to include new summaries
    store._ensure_fts_index()

    stats = {
        "docs_processed": tier0["doc_path"].nunique(),
        "docs_skipped_small": skipped_small,
        "docs_skipped_session": skipped_session,
        "clusters_found": total_clusters,
        "summaries_generated": total_summaries,
        "total_tier1_chunks": total_summaries,
    }
    logger.info(f"RAPTOR build complete: {stats}")
    return stats
