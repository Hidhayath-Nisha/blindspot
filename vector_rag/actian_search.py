"""
TRIAGE - Semantic Vector Search Layer (Actian VectorAI DB)

Uses AsyncCortexClient (recommended for M1/ARM compatibility over sync CortexClient).
Falls back to in-memory cosine similarity if Actian is unavailable.

Docker: cd actian-vectorAI-db-beta && docker compose up
"""

import asyncio
import logging
import numpy as np

COLLECTION_NAME = "triage_crises"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

_memory_index = {
    "vectors": None,
    "payloads": []
}


def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _row_to_text(row: dict) -> str:
    parts = [f"Country: {row.get('Country_Name', row.get('iso3', 'Unknown'))}"]
    if 'Crisis_Severity_Score' in row:
        parts.append(f"Crisis Severity Score: {row['Crisis_Severity_Score']:.2f}")
    if 'Funding_Ratio' in row:
        parts.append(f"Funding Coverage: {row['Funding_Ratio']:.1f}%")
    if 'funding_required' in row:
        parts.append(f"Funding Required: ${float(row.get('funding_required', 0)):,.0f}")
    if 'funding_received' in row:
        parts.append(f"Funding Received: ${float(row.get('funding_received', 0)):,.0f}")
    return ". ".join(parts)


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norms @ query_norm


async def _actian_ingest_async(vectors, payloads):
    from cortex import AsyncCortexClient, DistanceMetric
    async with AsyncCortexClient("localhost:50051") as client:
        await client.health_check()
        if await client.has_collection(COLLECTION_NAME):
            await client.delete_collection(COLLECTION_NAME)
        await client.create_collection(
            name=COLLECTION_NAME,
            dimension=EMBEDDING_DIM,
            distance_metric=DistanceMetric.COSINE,
        )
        await client.batch_upsert(
            collection=COLLECTION_NAME,
            ids=list(range(len(payloads))),
            vectors=[v.tolist() for v in vectors],
            payloads=payloads,
        )
    return len(payloads)


async def _actian_search_async(query_vector, top_k):
    from cortex import AsyncCortexClient
    async with AsyncCortexClient("localhost:50051") as client:
        results = await client.search(
            collection=COLLECTION_NAME,
            query=query_vector.tolist(),
            top_k=top_k,
        )
    return [r.payload for r in results]


def ingest_dataframe(df):
    """
    Embeds all crisis rows and stores in Actian VectorAI DB (async client).
    Falls back to in-memory index if Actian is unavailable.
    Always returns True — semantic search will work either way.
    """
    global _memory_index

    model = _get_embedding_model()
    rows = df.to_dict(orient="records")
    texts = [_row_to_text(r) for r in rows]
    vectors = model.encode(texts, show_progress_bar=False)

    payloads = [
        {
            "iso3": str(r.get("iso3", "")),
            "country": str(r.get("Country_Name", r.get("iso3", ""))),
            "severity": float(r.get("Crisis_Severity_Score", 0)),
            "funding_ratio": float(r.get("Funding_Ratio", 0)),
            "funding_required": float(r.get("funding_required", 0)),
            "funding_received": float(r.get("funding_received", 0)),
            "text": texts[i],
        }
        for i, r in enumerate(rows)
    ]

    # Try Actian with AsyncCortexClient
    try:
        n = asyncio.run(_actian_ingest_async(vectors, payloads))
        logging.info(f"Actian VectorAI DB: Ingested {n} crisis records.")
    except Exception as e:
        logging.warning(f"Actian unavailable, using in-memory index: {e}")

    # Always build in-memory fallback
    _memory_index["vectors"] = vectors
    _memory_index["payloads"] = payloads
    logging.info(f"In-memory index: {len(rows)} crisis records ready.")

    return True


def search_relevant_context(query: str, top_k: int = 5) -> str:
    """
    Semantically searches for the most relevant crisis records.
    Tries Actian AsyncCortexClient first, falls back to in-memory cosine similarity.
    """
    model = _get_embedding_model()
    query_vector = model.encode([query], show_progress_bar=False)[0]

    results = []
    actian_used = False

    try:
        results = asyncio.run(_actian_search_async(query_vector, top_k))
        actian_used = True
    except Exception:
        pass

    if not results and _memory_index["vectors"] is not None:
        scores = _cosine_similarity(query_vector, _memory_index["vectors"])
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [_memory_index["payloads"][i] for i in top_indices]

    if not results:
        return None

    source = "Actian VectorAI DB" if actian_used else "Semantic Index"
    lines = [f"Relevant Crisis Data [{source}]:"]
    for p in results:
        lines.append(
            f"- {p['country']} ({p['iso3']}): "
            f"Severity={p['severity']:.2f}, "
            f"Funding Coverage={p['funding_ratio']:.1f}%, "
            f"Required=${p['funding_required']:,.0f}, "
            f"Received=${p['funding_received']:,.0f}"
        )
    return "\n".join(lines)
