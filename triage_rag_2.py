"""
TRIAGE — Layer 4: Comparable Crisis RAG System
Uses TF-IDF embeddings (no torch/sentence-transformers required)
Stores vectors in Actian VectorAI via cortex client
"""

import pandas as pd
import numpy as np
from cortex import CortexClient, DistanceMetric
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ── CONFIG ─────────────────────────────────────────────────────────────────────
HRP_FILE        = "humanitarian-response-plans.csv"
RELIEFWEB_FILE  = "data_reliefweb_crisis_figures_data.csv"
FTS_FILE        = "fts_requirements_funding_global.csv"
VECTORIZER_FILE = "triage_vectorizer.pkl"
COLLECTION      = "crisis_documents"
DIMENSION       = 384

# ── STEP 1: LOAD FILES ─────────────────────────────────────────────────────────
print("Loading datasets...")

hrp = pd.read_csv(HRP_FILE, skiprows=1)
hrp.columns = ['code','internalId','startDate','endDate','planVersion',
                'categories','locations','years','origRequirements','revisedRequirements']

rw = pd.read_csv(RELIEFWEB_FILE)

fts = pd.read_csv(FTS_FILE, skiprows=1)
fts.columns = ['countryCode','id','name','code','typeId','typeName',
                'startDate','endDate','year','requirements','funding','percentFunded']

print(f"HRP: {len(hrp)} plans | ReliefWeb: {len(rw)} rows | FTS: {len(fts)} rows")


# ── STEP 2: BUILD DOCUMENT CORPUS ──────────────────────────────────────────────
print("\nBuilding document corpus...")
documents = []

# --- From HRP: one document per response plan ---
for _, row in hrp.iterrows():
    if pd.isna(row['planVersion']):
        continue
    orig    = row['origRequirements'] if pd.notna(row['origRequirements']) else 0
    revised = row['revisedRequirements'] if pd.notna(row['revisedRequirements']) else 0
    text = (
        f"Humanitarian Response Plan: {row['planVersion']} "
        f"Countries: {row['locations']} "
        f"Year: {row['years']} "
        f"Original Requirements USD: {orig} "
        f"Revised Requirements USD: {revised} "
        f"Plan Type: {row['categories']}"
    )
    documents.append({
        "source":  "HRP",
        "country": str(row['locations']),
        "year":    str(row['years']),
        "text":    text.strip()
    })

# --- From ReliefWeb: group by crisis + year to build crisis profiles ---
rw['year'] = pd.to_datetime(rw['figure_date'], errors='coerce').dt.year

for (crisis, iso3, year), group in rw.groupby(['crisis_name', 'crisis_iso3', 'year']):
    if pd.isna(year):
        continue
    figures_text = " | ".join([
        f"{r['figure_name']}: {r['figure_value']:,.0f}"
        for _, r in group.iterrows()
        if pd.notna(r['figure_value'])
    ])
    if not figures_text:
        continue
    text = (
        f"Crisis: {crisis} Country: {iso3} "
        f"Year: {int(year)} "
        f"Humanitarian Figures: {figures_text}"
    )
    documents.append({
        "source":  "ReliefWeb",
        "country": str(iso3),
        "year":    str(int(year)),
        "text":    text.strip()
    })

# --- From FTS: only include plans with actual funding > 0 ---
fts_clean = fts[
    fts['requirements'].notna() &
    fts['funding'].notna() &
    (fts['funding'].astype(float) > 0)  # FIX: exclude zero-funding rows
].copy()

for _, row in fts_clean.iterrows():
    if pd.isna(row['name']):
        continue
    pct = float(row['percentFunded']) if pd.notna(row['percentFunded']) else 0
    # FIX: 4-category funding status instead of binary
    status = (
        "Critically underfunded" if pct < 30 else
        "Severely underfunded"   if pct < 60 else
        "Partially funded"       if pct < 80 else
        "Successfully funded"
    )
    text = (
        f"Funding Plan: {row['name']} "
        f"Country: {row['countryCode']} "
        f"Year: {row['year']} "
        f"Requirements USD: {float(row['requirements']):,.0f} "
        f"Funding Received USD: {float(row['funding']):,.0f} "
        f"Percent Funded: {pct}% "
        f"Funding Status: {status}"
    )
    documents.append({
        "source":  "FTS",
        "country": str(row['countryCode']),
        "year":    str(row['year']),
        "text":    text.strip()
    })

doc_df = pd.DataFrame(documents)
doc_df = doc_df[doc_df['text'].str.len() > 50].reset_index(drop=True)
doc_df = doc_df.drop_duplicates(subset=['text']).reset_index(drop=True)  # FIX: remove duplicates
print(f"Total documents: {len(doc_df)}")
print(doc_df['source'].value_counts().to_string())


# ── STEP 3: TF-IDF EMBEDDINGS ──────────────────────────────────────────────────
print("\nBuilding TF-IDF embeddings...")

vectorizer = TfidfVectorizer(
    max_features=384,   # vector dimension
    ngram_range=(1, 2), # unigrams + bigrams for better matching
    sublinear_tf=True,  # dampen high frequency terms
    min_df=1
)

texts      = doc_df['text'].tolist()
embeddings = vectorizer.fit_transform(texts).toarray()  # shape: (n_docs, 384)
print(f"Embeddings shape: {embeddings.shape}")

# Save vectorizer locally — needed at query time
with open(VECTORIZER_FILE, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"Vectorizer saved to {VECTORIZER_FILE}")


# ── STEP 4: STORE IN ACTIAN VECTORAI ───────────────────────────────────────────
print("\nConnecting to Actian VectorAI...")

with CortexClient("localhost:50051") as client:
    version, _ = client.health_check()
    print(f"Connected: {version}")

    # FIX: always delete and recreate collection to avoid stale/duplicate data
    try:
        client.delete_collection(COLLECTION)
        print("Old collection deleted")
    except:
        pass

    client.create_collection(
        name=COLLECTION,
        dimension=DIMENSION,
        distance_metric=DistanceMetric.COSINE,
    )
    print("Collection created fresh")

    ids      = list(range(len(doc_df)))
    vectors  = [emb.tolist() for emb in embeddings]
    payloads = doc_df[['source', 'country', 'year', 'text']].to_dict('records')

    client.batch_upsert(COLLECTION, ids, vectors, payloads)
    print(f"Done. {len(doc_df)} documents stored in VectorAI")


# ── STEP 5: QUERY FUNCTION ─────────────────────────────────────────────────────
def find_comparable_crises(country: str,
                            severity_score: float,
                            total_idps: int,
                            food_phase: int,
                            funding_coverage_pct: float,
                            top_k: int = 5) -> list:
    """
    Find the top_k most comparable past crises to the given crisis profile.

    Parameters:
        country              : ISO3 country code e.g. 'SDN'
        severity_score       : 0-100 composite severity score
        total_idps           : total internally displaced persons
        food_phase           : IPC food security phase 1-5
        funding_coverage_pct : percent of needs funded e.g. 38.0
        top_k                : number of comparable crises to return

    Returns:
        List of dicts with country, year, summary, source, similarity_pct
    """

    # FIX: use 4-category status in query to match document style
    funding_status = (
        "Critically underfunded" if funding_coverage_pct < 30 else
        "Underfunded"   if funding_coverage_pct < 50 else
        "Partially funded"       if funding_coverage_pct < 75 else
        "Successfully funded"
    )

    # Build query text matching the style of embedded documents
    query = (
        f"Crisis Country: {country} "
        f"Severity: {severity_score} "
        f"Displaced persons: {total_idps:,} "
        f"Food security phase: {food_phase} "
        f"Percent funded: {funding_coverage_pct}% "
        f"Funding Status: {funding_status} "
        f"Underfunded emergency humanitarian response needed"
    )

    # Load vectorizer and embed the query
    with open(VECTORIZER_FILE, "rb") as f:
        vec = pickle.load(f)

    query_embedding = vec.transform([query]).toarray()[0].tolist()

    # Search Actian VectorAI
    with CortexClient("localhost:50051") as client:
        results    = client.search(COLLECTION, query_embedding, top_k=top_k)
        result_ids = [r.id for r in results]
        fetched    = client.get_many(COLLECTION, result_ids)

        return [{
            "country":        payload['country'],
            "year":           payload['year'],
            "summary":        payload['text'][:600],
            "source":         payload['source'],
            "similarity_pct": round(result.score * 100, 1)
        } for result, (_, payload) in zip(results, fetched)]


# ── STEP 6: TEST ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nTesting query: Yem crisis profile...")
    results = find_comparable_crises(
        country="AFG",
        severity_score=45,
        total_idps=2_000_00,
        food_phase=4,
        funding_coverage_pct=42.0
    )

    print(f"\nTop {len(results)} comparable past crises to Sudan:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['country']} ({r['year']}) — {r['similarity_pct']}% similar [{r['source']}]")
        print(f"   {r['summary'][:600]}")
        print()