# TRIAGE: Global AI Capital Allocation Engine

> **A Databricks-powered intelligence layer that tells Sovereign Wealth Funds and humanitarian organizations exactly where in the world to deploy capital — before the crisis peaks.**

---

## The Problem

Humanitarian funding is still largely driven by media attention and political optics — not data. The result is a world where high-visibility crises receive $10 for every $1 of urgent but "quiet" need, while statistically severe but under-reported regions starve for funding. This is not just inefficient. It costs lives.

## The Solution

TRIAGE is a full-stack analytical platform that ingests multi-source humanitarian datasets, runs a principal component analysis (PCA) scoring model in Databricks, and produces a single, defensible **Crisis Severity Score** for every tracked country — updated in real time. When a decision-maker enters a deployment budget, the AI engine runs a proportional capital allocation algorithm and outputs a ranked priority list of exactly which regions should receive funds, and how much.

No guesswork. No media bias. Pure data-driven triage.

---

## Architecture

```
Raw Humanitarian Data (UN FTS, UCDP, IDMC)
           |
           v
  Databricks Unity Catalog
  [01_triage_pca_scoring.py]
  PCA Dimensionality Reduction
  + Crisis Severity Score (0-100)
           |
           v
  Streamlit Live Dashboard [app/main.py]
  - Orthographic Crisis Map (Plotly)
  - 3 KPI Cards (Live severity, crises tracked, highest zone)
  - AI Capital Allocation Engine (severity-weighted proportional fund distribution)
  - Gemini AI Intelligence Assistant (domain-restricted analytical chatbot)
           |
           v
  Vector RAG Layer [vector_rag/actian_search.py]
  Actian Vector Database for document-grounded Q&A
```

---

## Key Features

### Real-Time Crisis Intelligence Map
An interactive orthographic globe rendered in Plotly, where every bubble represents an active crisis zone. Color-coded by funding coverage ratio. Hover for granular statistics. Rotatable, draggable, and live-updated from the Databricks cluster.

### PCA-Based Severity Scoring
The model ingests five dimensions of crisis severity — displacement, conflict intensity, food insecurity, funding gap, and IDP population — and reduces them into a single normalized Crisis Severity Score using Principal Component Analysis. The methodology is statistically defensible and reproducible.

### AI Capital Allocation Engine
A sovereign wealth fund UX. Enter a deployment budget (e.g., $200M), click the button, and the system runs a severity-proportional allocation algorithm across the top 10 most critical regions. The output is a clean, ranked manifest: which countries, in what order, at what funding level. Every number is derived from factual Databricks data — zero synthetic approximations.

### Gemini AI Intelligence Assistant
A domain-restricted conversational AI powered by the Gemini 2.5 API. The assistant is grounded in live crisis context from the Databricks cluster and is authorized only to answer quantitative questions about funding, geopolitical risk, and capital allocation strategy. Jargon-free. Professional. No hallucinations outside domain scope.

### Actian Vector RAG — 3-Persona Crisis Intelligence
A vector database layer powered by Actian VectorAI DB (Dockerized) with TF-IDF embeddings across 2,388 unique historical UN humanitarian documents from three authoritative sources:

- **HRP**: UN Humanitarian Response Plans (910 plans)
- **ReliefWeb**: Crisis figures by country and year (18,041 rows)
- **FTS**: UN OCHA Financial Tracking Service funding data (3,669 rows)

When a crisis is selected, the system queries Actian VectorAI via cosine similarity search and returns the most comparable historical crises. These results power three persona-specific intelligence briefs:

- **💰 Donor Brief** — Emotional hook + ROI analysis + historical proof that funding stabilizes crises. Motivates donors to act now with data-backed urgency.
- **📰 Journalist Brief** — The untold story angle + verified UN data points ready for publication. Shows which crises are severely undercovered relative to their severity score.
- **🔵 UN Coordinator Brief** — Operational intelligence including priority clusters, funding gap analysis, and comparable past response strategies that worked.

All briefs are generated in real time from 2,388 embedded UN documents. No hallucinations. Every comparable crisis is a real historical record from UN OCHA data.

---

## Data Sources

| Source | Description |
|---|---|
| UN OCHA Financial Tracking Service (FTS) | Funding requirements and contributions by crisis |
| UCDP Georeferenced Event Dataset (GED) | Conflict events with geographic precision |
| IDMC Internal Displacement Database | IDP and refugee displacement statistics |
| IPC Global Acute Food Insecurity | Food insecurity classifications by country |
| UN ReliefWeb Project Database | Active humanitarian project locations |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Warehouse | Databricks Unity Catalog |
| ML Pipeline | PySpark, Scikit-learn (PCA) |
| Live Dashboard | Streamlit |
| Data Visualization | Plotly Express / Graph Objects |
| AI Assistant | Google Gemini 2.5 Flash |
| Vector Search | Actian VectorAI DB (Docker) + TF-IDF embeddings |
| Optimizer Model | SciPy SLSQP (Allocation) |
| Environment | Python 3.10+, dotenv |

---

## Getting Started

### Prerequisites

```bash
# Python 3.10+
# Active Databricks workspace with Unity Catalog enabled
# Google Gemini API key
# Actian database instance (optional, for RAG layer)
```

### Installation

```bash
git clone https://github.com/your-org/triage-hacklytics.git
cd triage-hacklytics

# Create a virtual environment
python -m venv databricks/venv
source databricks/venv/bin/activate

# Install dependencies
pip install streamlit pandas numpy plotly scipy scikit-learn \
            python-dotenv databricks-sql-connector google-generativeai
```

### Configuration

Create a `.env` file in the project root:

```env
DATABRICKS_SERVER_HOSTNAME=your-workspace.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
DATABRICKS_TOKEN=your-databricks-personal-access-token
GEMINI_API_KEY=your-google-gemini-api-key
```

### Running the Dashboard

```bash
source databricks/venv/bin/activate
streamlit run app/main.py
```

The dashboard will connect to your Databricks cluster automatically. If the cluster is unavailable, the application falls back gracefully to local CSV data in `assets/`.

### Running the VectorAI RAG Pipeline
```bash
# Step 1: Start Actian VectorAI DB (Mac M-chip — requires Rosetta 2)
docker run --platform linux/amd64 -p 50051:50051 williamimoh/actian-vectorai-db:1.0b

# Step 2: Populate VectorAI with UN historical documents (run once)
cd vector_rag
python triage_rag_2.py

# Step 3: Start the dashboard — RAG layer is now active
streamlit run app/main.py
```

### Running the Databricks Scoring Pipeline

```python
# Upload and run in Databricks
databricks/01_triage_pca_scoring.py
```

This notebook reads from the raw humanitarian source tables, runs PCA, and writes the scored output to `hacklytics_db.default.triage_master_optimized`.

---

## Project Structure

```
triage-hacklytics/
├── app/
│   ├── main.py                    # Live Streamlit dashboard
│   └── genai_briefs.py            # Gemini brief generation module
├── assets/
│   ├── triage_master_optimized.csv  # Fallback scored data
│   ├── header.webp                  # Dashboard header image
│   └── [raw source datasets]
├── databricks/
│   └── 01_triage_pca_scoring.py   # PCA scoring pipeline (Spark)
├── models/
│   └── allocation_optimizer.py    # SLSQP capital allocation model
├── vector_rag/
│   └── actian_search.py           # Actian vector DB search layer
│   ├── triage_rag_2.py            # One-time ingestion script (populates VectorAI)
│   └── triage_vectorizer.pkl      # Saved TF-IDF vectorizer (generated at runtime)
└── .env                           # API credentials (not committed)
```

---

## Design Philosophy

TRIAGE operates on a single principle: **every output must be traceable to data**. There are no simulated multipliers, no assumed shortfall proxies, and no hardcoded heuristics in the capital allocation model. Every severity score is a mathematically derived output of the PCA model. Every funding recommendation is a strict proportional derivation of those scores.

This makes the platform defensible in front of fund managers, UN bodies, and academic reviewers alike.

---

## Hacklytics 2026

Built at Hacklytics 2026, Georgia Tech's premier data science hackathon. TRIAGE demonstrates that with the right data infrastructure, capital allocation in humanitarian crises can be as rigorous as any institutional investment strategy.

---

*TRIAGE is a research and demonstration prototype. It is not intended as a primary decision-support tool without peer review and domain expert validation.*
