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

### Actian Vector RAG
A vector database layer using Actian for semantic document search. Enables natural language querying over UN humanitarian briefs, funding reports, and crisis assessments with retrieval-augmented generation.

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
| Vector Search | Actian Vector Database |
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
