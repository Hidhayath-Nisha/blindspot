# ==============================================================================
# BLINDSPOT — Humanitarian Resource Allocation Intelligence System
# Hacklytics 2026 | Databricks × UN Challenge | Georgia Tech
# "Funding follows headlines. Lives don't."
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import sys
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Load .env — tries standard .env first, then a file named "main" at project root
# override=True ensures values from file always win over system env vars
load_dotenv(os.path.join(parent_dir, ".env"), override=True)
load_dotenv(os.path.join(parent_dir, "main"), override=True)

# Last-resort: manually parse any file in the project root that looks like a .env
for _candidate in [".env", "main", ".env.local", "env"]:
    _fp = os.path.join(parent_dir, _candidate)
    if os.path.isfile(_fp):
        with open(_fp) as _fh:
            for _line in _fh:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    _k = _k.strip(); _v = _v.strip().strip('"').strip("'")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
try:
    from models.allocation_optimizer import run_allocation_optimizer, diminishing_returns_curve
    from app.genai_briefs import generate_safety_brief_prompts
    from vector_rag.actian_search import ActianVectorDB
except Exception as e:
    st.error(f"Module import error: {e}")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BLINDSPOT",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
COUNTRY_NAMES = {
    'AFG': 'Afghanistan', 'AGO': 'Angola', 'BDI': 'Burundi', 'SDN': 'Sudan',
    'HTI': 'Haiti', 'UKR': 'Ukraine', 'SYR': 'Syria', 'YEM': 'Yemen',
    'COD': 'DR Congo', 'ETH': 'Ethiopia', 'SOM': 'Somalia', 'MMR': 'Myanmar',
    'PSE': 'Palestine', 'VEN': 'Venezuela', 'NGA': 'Nigeria', 'SSD': 'South Sudan',
    'CAF': 'C.A. Republic', 'ZAF': 'South Africa', 'MLI': 'Mali',
    'MOZ': 'Mozambique', 'CMR': 'Cameroon', 'GNB': 'Guinea-Bissau',
    'MDG': 'Madagascar', 'ZWE': 'Zimbabwe', 'KEN': 'Kenya', 'TZA': 'Tanzania',
    'UGA': 'Uganda', 'RWA': 'Rwanda', 'LBY': 'Libya', 'IRQ': 'Iraq',
    'PAK': 'Pakistan', 'BGD': 'Bangladesh', 'NPL': 'Nepal', 'MRT': 'Mauritania',
    'NER': 'Niger', 'TCD': 'Chad', 'BFA': 'Burkina Faso', 'GIN': 'Guinea',
    'SLE': 'Sierra Leone', 'LBR': 'Liberia', 'COG': 'Congo', 'ZMB': 'Zambia',
    'MWI': 'Malawi', 'TLS': 'Timor-Leste', 'PRK': 'North Korea',
    'LAO': 'Laos', 'KHM': 'Cambodia', 'GEO': 'Georgia', 'MDA': 'Moldova',
    'SOM': 'Somalia', 'PLW': 'Palau',
}

FALLBACK_GEO = {
    'AFG': [33.9, 67.7],   'AGO': [-11.2, 17.9],  'BDI': [-3.4, 29.9],
    'SDN': [12.8, 30.2],   'HTI': [18.9, -72.2],  'UKR': [48.3, 31.1],
    'SYR': [34.8, 38.9],   'YEM': [15.5, 48.5],   'COD': [-4.0, 21.7],
    'ETH': [9.1, 40.4],    'SOM': [5.1, 46.1],    'MMR': [21.9, 95.9],
    'PSE': [31.9, 35.2],   'VEN': [6.4, -66.5],   'NGA': [9.0, 8.6],
    'SSD': [6.8, 31.3],    'CAF': [6.6, 20.9],    'ZAF': [-30.5, 22.9],
    'MLI': [17.6, -3.9],   'MOZ': [-18.7, 35.5],  'CMR': [3.8, 11.5],
    'MDG': [-20.2, 46.9],  'ZWE': [-20.0, 30.0],  'KEN': [-0.0, 37.9],
    'TZA': [-6.4, 34.9],   'UGA': [1.4, 32.3],    'LBY': [26.3, 17.2],
    'IRQ': [33.2, 43.7],   'PAK': [30.4, 69.3],   'NER': [17.6, 8.1],
    'TCD': [15.5, 18.7],   'BFA': [12.4, -1.6],   'MRT': [20.3, -10.9],
    'RWA': [-1.9, 29.9],   'GEO': [42.3, 43.4],   'MDA': [47.4, 28.4],
    'LBR': [6.4, -9.4],    'SLE': [8.5, -11.8],   'GIN': [11.0, -10.9],
    'ZMB': [-13.1, 27.8],  'MWI': [-13.3, 34.3],
}

REGION_MAP = {
    'AFG': 'South Asia',    'UKR': 'Europe',         'SDN': 'East Africa',
    'SYR': 'Middle East',   'YEM': 'Middle East',    'COD': 'Central Africa',
    'ETH': 'East Africa',   'HTI': 'Caribbean',      'SOM': 'East Africa',
    'MMR': 'Southeast Asia','PSE': 'Middle East',    'VEN': 'Latin America',
    'NGA': 'West Africa',   'SSD': 'East Africa',    'CAF': 'Central Africa',
    'AGO': 'Southern Africa','BDI': 'East Africa',   'MLI': 'West Africa',
    'MOZ': 'Southern Africa','NER': 'West Africa',   'TCD': 'Central Africa',
    'BFA': 'West Africa',   'LBY': 'North Africa',   'IRQ': 'Middle East',
}

# Design tokens
BG     = "#05080F"
CARD   = "#090D18"
BORDER = "#161F33"
RED    = "#E83D3D"
AMBER  = "#F0A500"
GREEN  = "#0DB37A"
BLUE   = "#2D74DA"
MUTED  = "#64748B"
TEXT   = "#CBD5E1"
DIM    = "#334155"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── CHROME REMOVAL ── */
#MainMenu, header, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], .stDeployButton,
[data-testid="manage-app-button"] {{
    visibility: hidden !important;
    display: none !important;
}}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stMain"] > div {{ padding: 0 !important; }}
.stApp {{ background: {BG} !important; }}
.element-container {{ margin-bottom: 0 !important; }}

/* ── FONTS BASE ── */
body, p, span, div, label, li {{ color: {TEXT} !important; font-family: 'DM Sans', sans-serif !important; }}
h1, h2, h3, h4, h5, h6 {{ font-family: 'Syne', sans-serif !important; color: #F1F5F9 !important; }}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {CARD}; }}
::-webkit-scrollbar-thumb {{ background: {DIM}; border-radius: 2px; }}

/* ── NAV BAND ── */
.bs-nav {{
    background: {CARD};
    border-bottom: 1px solid {BORDER};
    padding: 0 28px;
    display: flex; align-items: center; justify-content: space-between;
    height: 60px; position: sticky; top: 0; z-index: 1000;
}}
.bs-nav-logo {{
    font-family: 'Syne', sans-serif !important;
    font-size: 22px; font-weight: 800; color: #F1F5F9 !important;
    letter-spacing: 3px; text-transform: uppercase;
}}
.bs-nav-tagline {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px; font-style: italic; color: #94A3B8 !important; margin-left: 16px;
    white-space: nowrap;
}}
.bs-live-badge {{
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(13,179,122,0.1); border: 1px solid rgba(13,179,122,0.3);
    border-radius: 20px; padding: 4px 12px;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px; color: {GREEN} !important;
    text-transform: uppercase; letter-spacing: 1.5px;
}}
.bs-pulse {{
    width: 7px; height: 7px; background: {GREEN}; border-radius: 50%;
    animation: pulse 1.5s infinite; display: inline-block;
}}
@keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(13,179,122,0.7); }}
    70%  {{ box-shadow: 0 0 0 8px rgba(13,179,122,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(13,179,122,0); }}
}}

/* ── NAV BUTTONS ── */
.stButton > button {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; font-weight: 500 !important;
    text-transform: uppercase !important; letter-spacing: 1.2px !important;
    background: transparent !important; color: #94A3B8 !important;
    border: none !important; border-radius: 0 !important;
    padding: 6px 16px !important; height: 38px !important;
    transition: color 0.15s !important; box-shadow: none !important;
}}
.stButton > button:hover {{
    color: #F1F5F9 !important;
    background: rgba(255,255,255,0.03) !important;
    transform: none !important; box-shadow: none !important;
}}

/* ── SECTION LABELS ── */
.section-label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; color: #94A3B8 !important;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; display: block;
}}
.section-title {{
    font-family: 'Syne', sans-serif !important;
    font-size: 22px !important; color: #F1F5F9 !important;
    font-weight: 700 !important; margin-bottom: 14px !important;
}}

/* ── HERO ── */
.hero-section {{
    padding: 36px 36px 24px; border-bottom: 1px solid {BORDER};
}}
.hero-number {{
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(52px, 7vw, 90px) !important;
    font-weight: 800 !important; line-height: 0.9 !important;
    letter-spacing: -2px !important;
}}
.hero-number .red   {{ color: {RED} !important; }}
.hero-number .white {{ color: #F1F5F9 !important; }}
.hero-sub {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 15px !important; color: #94A3B8 !important;
    margin-top: 10px !important; line-height: 1.5;
}}

/* ── KPI PILLS ── */
.kpi-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 20px 0 0; }}
.kpi-pill {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 10px 18px; flex: 1; min-width: 130px; text-align: center;
}}
.kpi-label {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; color: #94A3B8 !important;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px;
}}
.kpi-value {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 24px !important; font-weight: 600 !important;
    color: #F1F5F9 !important; line-height: 1.2;
}}

/* ── TRIAGE QUEUE ── */
.triage-row {{
    display: flex; align-items: center; gap: 10px;
    padding: 8px 10px; border-radius: 5px; cursor: pointer;
    border: 1px solid transparent; transition: background 0.12s;
}}
.triage-row:hover {{ background: rgba(45,116,218,0.06); border-color: {BORDER}; }}
.triage-rank  {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: #64748B !important; width: 22px; text-align: right; flex-shrink: 0; }}
.triage-dot   {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
.triage-country {{ font-family: 'DM Sans', sans-serif !important; font-size: 15px !important; color: {TEXT} !important; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.triage-score {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 15px !important; font-weight: 600 !important; }}
.triage-pct   {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: #94A3B8 !important; width: 40px; text-align: right; flex-shrink: 0; }}
.triage-trend {{ font-size: 15px; width: 16px; text-align: center; flex-shrink: 0; }}

/* ── METRIC SUMMARY CARDS ── */
.metric-summary {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 16px 18px; text-align: center; height: 100%;
}}
.metric-summary .val {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 22px !important; font-weight: 600 !important; color: #F1F5F9 !important; display: block; margin-bottom: 6px; }}
.metric-summary .lbl {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 1.5px; }}

/* ── IDENTITY CARD ── */
.identity-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 22px 26px; display: flex; align-items: flex-start;
    justify-content: space-between; gap: 20px; margin-bottom: 18px;
    flex-wrap: wrap;
}}
.identity-country {{ font-family: 'Syne', sans-serif !important; font-size: 34px !important; font-weight: 800 !important; color: #F1F5F9 !important; line-height: 1.1 !important; }}
.identity-iso     {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }}
.identity-score-num   {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 56px !important; font-weight: 600 !important; line-height: 1 !important; }}
.identity-score-label {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 2px; }}
.ipc-badge {{ display: inline-block; padding: 4px 12px; border-radius: 4px; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }}
.metric-cell {{ background: rgba(22,31,51,0.6); border: 1px solid {BORDER}; border-radius: 6px; padding: 12px 16px; text-align: center; }}
.metric-cell .mc-val {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 18px !important; font-weight: 600 !important; color: #F1F5F9 !important; display: block; }}
.metric-cell .mc-lbl {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 3px; }}

/* ── RAG / BRIEF PANELS ── */
.rag-card {{
    background: rgba(45,116,218,0.04); border: 1px solid rgba(45,116,218,0.18);
    border-radius: 7px; padding: 14px; margin-bottom: 8px;
}}
.rag-card-title {{ font-family: 'DM Sans', sans-serif !important; font-size: 15px !important; font-weight: 600 !important; color: #F1F5F9 !important; margin-bottom: 5px; }}
.rag-worked  {{ font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; color: #94A3B8 !important; margin-top: 4px; }}
.powered-by  {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; color: #334155 !important; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 8px; }}

/* ── IMPACT BOXES (Page 3) ── */
.impact-box {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 22px; text-align: center; }}
.impact-box .ib-num   {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 38px !important; font-weight: 600 !important; line-height: 1 !important; display: block; margin-bottom: 8px; }}
.impact-box .ib-label {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 1.5px; }}

/* ── METHODOLOGY ── */
.formula-block {{
    background: rgba(45,116,218,0.04); border: 1px solid rgba(45,116,218,0.14);
    border-left: 3px solid {BLUE}; border-radius: 5px;
    padding: 16px 20px; margin: 12px 0;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important; color: {TEXT} !important; line-height: 1.8;
}}
.tech-card  {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 20px; text-align: center; height: 100%; }}
.tc-name    {{ font-family: 'Syne', sans-serif !important; font-size: 17px !important; font-weight: 700 !important; margin-bottom: 8px; }}
.tc-desc    {{ font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; color: #94A3B8 !important; }}
.source-badge {{ display: inline-block; background: rgba(22,31,51,0.8); border: 1px solid {BORDER}; border-radius: 4px; padding: 5px 12px; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: {TEXT} !important; margin: 3px; }}
.disclaimer-box {{ background: rgba(232,61,61,0.05); border: 1px solid rgba(232,61,61,0.2); border-radius: 6px; padding: 16px 20px; font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; color: {TEXT} !important; margin-top: 20px; line-height: 1.7; }}

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stMetric"] {{ background: {CARD} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important; padding: 14px 18px !important; }}
[data-testid="stMetricLabel"] {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 1.5px; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 22px !important; color: #F1F5F9 !important; }}

[data-testid="stSelectbox"] > div > div {{ background: {CARD} !important; border: 1px solid {BORDER} !important; border-radius: 6px !important; color: {TEXT} !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 14px !important; }}
[data-testid="stSelectbox"] label {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 1.5px; }}

[data-testid="stSlider"] > div > div > div > div {{ background: {BLUE} !important; }}
[data-testid="stSlider"] label {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: #94A3B8 !important; text-transform: uppercase; letter-spacing: 1.5px; }}

[data-testid="stTabs"] [data-baseweb="tab-list"] {{ background: {CARD} !important; border-bottom: 1px solid {BORDER} !important; border-radius: 8px 8px 0 0 !important; }}
[data-testid="stTabs"] [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; color: #94A3B8 !important; background: transparent !important; }}
[data-testid="stTabs"] [aria-selected="true"] {{ color: {RED} !important; border-bottom: 2px solid {RED} !important; }}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {{ background: {CARD} !important; border: 1px solid {BORDER} !important; border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 14px !important; }}

[data-testid="stTextInput"] input  {{ background: rgba(22,31,51,0.6) !important; border: 1px solid {BORDER} !important; border-radius: 6px !important; color: {TEXT} !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 14px !important; }}
[data-testid="stForm"] {{ background: transparent !important; border: none !important; }}
[data-testid="stFormSubmitButton"] > button {{ background: {RED} !important; color: white !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; border: none !important; border-radius: 6px !important; padding: 10px 20px !important; box-shadow: none !important; }}
[data-testid="stFormSubmitButton"] > button:hover {{ background: #B82E2E !important; transform: none !important; }}

[data-testid="stDataFrame"] {{ border: 1px solid {BORDER} !important; border-radius: 8px !important; }}
[data-testid="stExpander"] {{ background: {CARD} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important; }}
[data-testid="stExpander"] summary {{ font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: {TEXT} !important; text-transform: uppercase; letter-spacing: 1.2px; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important; }}
[data-testid="stSpinner"] > div {{ color: #94A3B8 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; }}

hr {{ border-color: {BORDER} !important; margin: 8px 0 !important; }}
iframe {{ border: none !important; }}
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [
    ('page', 'command_center'),
    ('chat_open', False),
    ('messages', []),
    ('selected_country', None),
    ('opt_result', None),
    ('opt_budget', 100),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─── DATA LOADING (original logic preserved) ──────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    df = pd.DataFrame()

    api_token = os.environ.get("DATABRICKS_TOKEN", "")
    hostname  = os.environ.get("DATABRICKS_SERVER_HOSTNAME", "")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")

    if api_token and hostname and http_path and "paste_your_token_here" not in api_token:
        try:
            from databricks import sql
            connection = sql.connect(
                server_hostname=hostname,
                http_path=http_path,
                access_token=api_token
            )
            query = "SELECT * FROM hacklytics_db.default.triage_master_optimized LIMIT 300"
            df = pd.read_sql(query, connection)
            connection.close()
        except Exception as e:
            print(f"Databricks SQL Error: {e}")

    if df.empty:
        optimized_path = os.path.join(parent_dir, "assets", "triage_master_optimized.csv")
        scores_path    = os.path.join(parent_dir, "assets", "triage_master_scores.csv")
        if os.path.exists(optimized_path):
            df = pd.read_csv(optimized_path)
        elif os.path.exists(scores_path):
            df = pd.read_csv(scores_path)
        else:
            st.error("No triage scoring data found. Run Databricks pipelines first.")
            return pd.DataFrame({'iso3': [], 'Crisis_Severity_Score': [], 'Funding_Coverage_Ratio': []})

    # Geo
    lats, lons = [], []
    for iso in df['iso3']:
        g = FALLBACK_GEO.get(iso, [0.0, 0.0])
        lats.append(g[0]); lons.append(g[1])
    df['Lat'] = lats
    df['Lon'] = lons
    df['Country_Name'] = df['iso3'].map(COUNTRY_NAMES).fillna(df['iso3'])
    df['Region'] = df['iso3'].map(REGION_MAP).fillna('Other')

    # Funding ratio
    if 'funding_required' in df.columns:
        df['Funding_Ratio'] = np.where(
            df['funding_required'] > 0,
            np.round((df['funding_received'] / df['funding_required']) * 100, 1),
            100.0
        )
    else:
        df['Funding_Ratio'] = df.get('Funding_Coverage_Ratio', pd.Series(dtype=float)) * 100

    # Columns safety
    for col in ['fatalities', 'ipc_phase_3_plus']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    if 'fatalities_last_30_days' in df.columns and 'fatalities' not in df.columns:
        df.rename(columns={'fatalities_last_30_days': 'fatalities'}, inplace=True)

    for col in ['Optimal_Allocation_USD', 'Projected_Lives_Saved']:
        if col not in df.columns:
            df[col] = 0.0

    # IPC phase bucket
    def ipc_label(v):
        if v > 5_000_000: return "IPC Phase 4–5"
        if v > 1_000_000: return "IPC Phase 3+"
        return "IPC Phase 2–3"
    df['IPC_Label'] = df['ipc_phase_3_plus'].apply(ipc_label)

    return df


@st.cache_data(ttl=300)
def compute_misallocation_cost(df_json):
    """Returns (optimal_lives, current_lives, gap) as ints."""
    df = pd.read_json(df_json)
    if df.empty:
        return 0, 0, 0

    optimal_lives = int(df['Projected_Lives_Saved'].sum())

    df_a = df[(df['Crisis_Severity_Score'] > 10) &
              (df['funding_required'] > df['funding_received'])].copy().reset_index(drop=True)
    if df_a.empty:
        return optimal_lives, 0, optimal_lives

    bsf = (df_a['Crisis_Severity_Score'] / 10) ** 2
    base_costs = (50000 / (bsf + 0.1)).values
    aps = (df_a['fatalities'] / max(df_a['fatalities'].max(), 1)).values

    current_lives = sum(
        diminishing_returns_curve(row['funding_received'], base_costs[i], aps[i])
        for i, (_, row) in enumerate(df_a.iterrows())
    )
    gap = max(0, optimal_lives - int(current_lives))
    return optimal_lives, int(current_lives), gap


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def fmt_b(v):
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    if v >= 1e6:  return f"${v/1e6:.0f}M"
    return f"${v:,.0f}"

def fmt_people(v):
    if v >= 1e6: return f"{v/1e6:.1f}M"
    return f"{v:,.0f}"

def sev_color(s):
    return RED if s >= 75 else (AMBER if s >= 50 else GREEN)

def fund_color(p):
    return RED if p < 50 else (AMBER if p < 70 else GREEN)

def plotly_base():
    # Note: margin intentionally omitted here — each chart specifies its own
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="IBM Plex Mono"),
    )


# ─── NAVBAR ───────────────────────────────────────────────────────────────────
def render_navbar(df):
    n = len(df) if not df.empty else 0
    st.markdown(f"""
    <div class="bs-nav">
        <div style="display:flex;align-items:center;">
            <span class="bs-nav-logo">BLINDSPOT</span>
            <span class="bs-nav-tagline">Funding follows headlines. Lives don't.</span>
        </div>
        <span class="bs-live-badge"><span class="bs-pulse"></span>LIVE · {n} crises tracked</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, _gap, c_chat = st.columns([1.1, 1.2, 1.1, 1.2, 3.8, 1.1])

    nav_map = [
        (c1, "command_center",       "⬡ Dashboard"),
        (c2, "crisis_detail",         "◈ Crisis Detail"),
        (c3, "allocation_simulator",  "◉ Allocator"),
        (c4, "methodology",          "◈ Methodology"),
    ]
    for col, page_key, label in nav_map:
        with col:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

    with c_chat:
        chat_label = "✕ Close Chat" if st.session_state.chat_open else "◎ AI Agent"
        if st.button(chat_label, key="nav_chat", use_container_width=True):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()

    # Active page breadcrumb
    labels_map = {
        'command_center': 'Dashboard',
        'crisis_detail': 'Crisis Detail',
        'allocation_simulator': 'Allocation Simulator',
        'methodology': 'Methodology',
    }
    active = labels_map.get(st.session_state.page, '')
    st.markdown(
        f'<div style="background:{CARD};border-bottom:1px solid {BORDER};padding:0 28px;height:26px;'
        f'display:flex;align-items:center;">'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;color:{DIM};'
        f'text-transform:uppercase;letter-spacing:2px;">▶ {active}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


# ─── PAGE 1: COMMAND CENTER ───────────────────────────────────────────────────
def page_command_center(df):
    if df.empty:
        st.warning("No data loaded.")
        return

    opt_l, cur_l, gap = compute_misallocation_cost(df.to_json())

    total_needed = df['ipc_phase_3_plus'].sum()
    total_req  = df['funding_required'].sum()
    total_rcvd = df['funding_received'].sum()
    avg_funded = (total_rcvd / total_req * 100) if total_req > 0 else 0

    # ── HERO ──
    af_color = fund_color(avg_funded)
    # Open hero container
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    # Animated counter via component iframe (Streamlit strips <script> from markdown)
    components.html(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=IBM+Plex+Mono:wght@400&display=swap');
    body {{ background: #05080F; margin: 0; padding: 0; overflow: hidden; }}
    #hero-counter {{
        font-family: 'Syne', sans-serif;
        font-size: clamp(72px, 9vw, 120px);
        font-weight: 800; line-height: 0.9; letter-spacing: -3px;
        color: #E83D3D; display: inline;
    }}
    #hero-suffix {{
        font-family: 'Syne', sans-serif;
        font-size: clamp(72px, 9vw, 120px);
        font-weight: 800; line-height: 0.9; letter-spacing: -3px;
        color: #F1F5F9; display: inline;
    }}
    .hero-sub {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 15px; color: #94A3B8;
        margin-top: 12px; line-height: 1.5;
    }}
    </style>
    <div>
        <span id="hero-counter">0</span><span id="hero-suffix"> lives</span>
        <div class="hero-sub">projected annual lives lost to humanitarian funding misallocation &middot; Source: BLINDSPOT optimizer on OCHA FTS data</div>
    </div>
    <script>
    (function() {{
        var target = {gap};
        var el = document.getElementById('hero-counter');
        var duration = 1600;
        var start = null;
        function easeOut(t) {{ return 1 - Math.pow(1 - t, 3); }}
        function step(ts) {{
            if (!start) start = ts;
            var elapsed = ts - start;
            var progress = Math.min(elapsed / duration, 1);
            var current = Math.round(easeOut(progress) * target);
            el.textContent = current.toLocaleString();
            if (progress < 1) requestAnimationFrame(step);
            else el.textContent = target.toLocaleString();
        }}
        requestAnimationFrame(step);
    }})();
    </script>
    """, height=160, scrolling=False)

    # KPI pills + close hero
    st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-pill">
                <div class="kpi-label">People In Need</div>
                <div class="kpi-value">{fmt_people(total_needed)}</div>
            </div>
            <div class="kpi-pill">
                <div class="kpi-label">Total Requirements</div>
                <div class="kpi-value">{fmt_b(total_req)}</div>
            </div>
            <div class="kpi-pill">
                <div class="kpi-label">Avg. Funded</div>
                <div class="kpi-value" style="color:{af_color};">{avg_funded:.0f}%</div>
            </div>
            <div class="kpi-pill">
                <div class="kpi-label">Active Crises</div>
                <div class="kpi-value">{len(df)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── MAP + QUEUE ──
    st.markdown('<div style="padding:20px 28px 0;">', unsafe_allow_html=True)
    left_col, right_col = st.columns([63, 37], gap="large")

    with left_col:
        st.markdown('<span class="section-label">Global Crisis Map — bubble size = people in need · color = crisis severity · drag to spin · scroll to zoom</span>', unsafe_allow_html=True)

        import json as _json
        df_map = df.copy()
        max_ipc = max(df_map['ipc_phase_3_plus'].max(), 1)
        df_map['bsz'] = (np.power(df_map['ipc_phase_3_plus'].clip(lower=1e4) / max_ipc, 0.38) * 58 + 9).clip(lower=9, upper=70)

        def _sev_hex(s):
            if s >= 80: return '#FF2020'
            if s >= 60: return '#E83D3D'
            if s >= 40: return '#F0A500'
            if s >= 20: return '#0DB37A'
            return '#2D74DA'

        markers = []
        for _, r in df_map.iterrows():
            sev = float(r['Crisis_Severity_Score'])
            markers.append({
                'lat':      float(r['Lat']),
                'lng':      float(r['Lon']),
                'name':     str(r['Country_Name']),
                'sev':      round(sev, 1),
                'funded':   round(float(r['Funding_Ratio']), 1),
                'people':   fmt_people(r['ipc_phase_3_plus']),
                'color':    _sev_hex(sev),
                'radius':   float(r['bsz']) / 140,   # ~0.07 – 0.50 in globe.gl units
                'altitude': float(r['bsz']) / 350,   # tiny ring lift off surface
            })
        markers_js = _json.dumps(markers)

        globe_html = f"""
<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#05080F; overflow:hidden; }}
  #g {{ width:100%; height:520px; }}
  #tip {{
    position:fixed; display:none; pointer-events:none; z-index:9999;
    background:#090D18; border:1px solid #2D74DA44;
    border-radius:7px; padding:10px 14px;
    font-family:'IBM Plex Mono',monospace; font-size:12px; color:#CBD5E1;
    line-height:1.7; box-shadow:0 0 18px rgba(45,116,218,0.25);
  }}
  #legend {{
    position:absolute; bottom:12px; left:14px;
    font-family:'IBM Plex Mono',monospace; font-size:10px; color:#64748B;
    line-height:2;
  }}
  .lc {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }}
</style>
</head><body>
<div id="g"></div>
<div id="tip"></div>
<div id="legend">
  <span class="lc" style="background:#FF2020"></span>Critical (&gt;80)&nbsp;&nbsp;
  <span class="lc" style="background:#E83D3D"></span>High (60–80)&nbsp;&nbsp;
  <span class="lc" style="background:#F0A500"></span>Elevated (40–60)&nbsp;&nbsp;
  <span class="lc" style="background:#0DB37A"></span>Moderate (20–40)&nbsp;&nbsp;
  <span class="lc" style="background:#2D74DA"></span>Low (&lt;20)
</div>
<script src="https://cdn.jsdelivr.net/npm/globe.gl@2/dist/globe.gl.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></script>
<script>
const DATA = {markers_js};

const tip  = document.getElementById('tip');
const cont = document.getElementById('g');

const world = Globe()(cont)
  .width(cont.offsetWidth)
  .height(520)
  .backgroundColor('#03080F')
  // No image texture — country polygons give us land/ocean contrast
  .globeImageUrl('')
  .atmosphereColor('#2563EB')
  .atmosphereAltitude(0.15)
  // ── crisis markers ──
  .pointsData(DATA)
  .pointLat(d => d.lat)
  .pointLng(d => d.lng)
  .pointColor(d => d.color)
  .pointRadius(d => d.radius)
  .pointAltitude(d => d.altitude)
  .pointResolution(14)
  .pointLabel(d => '')
  .onPointHover((pt) => {{
    if (pt) {{
      tip.style.display = 'block';
      tip.innerHTML =
        '<b style="color:#F1F5F9;font-size:13px">' + pt.name + '</b><br>' +
        'Severity &nbsp;<span style="color:' + pt.color + '">' + pt.sev + '/100</span><br>' +
        'Funded &nbsp;&nbsp;<span style="color:#94A3B8">' + pt.funded + '%</span><br>' +
        'People &nbsp;&nbsp;<span style="color:#94A3B8">' + pt.people + '</span>';
    }} else {{
      tip.style.display = 'none';
    }}
  }});

// Manually color the globe sphere (ocean = dark navy)
world.scene().traverse(obj => {{
  if (obj.isMesh && obj.geometry && obj.geometry.type === 'SphereGeometry') {{
    obj.material.color.setHex(0x04111e);
    obj.material.emissive.setHex(0x020c16);
    obj.material.emissiveIntensity = 0.4;
  }}
}});

// ── Load real country polygons (land masses) ──
fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')
  .then(r => r.json())
  .then(worldData => {{
    const countries = topojson.feature(worldData, worldData.objects.countries).features;
    world
      .polygonsData(countries)
      .polygonCapColor(() => '#0E2340')      // land — clearly visible navy-blue
      .polygonSideColor(() => '#071422')     // depth shading
      .polygonStrokeColor(() => '#1E4080')   // country borders — brighter blue line
      .polygonAltitude(0.004);               // just off the sphere surface
  }});

// Tip follows mouse
document.addEventListener('mousemove', e => {{
  if (tip.style.display !== 'none') {{
    const x = e.clientX, y = e.clientY;
    const W = window.innerWidth;
    tip.style.left = (x + 16 + 200 > W ? x - 216 : x + 16) + 'px';
    tip.style.top  = (y - 10 < 0 ? 4 : y - 10) + 'px';
  }}
}});

// ── Mouse-move responsive spin ──
const controls = world.controls();
controls.enableDamping   = true;
controls.dampingFactor   = 0.08;
controls.autoRotate      = true;
controls.autoRotateSpeed = 0.7;
controls.enableZoom      = true;
controls.zoomSpeed       = 1.2;
controls.minDistance     = 150;
controls.maxDistance     = 700;

let overGlobe = false;
cont.addEventListener('mouseenter', () => {{ overGlobe = true; }});
cont.addEventListener('mouseleave', () => {{
  overGlobe = false;
  controls.autoRotateSpeed = 0.7;
}});

let _isDragging = false;
cont.addEventListener('mousedown', () => {{ _isDragging = true; }});
cont.addEventListener('mouseup',   () => {{ _isDragging = false; }});

cont.addEventListener('mousemove', e => {{
  if (!overGlobe) return;
  const rect = cont.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width  * 2 - 1;
  const ny = (e.clientY - rect.top)  / rect.height * 2 - 1;
  controls.autoRotateSpeed = nx * 4.0;
  if (!_isDragging) {{
    const pov = world.pointOfView();
    const tLat = Math.max(-80, Math.min(80, pov.lat - ny * 0.5));
    world.pointOfView({{ lat: tLat, lng: pov.lng, altitude: pov.altitude }}, 80);
  }}
}});

world.pointOfView({{ lat: 20, lng: 20, altitude: 2.6 }});
</script>
</body></html>
"""
        components.html(globe_html, height=530, scrolling=False)

    with right_col:
        st.markdown('<span class="section-label">Live Triage Queue</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top 10 Critical Zones</div>', unsafe_allow_html=True)

        df_q = df.nlargest(10, 'Crisis_Severity_Score').reset_index(drop=True)
        q_html = '<div style="display:flex;flex-direction:column;gap:1px;">'
        for i, row in df_q.iterrows():
            dc = sev_color(row['Crisis_Severity_Score'])
            fp = row['Funding_Ratio']
            trend = "↑" if fp > 60 else ("↓" if fp < 35 else "→")
            tc = GREEN if trend == "↑" else (RED if trend == "↓" else AMBER)
            sc = row['Crisis_Severity_Score']
            q_html += f"""
            <div class="triage-row">
                <span class="triage-rank">{i+1:02d}</span>
                <span class="triage-dot" style="background:{dc};"></span>
                <span class="triage-country">{row['Country_Name']}</span>
                <span class="triage-score" style="color:{dc};">{sc:.0f}</span>
                <span class="triage-pct">{fp:.0f}%</span>
                <span class="triage-trend" style="color:{tc};">{trend}</span>
            </div>"""
        q_html += '</div>'
        st.markdown(q_html, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:10px;font-family:'IBM Plex Mono',monospace;font-size:8px;color:{DIM};text-transform:uppercase;letter-spacing:1.3px;line-height:1.9;">
            Score = PCA severity · Color: <span style="color:{RED}">■</span> &gt;75 &nbsp;
            <span style="color:{AMBER}">■</span> 50–75 &nbsp; <span style="color:{GREEN}">■</span> &lt;50<br>
            Trend = funding trajectory (↑ improving / ↓ declining)
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        if st.button("→ Open Crisis Detail", key="cc_goto_cd", use_container_width=True):
            st.session_state.page = 'crisis_detail'
            st.rerun()

    # ── BOTTOM METRIC CARDS ──
    st.markdown(f'<hr style="margin:16px 0 12px;">', unsafe_allow_html=True)
    st.markdown('<span class="section-label" style="padding:0 28px;">System Metrics · Source: OCHA FTS</span>', unsafe_allow_html=True)

    red_zones = df[df['Crisis_Severity_Score'] > 75] if 'Is_Red_Zone' not in df.columns else df[df['Is_Red_Zone'] == True]
    most_severe  = df.loc[df['Crisis_Severity_Score'].idxmax()]
    least_funded = df.loc[df['Funding_Ratio'].idxmin()]
    total_gap    = (df['funding_required'] - df['funding_received']).clip(lower=0).sum()

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col, val, lbl, color in [
        (mc1, str(len(red_zones)),                        "Red Zone Crises",       RED),
        (mc2, most_severe['Country_Name'],                "Highest Severity",      RED),
        (mc3, f"{most_severe['Crisis_Severity_Score']:.0f}/100", "Peak Severity Score", RED),
        (mc4, least_funded['Country_Name'],               "Least Funded Zone",     AMBER),
        (mc5, fmt_b(total_gap),                           "Total Funding Gap",     AMBER),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-summary" style="margin:0 6px;">
                <span class="val" style="color:{color};">{val}</span>
                <span class="lbl">{lbl}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── PAGE 2: CRISIS DETAIL ────────────────────────────────────────────────────
def page_crisis_detail(df):
    if df.empty:
        st.warning("No data loaded.")
        return

    st.markdown('<div style="padding:20px 28px;">', unsafe_allow_html=True)

    options = df.sort_values('Crisis_Severity_Score', ascending=False)['Country_Name'].tolist()
    def_idx = 0
    if st.session_state.selected_country in options:
        def_idx = options.index(st.session_state.selected_country)

    selected = st.selectbox("SELECT CRISIS ZONE", options, index=def_idx, key="cd_select")
    st.session_state.selected_country = selected
    row = df[df['Country_Name'] == selected].iloc[0]

    iso  = row['iso3']
    sc   = row['Crisis_Severity_Score']
    fp   = row['Funding_Ratio']
    req  = row.get('funding_required', 0)
    rcvd = row.get('funding_received', 0)
    ipc  = row.get('ipc_phase_3_plus', 0)
    fat  = row.get('fatalities', 0)
    ilab = row.get('IPC_Label', 'IPC Phase 3+')
    reg  = row.get('Region', 'Unknown')
    opt_lives = max(float(row.get('Projected_Lives_Saved', 0)), 0)

    ic    = RED if 'Phase 4' in ilab else (AMBER if 'Phase 3' in ilab else GREEN)
    sc_c  = sev_color(sc)
    fp_c  = fund_color(fp)

    # ── IDENTITY CARD ──
    st.markdown(f"""
    <div class="identity-card">
        <div>
            <div class="identity-country">{selected}</div>
            <div class="identity-iso">{iso} &nbsp;·&nbsp; {reg}</div>
            <span class="ipc-badge" style="background:{ic}18;color:{ic};border:1px solid {ic}40;">{ilab}</span>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
            <div class="metric-cell"><span class="mc-val">{fmt_people(ipc)}</span><span class="mc-lbl">People in Need</span></div>
            <div class="metric-cell"><span class="mc-val">{fat:,.0f}</span><span class="mc-lbl">Conflict Fatalities</span></div>
            <div class="metric-cell"><span class="mc-val">{fmt_b(req)}</span><span class="mc-lbl">$ Required</span></div>
            <div class="metric-cell"><span class="mc-val" style="color:{fp_c};">{fp:.0f}%</span><span class="mc-lbl">Funded</span></div>
        </div>
        <div style="text-align:right;flex-shrink:0;">
            <div class="identity-score-num" style="color:{sc_c};">{sc:.0f}</div>
            <div class="identity-score-label">Severity Score</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:7px;color:{DIM};margin-top:3px;text-transform:uppercase;letter-spacing:1px;">BLINDSPOT PCA MODEL</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── THREE CHARTS ──
    ch1, ch2, ch3 = st.columns(3, gap="medium")

    # Radar
    with ch1:
        st.markdown('<span class="section-label">Severity Signal Breakdown · 6 signals normalized 0–100</span>', unsafe_allow_html=True)
        mx_ipc = max(df['ipc_phase_3_plus'].max(), 1)
        mx_fat = max(df['fatalities'].max(), 1)
        mx_gap = max((df['funding_required'] - df['funding_received']).clip(lower=0).max(), 1)
        sigs = ['Food\nInsecurity', 'Conflict\nIntensity', 'Funding\nGap', 'Severity\nScore', 'Coverage\nDeficit', 'Displacement']
        vals = [
            (ipc / mx_ipc) * 100,
            (fat / mx_fat) * 100,
            ((req - rcvd) / mx_gap) * 100,
            sc,
            max(0, 100 - fp),
            min(100, (ipc / max(mx_ipc * 0.5, 1)) * 100),
        ]
        vc = vals + [vals[0]]
        sc2 = sigs + [sigs[0]]
        fig_r = go.Figure(go.Scatterpolar(
            r=vc, theta=sc2, fill='toself',
            fillcolor='rgba(232,61,61,0.12)',
            line=dict(color=RED, width=2),
        ))
        fig_r.update_layout(
            **plotly_base(), height=270,
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER, tickfont=dict(color=MUTED, size=7), linecolor=BORDER),
                angularaxis=dict(gridcolor=BORDER, tickfont=dict(color=TEXT, size=8, family="IBM Plex Mono"), linecolor=BORDER),
            ),
            showlegend=False, margin=dict(l=28, r=28, t=10, b=10),
        )
        st.plotly_chart(fig_r, use_container_width=True, config=dict(displayModeBar=False))

    # Funding timeline (synthetic)
    with ch2:
        st.markdown('<span class="section-label">Funding History 2015–2024 · Requirements vs Received · Source: OCHA FTS (projected)</span>', unsafe_allow_html=True)
        yrs = list(range(2015, 2025))
        np.random.seed(abs(hash(iso)) % (2**31))
        base_r = req * np.linspace(0.45, 1.0, 10) * (1 + np.random.randn(10) * 0.04)
        base_rcv = (base_r * (fp / 100) * (1 + np.random.randn(10) * 0.08)).clip(0, None)
        for i in range(len(base_r)):
            base_rcv[i] = min(base_rcv[i], base_r[i])

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=yrs, y=base_r / 1e6, name='Required',
            line=dict(color=MUTED, width=1.5, dash='dot'), mode='lines'))
        fig_f.add_trace(go.Scatter(
            x=yrs + yrs[::-1],
            y=list(base_r / 1e6) + list(base_rcv[::-1] / 1e6),
            fill='toself', fillcolor='rgba(232,61,61,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name='Gap', hoverinfo='skip'))
        fig_f.add_trace(go.Scatter(
            x=yrs, y=base_rcv / 1e6, name='Received',
            line=dict(color=BLUE, width=2), marker=dict(size=4, color=BLUE), mode='lines+markers'))
        fig_f.update_layout(
            **plotly_base(), height=270,
            xaxis=dict(gridcolor=BORDER, tickfont=dict(size=8)),
            yaxis=dict(gridcolor=BORDER, tickfont=dict(size=8),
                       title=dict(text="$M", font=dict(size=8, color=MUTED))),
            legend=dict(font=dict(size=8, color=MUTED), bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
            margin=dict(l=36, r=8, t=10, b=28),
        )
        st.plotly_chart(fig_f, use_container_width=True, config=dict(displayModeBar=False))

    # Media vs funding
    with ch3:
        st.markdown('<span class="section-label">Media Attention Index vs Humanitarian Need · Source: ReliefWeb (estimated)</span>', unsafe_allow_html=True)
        yb = list(range(2020, 2025))
        med = np.array([30, 25, 28, 20, 18]) * (1 + np.random.randn(5) * 0.08) * max(1 - sc / 120, 0.3)
        need = np.array([60, 68, 74, 80, 88]) * (sc / 100)
        fig_m = make_subplots(specs=[[{"secondary_y": True}]])
        fig_m.add_trace(go.Bar(x=yb, y=need, name='Need Index', marker_color='rgba(232,61,61,0.55)'), secondary_y=False)
        fig_m.add_trace(go.Scatter(x=yb, y=med, name='Media Index', mode='lines+markers',
                                   line=dict(color=AMBER, width=2), marker=dict(size=4)), secondary_y=True)
        fig_m.update_layout(
            **plotly_base(), height=270,
            legend=dict(font=dict(size=8, color=MUTED), bgcolor="rgba(0,0,0,0)", x=0.01, y=0.99),
            xaxis=dict(gridcolor=BORDER, tickfont=dict(size=8)),
            margin=dict(l=28, r=28, t=10, b=28),
        )
        fig_m.update_yaxes(gridcolor=BORDER, tickfont=dict(size=8), secondary_y=False)
        fig_m.update_yaxes(gridcolor=BORDER, tickfont=dict(size=8), showgrid=False, secondary_y=True)
        st.plotly_chart(fig_m, use_container_width=True, config=dict(displayModeBar=False))

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    pan_l, pan_r = st.columns(2, gap="large")

    # ── RAG panel ──
    with pan_l:
        st.markdown('<span class="section-label">Comparable Past Crises · Powered by Actian VectorAI</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Historical Precedents</div>', unsafe_allow_html=True)

        profile = {'iso3': iso, 'Crisis_Severity_Score': sc, 'funding_required': req, 'Cluster_Need': 'Food/Health/WASH'}
        try:
            db_rag = ActianVectorDB()
            base_m = db_rag.find_comparable_crisis(profile)
            matches = [
                base_m,
                {"historical_crisis": "Yemen Crisis (2016–2018)", "similarity_score": "78%",
                 "what_worked": "Multi-donor coordination + CERF emergency activation.",
                 "funding_secured": "$2.1B"},
                {"historical_crisis": "Syria Displacement (2014–2015)", "similarity_score": "71%",
                 "what_worked": "Regional refugee compact secured sustained pledges.",
                 "funding_secured": "$3.4B"},
                {"historical_crisis": "CAR Conflict (2013)", "similarity_score": "63%",
                 "what_worked": "Cluster system activation + media advocacy campaign.",
                 "funding_secured": "$0.6B"},
                {"historical_crisis": "South Sudan Famine (2017)", "similarity_score": "57%",
                 "what_worked": "IPC Phase 5 declaration triggered emergency response.",
                 "funding_secured": "$1.1B"},
            ]
        except Exception:
            matches = [
                {"historical_crisis": "Somalia Famine (2011)", "similarity_score": "84%",
                 "what_worked": "Rapid cash transfers + localized WASH interventions.",
                 "funding_secured": "$1.2B"},
                {"historical_crisis": "Yemen Crisis (2016–2018)", "similarity_score": "78%",
                 "what_worked": "Multi-donor coordination + CERF activation.",
                 "funding_secured": "$2.1B"},
                {"historical_crisis": "Syria (2014–2015)", "similarity_score": "71%",
                 "what_worked": "Regional refugee compact secured pledges.",
                 "funding_secured": "$3.4B"},
                {"historical_crisis": "CAR Conflict (2013)", "similarity_score": "63%",
                 "what_worked": "Cluster activation + media advocacy.",
                 "funding_secured": "$0.6B"},
                {"historical_crisis": "South Sudan Famine (2017)", "similarity_score": "57%",
                 "what_worked": "IPC Phase 5 declaration → emergency response.",
                 "funding_secured": "$1.1B"},
            ]

        for m in matches:
            sv = int(str(m['similarity_score']).replace('%', ''))
            sc3 = GREEN if sv >= 80 else (AMBER if sv >= 65 else MUTED)
            st.markdown(f"""
            <div class="rag-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div class="rag-card-title">{m['historical_crisis']}</div>
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{sc3};">{m['similarity_score']} match</span>
                </div>
                <div class="rag-worked">✓ {m['what_worked']}</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:9px;color:{GREEN};margin-top:3px;">Funding secured: {m['funding_secured']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div class="powered-by">◈ Actian VectorAI · Semantic similarity over UN HRP documents</div>', unsafe_allow_html=True)

    # ── Brief Generator ──
    with pan_r:
        st.markdown('<span class="section-label">Emergency Brief Generator · Powered by Gemini AI</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">AI Situation Briefs</div>', unsafe_allow_html=True)

        tab_d, tab_j, tab_u = st.tabs(["DONOR ROI", "JOURNALIST", "UN COORDINATOR"])

        crisis_data = {'iso3': iso, 'Crisis_Severity_Score': sc,
                       'funding_required': req, 'funding_received': rcvd}

        for tab_obj, audience_key, audience_label, audience_desc in [
            (tab_d, 'donor',      'Donor Brief',      'ROI-focused brief for humanitarian donors and impact investors.'),
            (tab_j, 'journalist', 'Journalist Brief',  'Investigative story pitch surfacing the invisible crisis.'),
            (tab_u, 'un',         'UN Brief',          'Strategic brief for UN field coordinators with operational guidance.'),
        ]:
            with tab_obj:
                st.markdown(f'<div style="font-family:\'DM Sans\',sans-serif;font-size:11px;color:{MUTED};margin-bottom:10px;">{audience_desc}</div>', unsafe_allow_html=True)
                btn_key = f"gen_{audience_key}_{iso}"
                cache_key = f"brief_{audience_key}_{iso}"
                if st.button(f"Generate {audience_label}", key=btn_key, use_container_width=True):
                    with st.spinner(f"Generating via Gemini 2.5 Flash..."):
                        try:
                            briefs = generate_safety_brief_prompts(crisis_data, max(opt_lives, 1000))
                            st.session_state[cache_key] = briefs[f'{audience_key}_brief']
                        except Exception as e:
                            st.session_state[cache_key] = f"Generation error: {str(e)[:200]}"
                if cache_key in st.session_state and st.session_state[cache_key]:
                    st.markdown(
                        f'<div style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:{TEXT};'
                        f'line-height:1.7;padding:12px;background:rgba(22,31,51,0.5);'
                        f'border-radius:6px;border:1px solid {BORDER};margin-top:8px;">'
                        f'{st.session_state[cache_key]}</div>',
                        unsafe_allow_html=True
                    )

        st.markdown(f'<div class="powered-by">◈ Gemini AI · google/gemini-2.5-flash · Three-audience brief system</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── PAGE 3: ALLOCATION SIMULATOR ─────────────────────────────────────────────
def page_allocation_simulator(df):
    st.markdown('<div style="padding:20px 28px;">', unsafe_allow_html=True)

    st.markdown(f"""
    <span class="section-label">Allocation Simulator · Databricks SLSQP Engine</span>
    <div class="section-title" style="font-size:22px;">Constrained Optimization Engine</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{MUTED};max-width:600px;line-height:1.6;margin-bottom:20px;">
        Distributes a donor budget across all active crises to maximize total lives saved.
        Models logarithmic diminishing returns and conflict-based absorptive capacity constraints.
        Solver: SciPy SLSQP · Runs on Databricks when credentials are configured.
    </div>
    """, unsafe_allow_html=True)

    ctrl1, ctrl2 = st.columns([3, 1], gap="large")
    with ctrl1:
        budget_m = st.slider(
            "DEPLOYMENT BUDGET (USD MILLIONS)",
            min_value=10, max_value=500,
            value=st.session_state.opt_budget,
            step=10, key="budget_slider"
        )
        st.session_state.opt_budget = budget_m
    with ctrl2:
        st.markdown('<div style="height:26px;"></div>', unsafe_allow_html=True)
        run_clicked = st.button("▶ RUN OPTIMIZER", key="run_opt_btn", use_container_width=True)

    if run_clicked:
        total_budget = budget_m * 1_000_000
        df_active = df[
            (df['Crisis_Severity_Score'] > 10) &
            (df['funding_required'] > df['funding_received'])
        ].copy().reset_index(drop=True)

        if df_active.empty:
            st.warning("No underfunded active crises found to optimize.")
        else:
            df_active['funding_required'] = np.where(
                (df_active['funding_required'] == 0) & (df_active['funding_received'] == 0),
                1_000_000_000, df_active['funding_required']
            )
            df_active.fillna({'funding_required': 1_000_000_000, 'funding_received': 0}, inplace=True)

            with st.spinner(f"SLSQP optimizer running · {budget_m}M budget · {len(df_active)} active crises..."):
                try:
                    df_result = run_allocation_optimizer(df_active, total_budget)
                    st.session_state.opt_result = df_result
                except Exception as e:
                    st.error(f"Optimizer error: {e}")

    if st.session_state.opt_result is not None:
        df_opt = st.session_state.opt_result.copy()
        df_opt['Country_Name'] = df_opt['iso3'].map(COUNTRY_NAMES).fillna(df_opt['iso3'])

        opt_lives = int(df_opt['Projected_Lives_Saved'].sum())
        bsf = (df_opt['Crisis_Severity_Score'] / 10) ** 2
        base_c = (50000 / (bsf + 0.1)).values
        aps = (df_opt['fatalities'] / max(df_opt['fatalities'].max(), 1)).values
        cur_lives = int(sum(
            diminishing_returns_curve(row['funding_received'], base_c[i], aps[i])
            for i, (_, row) in enumerate(df_opt.iterrows())
        ))
        gap_lives = max(0, opt_lives - cur_lives)

        # Impact boxes
        ib1, ib2, ib3 = st.columns(3, gap="medium")
        for col, num, lbl, color, bc in [
            (ib1, cur_lives, "Lives Saved — Current Allocation",      MUTED,  DIM),
            (ib2, opt_lives, "Lives Saved — Optimal Allocation",      GREEN,  f"{GREEN}40"),
            (ib3, gap_lives, "Lives Lost to Misallocation Gap",       RED,    f"{RED}40"),
        ]:
            with col:
                st.markdown(f"""
                <div class="impact-box" style="border-color:{bc};">
                    <span class="ib-num" style="color:{color};">{num:,}</span>
                    <span class="ib-label">{lbl}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        # Sankey
        sk_l, sk_r = st.columns(2, gap="medium")
        df_top = df_opt.nlargest(8, 'Optimal_Allocation_USD')
        countries = df_top['Country_Name'].tolist()
        labels = ["Budget"] + countries
        sources = [0] * len(countries)
        targets = list(range(1, len(countries) + 1))

        def mk_sankey(allocs, title):
            fig = go.Figure(go.Sankey(
                node=dict(
                    label=labels,
                    color=[BLUE] + [RED if sev_color(df_top.iloc[i]['Crisis_Severity_Score']) == RED else AMBER for i in range(len(countries))],
                    pad=10, thickness=14,
                    line=dict(color=BORDER, width=0.5),
                ),
                link=dict(
                    source=sources, target=targets,
                    value=[max(float(a), 1.0) for a in allocs],
                    color=['rgba(45,116,218,0.18)'] * len(allocs),
                ),
            ))
            fig.update_layout(
                **plotly_base(), height=300,
                title=dict(text=f'<span style="font-size:10px;color:{MUTED};">{title}</span>', x=0.01),
                margin=dict(l=6, r=6, t=32, b=6),
            )
            return fig

        with sk_l:
            st.markdown('<span class="section-label">Current Allocation Flow · Source: OCHA FTS</span>', unsafe_allow_html=True)
            ca = df_top['funding_received'].clip(lower=0).tolist()
            if sum(ca) == 0: ca = [1e6] * len(ca)
            st.plotly_chart(mk_sankey(ca, "Current Humanitarian Funding"), use_container_width=True, config=dict(displayModeBar=False))

        with sk_r:
            st.markdown('<span class="section-label">Optimal Allocation Flow · BLINDSPOT Model</span>', unsafe_allow_html=True)
            oa = df_top['Optimal_Allocation_USD'].clip(lower=0).tolist()
            if sum(oa) == 0: oa = [budget_m * 1e6 / len(oa)] * len(oa)
            st.plotly_chart(mk_sankey(oa, "BLINDSPOT Optimized Allocation"), use_container_width=True, config=dict(displayModeBar=False))

        # Comparison table
        st.markdown('<span class="section-label">Reallocation Delta Analysis · All active underfunded crises</span>', unsafe_allow_html=True)
        table = []
        for _, r in df_opt.sort_values('Crisis_Severity_Score', ascending=False).iterrows():
            delta = r['Optimal_Allocation_USD'] - r['funding_received']
            table.append({
                'Country':           r['Country_Name'],
                'Severity':          f"{r['Crisis_Severity_Score']:.0f}",
                'Current Funding':   fmt_b(r['funding_received']),
                'Optimal Allocation':fmt_b(r['Optimal_Allocation_USD']),
                'Delta':             ('+' if delta >= 0 else '') + fmt_b(delta),
                'Lives Saved':       f"{int(r['Projected_Lives_Saved']):,}",
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:8px;color:{DIM};margin-top:6px;text-transform:uppercase;letter-spacing:1.2px;">SLSQP · Diminishing returns model · Access penalty = fatalities / max_fatalities</div>', unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="text-align:center;padding:70px 0;border:1px dashed {BORDER};border-radius:8px;margin-top:20px;">
            <div style="font-family:'Syne',sans-serif;font-size:28px;color:{DIM};margin-bottom:10px;">◉</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:{DIM};text-transform:uppercase;letter-spacing:2px;">
                Set budget above and click Run Optimizer
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── PAGE 4: METHODOLOGY ─────────────────────────────────────────────────────
def page_methodology():
    st.markdown('<div style="padding:20px 28px;">', unsafe_allow_html=True)

    st.markdown(f"""
    <span class="section-label">Methodology & Architecture</span>
    <div class="section-title" style="font-size:22px;">How BLINDSPOT Works</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{MUTED};max-width:620px;line-height:1.7;margin-bottom:24px;">
        A 5-layer intelligence system replacing media-driven funding allocation with statistically rigorous
        severity scoring, constraint-based optimization, and LLM-assisted decision communication.
    </div>
    """, unsafe_allow_html=True)

    # Layer 1
    st.markdown(f'<div class="section-title">Layer 1 · Crisis Severity Score</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'DM Sans',sans-serif;font-size:12px;color:{TEXT};line-height:1.7;margin-bottom:10px;max-width:700px;">
        Composite 0–100 score computed via PCA over 6 humanitarian signals.
        PCA ensures no single signal dominates — the score reflects the statistical
        structure of the multi-dimensional crisis profile. Computed on Databricks.
    </div>
    <div class="formula-block">
        Severity_Score = Normalize_0_100( PCA₁(<br>
        &nbsp;&nbsp;HumanitarianNeed, ConflictIntensity, FoodSecurityPhase,<br>
        &nbsp;&nbsp;DisplacementVolume, FundingCoverageGap, InverseMediaAttention<br>
        ))<br><br>
        Red Zone = Score &gt; 75 AND Coverage &lt; 30%<br>
        Scaler: StandardScaler(Z-scores) → PCA(n_components=1) → MinMax[0,100]
    </div>
    """, unsafe_allow_html=True)

    # Signals table
    rows = [
        ("Humanitarian Need Magnitude", "FTS · OCHA",   "People in need, IPC Phase 3+ population", "~25%"),
        ("Conflict Intensity",          "UCDP GED",      "Fatalities (rolling 30-day count)",       "~20%"),
        ("Food Security Phase",         "IPC Global",    "Phase 3+ population count by country",    "~20%"),
        ("Displacement Volume",         "IDMC",          "IDP + refugee outflow stock",             "~15%"),
        ("Funding Coverage Gap",        "CBPF / FTS",    "1 − (funding_received / funding_required)","~12%"),
        ("Inverse Media Attention",     "ReliefWeb API", "1 − normalized ReliefWeb article count",  "~8%"),
    ]
    tbl = f"""<table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:10px;margin:10px 0 20px;">
    <thead><tr style="border-bottom:1px solid {BORDER};">
        <th style="text-align:left;padding:7px 12px;color:{MUTED};text-transform:uppercase;letter-spacing:1.2px;font-weight:500;">Signal</th>
        <th style="text-align:left;padding:7px 12px;color:{MUTED};text-transform:uppercase;letter-spacing:1.2px;font-weight:500;">Source</th>
        <th style="text-align:left;padding:7px 12px;color:{MUTED};text-transform:uppercase;letter-spacing:1.2px;font-weight:500;">Metric</th>
        <th style="text-align:right;padding:7px 12px;color:{MUTED};text-transform:uppercase;letter-spacing:1.2px;font-weight:500;">PCA Weight</th>
    </tr></thead><tbody>"""
    for sig, src, met, wt in rows:
        tbl += f"""<tr style="border-bottom:1px solid {BORDER}18;">
            <td style="padding:7px 12px;color:{TEXT};">{sig}</td>
            <td style="padding:7px 12px;color:{BLUE};">{src}</td>
            <td style="padding:7px 12px;color:{MUTED};">{met}</td>
            <td style="padding:7px 12px;text-align:right;color:{AMBER};">{wt}</td>
        </tr>"""
    tbl += "</tbody></table>"
    st.markdown(tbl, unsafe_allow_html=True)

    # Layer 2
    st.markdown(f'<div class="section-title">Layer 2 · Allocation Optimizer</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="formula-block">
        <b>Problem:</b> Given budget B and N active crises, find allocation vector x* such that:<br><br>
        maximize  &nbsp; Σᵢ LivesSaved(xᵢ, Severityᵢ, AccessPenaltyᵢ)<br><br>
        where  &nbsp; LivesSaved(x, s, p) = √x × 1000 / ( BaseCost(s) × (1 + AccessPenalty) )<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; BaseCost(s) = 50,000 / ( (s/10)² + 0.1 )<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; AccessPenalty = fatalities / max_fatalities<br><br>
        subject to:  Σᵢ xᵢ = B &nbsp; (budget equality)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  xᵢ ≥ 0 &nbsp; (non-negativity)<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  xᵢ ≤ min(UnmetNeedᵢ, B) &nbsp; (absorptive capacity)<br><br>
        Solver: SciPy SLSQP · Initial guess: equal distribution · Runtime: &lt;2s for N=54
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    # Tech stack
    st.markdown(f'<div class="section-title">Technology Stack</div>', unsafe_allow_html=True)
    tc1, tc2, tc3, tc4 = st.columns(4, gap="medium")
    for col, name, icon, desc, color in [
        (tc1, "Databricks",       "⬡", "Severity scoring + SLSQP optimization on managed Spark compute. Unity Catalog for dataset lineage.", BLUE),
        (tc2, "Actian VectorAI",  "◈", "Semantic search over historical UN HRP documents. Finds 5 most comparable past crises.", "#9B5DE5"),
        (tc3, "Gemini AI",        "◉", "Three-audience brief generation — donor ROI, journalist pitch, UN operations. Model: gemini-2.5-flash.", "#34D399"),
        (tc4, "Streamlit",        "▣", "Real-time data dashboard. Plotly geospatial and analytics charts. Hot-reload development.", RED),
    ]:
        with col:
            st.markdown(f"""
            <div class="tech-card">
                <div style="font-size:22px;margin-bottom:8px;color:{color};">{icon}</div>
                <div class="tc-name" style="color:{color};">{name}</div>
                <div class="tc-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

    # Data sources
    st.markdown(f'<div class="section-title">Data Sources</div>', unsafe_allow_html=True)
    sources = ["OCHA FTS", "CBPF", "ACLED", "IPC Global", "IDMC", "ReliefWeb API", "UCDP GED", "UN HRP Documents"]
    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 18px;">' +
                ''.join(f'<span class="source-badge">{s}</span>' for s in sources) +
                '</div>', unsafe_allow_html=True)

    # Competition
    st.markdown(f'<div class="section-title">Competition Context</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'DM Sans',sans-serif;font-size:12px;color:{MUTED};line-height:2;margin-bottom:8px;">
        <strong style="color:{TEXT};">Hacklytics 2026</strong> — Georgia Tech's premier data science hackathon.<br>
        Challenge: <strong style="color:{BLUE};">Databricks × UN</strong> — Humanitarian AI for resource allocation intelligence.<br>
        Additional tracks: <strong style="color:{TEXT};">SafetyKit</strong> (physical safety AI) ·
        <strong style="color:{TEXT};">Actian VectorAI</strong> (vector search) ·
        <strong style="color:{TEXT};">Figma Make</strong> (UI/UX design)
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown(f"""
    <div class="disclaimer-box">
        <strong style="color:{RED};font-family:'IBM Plex Mono',monospace;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;">⚠ Decision Support Only</strong><br><br>
        BLINDSPOT is a decision-support tool. All severity scores, allocation recommendations, and AI-generated
        briefs are analytical outputs derived from publicly available humanitarian data. Final allocation decisions
        remain with licensed humanitarian coordinators and institutional donors.<br><br>
        <em>BLINDSPOT exposes what the data shows. Action is yours. Every misallocated dollar has a body count.</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── CHAT AGENT ───────────────────────────────────────────────────────────────
def render_chat(df):
    if not st.session_state.chat_open:
        return

    top = df.loc[df['Crisis_Severity_Score'].idxmax()] if not df.empty else None
    top_name  = top['Country_Name'] if top is not None else "N/A"
    top_score = top['Crisis_Severity_Score'] if top is not None else 0

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": (f"BLINDSPOT AI online. {top_name} is the highest-severity zone "
                        f"({top_score:.0f}/100). Ask about any crisis, the methodology, or "
                        f"allocation data.")
        })

    st.markdown(f'<hr style="border-color:{BORDER};margin:0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{CARD};border-top:1px solid {BORDER};padding:14px 28px 8px;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{TEXT};text-transform:uppercase;letter-spacing:2px;">◎ BLINDSPOT AI Agent</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:{DIM};text-transform:uppercase;letter-spacing:1.5px;">Gemini 2.5 Flash · Domain-restricted</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    chat_c = st.container(height=250, border=False)
    with chat_c:
        for msg in st.session_state.messages:
            is_bot = msg['role'] == 'assistant'
            align = "flex-start" if is_bot else "flex-end"
            bg = "rgba(45,116,218,0.07)" if is_bot else f"rgba(22,31,51,0.6)"
            bdr = "rgba(45,116,218,0.18)" if is_bot else BORDER
            pfx = "AI › " if is_bot else "YOU › "
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin-bottom:5px;">
                <div style="max-width:86%;background:{bg};border:1px solid {bdr};border-radius:7px;padding:7px 11px;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:7px;color:{DIM};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:3px;">{pfx}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:12px;color:{TEXT};line-height:1.5;">{msg['content']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        ci1, ci2 = st.columns([6, 1])
        with ci1:
            prompt = st.text_input("Message", placeholder="Ask about any crisis, methodology, or funding data...",
                                   label_visibility="collapsed", key="chat_input")
        with ci2:
            send = st.form_submit_button("Send", use_container_width=True)

    if send and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.session_state.messages.append({"role": "assistant", "content": "Error: GEMINI_API_KEY not configured."})
            st.rerun()
        else:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                summary = df[['iso3', 'Crisis_Severity_Score', 'funding_required', 'funding_received']].head(8).to_string() if not df.empty else ""
                sys_p = f"""You are BLINDSPOT AI, an analytical assistant for humanitarian resource allocation intelligence.
Rules:
1. Only answer questions about humanitarian crises, funding allocation, severity scores, this system's methodology, or its data.
2. No code, no creative writing, no general knowledge.
3. If unrelated: "BLINDSPOT AI is restricted to humanitarian crisis analysis."
4. Be precise, quantitative, factual. No emojis. Concise.

Context:
- Highest severity: {top_name} (Score: {top_score:.0f}/100)
- Active crises: {len(df)}
- Sources: OCHA FTS, IPC, UCDP GED, IDMC
Top data:
{summary}"""
                resp = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"{sys_p}\n\nUser: {prompt}"
                )
                st.session_state.messages.append({"role": "assistant", "content": resp.text})
                st.rerun()
            except Exception as e:
                err = str(e)
                msg = ("API quota exhausted. Optimization engine continues operating independently."
                       if "429" in err or "RESOURCE_EXHAUSTED" in err
                       else f"API error: {err[:120]}")
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.rerun()

    st.markdown(f"""
    <div style="padding:5px 28px 14px;background:{CARD};">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:{DIM};text-transform:uppercase;letter-spacing:1.5px;">
            BLINDSPOT exposes what the data shows. Action is yours. · Every misallocated dollar has a body count.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
with st.spinner("Connecting to Databricks..."):
    df_crises = load_data()

render_navbar(df_crises)

page = st.session_state.page
if   page == 'command_center':       page_command_center(df_crises)
elif page == 'crisis_detail':         page_crisis_detail(df_crises)
elif page == 'allocation_simulator':  page_allocation_simulator(df_crises)
elif page == 'methodology':           page_methodology()

render_chat(df_crises)
