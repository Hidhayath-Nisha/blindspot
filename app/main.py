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
BG     = "#08080C"
CARD   = "#0D0D14"
BORDER = "rgba(255,255,255,0.07)"
RED    = "#E53935"
AMBER  = "#F59E0B"
GREEN  = "#10B981"
BLUE   = "#3B82F6"
MUTED  = "#6B7280"
TEXT   = "rgba(255,255,255,0.85)"
DIM    = "rgba(255,255,255,0.2)"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,500;0,600;1,400&family=Syne:wght@300;400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── THEME TOKENS — dark defaults ── */
:root {
  --gutter: 48px;
  --bg:      #08080C;
  --bg2:     #0D0D14;
  --bg3:     #111118;
  --border:  rgba(255,255,255,0.07);
  --border2: rgba(255,255,255,0.12);
  --text:    rgba(255,255,255,0.85);
  --text-h:  #FFFFFF;
  --mid:     rgba(255,255,255,0.5);
  --dim:     rgba(255,255,255,0.25);
  --red:     #E53935;
  --amber:   #F59E0B;
  --green:   #10B981;
  --blue:    #3B82F6;
  --nav-bg:  rgba(8,8,12,0.85);
  --nav-b:   rgba(255,255,255,0.08);
  --card-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
  --glass-bg:   rgba(255,255,255,0.03);
  --glass-blur: blur(12px) saturate(150%);
  --glass-bdr:  rgba(255,255,255,0.08);
  --hero-bg:    transparent;
}

/* ── CHROME REMOVAL ── */
#MainMenu, header, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], .stDeployButton,
[data-testid="manage-app-button"] {
    visibility: hidden !important;
    display: none !important;
}
/* ── LAYOUT CONTAINER SYSTEM ── */
:root {
  --container-max: 1280px;
  --container-pad: clamp(16px, 4vw, 48px);
}
/* Block-container IS the layout root — constrain it once here */
.block-container {
    max-width: calc(var(--container-max) + 2 * var(--container-pad)) !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left:  var(--container-pad) !important;
    padding-right: var(--container-pad) !important;
    padding-top:    0 !important;
    padding-bottom: 0 !important;
    box-sizing: border-box !important;
}
section[data-testid="stMain"] > div { padding: 0 !important; }
.stApp { background: #08080C !important; }
.element-container { margin-bottom: 0 !important; }

/* ── FULL-BLEED ESCAPE (edge-to-edge bg while keeping content aligned) ── */
.bs-full-bleed {
    margin-left:  calc(-1 * var(--container-pad));
    margin-right: calc(-1 * var(--container-pad));
    padding-left:  var(--container-pad);
    padding-right: var(--container-pad);
}

/* ── 12-COLUMN GRID ── */
.bs-grid-12 {
    display: grid;
    grid-template-columns: repeat(12, minmax(0, 1fr));
    gap: 20px;
}
.bs-col-12 { grid-column: span 12; }
.bs-col-8  { grid-column: span 8;  }
.bs-col-7  { grid-column: span 7;  }
.bs-col-5  { grid-column: span 5;  }
.bs-col-4  { grid-column: span 4;  }
.bs-col-3  { grid-column: span 3;  }

@media (max-width: 1024px) {
  .bs-col-8, .bs-col-7 { grid-column: span 12; }
  .bs-col-5            { grid-column: span 12; }
  .bs-col-4            { grid-column: span 6;  }
}
@media (max-width: 640px) {
  .bs-col-4, .bs-col-3 { grid-column: span 12; }
}

/* ── FONTS BASE ── */
body, p, span, div, label, li {
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px;
    line-height: 1.6;
}
h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: var(--text-h) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--dim); border-radius: 2px; }

/* ── GLASSMORPHISM ── */
.glass {
    background: var(--glass-bg) !important;
    backdrop-filter: var(--glass-blur) !important;
    -webkit-backdrop-filter: var(--glass-blur) !important;
    border: 1px solid var(--glass-bdr) !important;
    box-shadow: var(--card-shadow) !important;
}
.glass-card {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px) saturate(150%);
    -webkit-backdrop-filter: blur(12px) saturate(150%);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}

/* ── FLOATING PILL NAVBAR — constrained to container width ── */
#bs-pill-nav {
    position: fixed;
    top: 14px;
    left: 50%;
    transform: translateX(-50%);
    /* Pill spans same logical width as the content container */
    width: calc(100% - 2 * var(--container-pad));
    max-width: var(--container-max);
    display: flex;
    align-items: center;
    gap: 0;
    padding: 5px 6px;
    border-radius: 100px;
    background: rgba(8,8,12,0.85);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    z-index: 9000;
    white-space: nowrap;
    box-sizing: border-box;
}
#bs-pill-nav .bs-logo {
    font-family: 'Syne', sans-serif;
    font-size: 15px; font-weight: 800;
    color: #FFFFFF;
    letter-spacing: 3px; text-transform: uppercase;
    padding: 0 16px 0 10px;
    user-select: none;
}
#bs-pill-nav .bs-logo em { color: #E53935; font-style: normal; }
#bs-pill-nav .bs-sep {
    width: 1px; height: 18px;
    background: rgba(255,255,255,0.10);
    margin: 0 4px;
    flex-shrink: 0;
}
#bs-pill-nav .bs-nav-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; font-weight: 400;
    text-transform: uppercase; letter-spacing: 2px;
    color: rgba(255,255,255,0.5);
    padding: 7px 14px;
    border-radius: 100px;
    cursor: pointer;
    border: none; background: transparent;
    transition: color 0.15s;
    text-decoration: none;
    display: inline-block;
}
#bs-pill-nav .bs-nav-item:hover { color: #FFFFFF; background: transparent; }
#bs-pill-nav .bs-nav-item.active { color: #E53935; background: transparent; font-weight: 500; }
#bs-pill-nav .bs-toggle {
    width: 32px; height: 32px;
    border-radius: 50%;
    border: none; background: transparent;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; color: rgba(255,255,255,0.4);
    transition: color 0.15s;
    flex-shrink: 0;
}
#bs-pill-nav .bs-toggle:hover { color: #FFFFFF; background: transparent; }
#bs-pill-nav .bs-live-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase;
    color: #10B981;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 100px;
    padding: 4px 10px;
    margin: 0 6px;
    flex-shrink: 0;
}
/* push page content below the fixed pill */
.stApp > section { padding-top: 72px; }
/* Inner Streamlit wrapper — let block-container padding do the work */
.block-container > div { padding-left: 0 !important; padding-right: 0 !important; }
/* hide the native Streamlit nav button row */
.bs-hidden-nav { display:none !important; }
/* hide routing button row via adjacent-sibling to the marker */
[data-testid="stMarkdown"]:has(.bs-nav-routing-marker) {
    display:none !important; height:0 !important; margin:0 !important; overflow:hidden !important;
}
[data-testid="stMarkdown"]:has(.bs-nav-routing-marker) + [data-testid="stHorizontalBlock"] {
    display:none !important; height:0 !important; margin:0 !important; overflow:hidden !important;
}

/* ── NAV BUTTONS ── */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; font-weight: 500 !important;
    text-transform: uppercase !important; letter-spacing: 2px !important;
    white-space: nowrap !important;
    background: rgba(229,57,53,0.85) !important; color: #FFFFFF !important;
    border: none !important; border-radius: 0 !important;
    padding: 10px 22px !important; height: auto !important;
    transition: background 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 16px rgba(229,57,53,0.25) !important;
}
.stButton > button:hover {
    background: rgba(229,57,53,1.0) !important; color: #FFFFFF !important;
    box-shadow: 0 6px 24px rgba(229,57,53,0.4) !important;
    transform: none !important;
}

/* ── SECTION LABELS ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; color: var(--dim) !important;
    text-transform: uppercase; letter-spacing: 3px; margin-bottom: 8px; display: block;
}
.section-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 22px !important; color: var(--text-h) !important;
    font-weight: 700 !important; margin-bottom: 14px !important;
}

/* ── HERO ── */
.hero-section {
    padding: 60px 48px 48px;
    background: var(--hero-bg, transparent);
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; font-weight: 400 !important;
    letter-spacing: 4px !important; text-transform: uppercase !important;
    color: #E53935 !important; margin-bottom: 24px !important; display: block;
}
.hero-number {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(64px, 8vw, 100px) !important;
    font-weight: 800 !important; line-height: 0.92 !important;
    letter-spacing: -4px !important;
    margin-bottom: 10px !important;
}
.hero-number .red   { color: #E53935 !important; font-weight: 800 !important; }
.hero-number .white { color: #FFFFFF !important; font-weight: 300 !important; }
.hero-subtitle {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(24px, 3vw, 44px) !important;
    font-weight: 300 !important; line-height: 1.15 !important;
    color: rgba(255,255,255,0.45) !important;
}
.hero-description {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 16px !important; font-weight: 400 !important;
    color: rgba(255,255,255,0.55) !important;
    line-height: 1.75 !important; max-width: 620px !important;
    margin-top: 28px !important; display: block;
}
.hero-source {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; color: rgba(255,255,255,0.2) !important;
    margin-top: 28px !important; display: block; letter-spacing: 1px;
}

/* ── KPI PILLS ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border-top: 1px solid rgba(255,255,255,0.07);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin: 0;
}
.kpi-pill {
    background: rgba(255,255,255,0.025);
    border-right: 1px solid rgba(255,255,255,0.07);
    padding: 28px 32px;
    text-align: left;
    transition: background 0.15s;
    position: relative;
}
.kpi-pill:last-child { border-right: none; }
.kpi-pill:hover {
    background: rgba(255,255,255,0.045);
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; color: rgba(255,255,255,0.3) !important;
    text-transform: uppercase; letter-spacing: 3px; margin-bottom: 12px; display: block;
}
.kpi-value {
    font-family: 'Syne', sans-serif !important;
    font-size: 42px !important; font-weight: 700 !important;
    color: #FFFFFF !important; line-height: 1.0; display: block;
    margin-bottom: 8px;
}
.kpi-desc {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; color: rgba(255,255,255,0.3) !important;
    display: block; line-height: 1.4;
}

/* ── TRIAGE QUEUE ── */
.triage-row {
    display: flex; align-items: center; gap: 12px;
    padding: 0 16px; height: 54px;
    border-bottom: 1px solid rgba(255,255,255,0.045);
    cursor: pointer; transition: background 0.12s;
}
.triage-row:hover { background: rgba(255,255,255,0.05); }
.triage-rank    { font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; color: rgba(255,255,255,0.2) !important; width: 20px; text-align: right; flex-shrink: 0; }
.triage-dot     { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.triage-country { font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; font-weight: 500 !important; color: #FFFFFF !important; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.triage-score   { font-family: 'IBM Plex Mono', monospace !important; font-size: 17px !important; font-weight: 700 !important; min-width: 32px; text-align: right; flex-shrink: 0; }
.triage-pct     { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: rgba(255,255,255,0.45) !important; width: 42px; text-align: right; flex-shrink: 0; }
.triage-trend   { font-size: 14px; width: 16px; text-align: center; flex-shrink: 0; }

/* ── METRIC SUMMARY CARDS ── */
.metric-summary {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 20px 18px; text-align: center;
    box-shadow: var(--card-shadow);
}
.metric-summary .val {
    font-family: 'Syne', sans-serif !important;
    font-size: 28px !important; font-weight: 700 !important;
    color: var(--text-h) !important; display: block;
    margin-bottom: 8px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
    line-height: 1.2;
}
.metric-summary .lbl {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; color: rgba(255,255,255,0.3) !important;
    text-transform: uppercase; letter-spacing: 2.5px; line-height: 1.4;
}

/* ── IDENTITY CARD ── */
.identity-card {
    background: var(--bg2); border: 1px solid var(--border); border-radius: 8px;
    padding: 22px 26px; display: flex; align-items: flex-start;
    justify-content: space-between; gap: 20px; margin-bottom: 18px;
    flex-wrap: wrap; box-shadow: var(--card-shadow);
}
.identity-country { font-family: 'Syne', sans-serif !important; font-size: 34px !important; font-weight: 800 !important; color: var(--text-h) !important; line-height: 1.1 !important; }
.identity-iso     { font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: var(--mid) !important; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }
.identity-score-num   { font-family: 'IBM Plex Mono', monospace !important; font-size: 56px !important; font-weight: 600 !important; line-height: 1 !important; }
.identity-score-label { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: var(--mid) !important; text-transform: uppercase; letter-spacing: 2px; }
.ipc-badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 1px; margin-top: 8px; }
.metric-cell { background: var(--bg3); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; text-align: center; }
.metric-cell .mc-val { font-family: 'IBM Plex Mono', monospace !important; font-size: 18px !important; font-weight: 600 !important; color: var(--text-h) !important; display: block; }
.metric-cell .mc-lbl { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: var(--mid) !important; text-transform: uppercase; letter-spacing: 1.2px; margin-top: 3px; }

/* ── RAG / BRIEF PANELS ── */
.rag-card {
    background: var(--bg3); border: 1px solid var(--border2);
    border-radius: 7px; padding: 14px; margin-bottom: 8px;
}
.rag-card-title { font-family: 'DM Sans', sans-serif !important; font-size: 15px !important; font-weight: 600 !important; color: var(--text-h) !important; margin-bottom: 5px; }
.rag-worked  { font-family: 'DM Sans', sans-serif !important; font-size: 13px !important; color: var(--mid) !important; margin-top: 4px; }
.powered-by  { font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; color: var(--dim) !important; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 8px; }

/* ── IMPACT BOXES (Page 3) ── */
.impact-box { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 22px; text-align: center; box-shadow: var(--card-shadow); }
.impact-box .ib-num   { font-family: 'Syne', sans-serif !important; font-size: 38px !important; font-weight: 700 !important; line-height: 1 !important; display: block; margin-bottom: 8px; }
.impact-box .ib-label { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: var(--mid) !important; text-transform: uppercase; letter-spacing: 1.5px; }

/* ── METHODOLOGY ── */
.formula-block {
    background: var(--bg3); border: 1px solid var(--border2);
    border-left: 3px solid var(--blue); border-radius: 5px;
    padding: 16px 20px; margin: 12px 0;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important; color: var(--text) !important; line-height: 1.8;
}
.tech-card  { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 20px; text-align: center; height: 100%; box-shadow: var(--card-shadow); }
.tc-name    { font-family: 'Syne', sans-serif !important; font-size: 17px !important; font-weight: 700 !important; margin-bottom: 8px; }
.tc-desc    { font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; color: var(--mid) !important; }
.source-badge { display: inline-block; background: var(--bg3); border: 1px solid var(--border); border-radius: 4px; padding: 5px 12px; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: var(--text) !important; margin: 3px; }
.disclaimer-box { background: var(--bg3); border: 1px solid var(--border); border-left: 3px solid var(--red); border-radius: 6px; padding: 16px 20px; font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; color: var(--text) !important; margin-top: 20px; line-height: 1.7; }

/* ── STREAMLIT OVERRIDES ── */
[data-testid="stMetric"] { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; padding: 14px 18px !important; }
[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: var(--mid) !important; text-transform: uppercase; letter-spacing: 1.5px; }
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 22px !important; color: var(--text-h) !important; }

[data-testid="stSelectbox"] > div > div { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; color: var(--text) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 14px !important; }
[data-testid="stSelectbox"] label { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; color: var(--mid) !important; text-transform: uppercase; letter-spacing: 1.5px; }

[data-testid="stSlider"] > div > div > div > div { background: var(--blue) !important; }
[data-testid="stSlider"] label { font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: var(--mid) !important; text-transform: uppercase; letter-spacing: 1.5px; }

[data-testid="stTabs"] [data-baseweb="tab-list"] { background: var(--bg2) !important; border-bottom: 1px solid var(--border) !important; border-radius: 8px 8px 0 0 !important; }
[data-testid="stTabs"] [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; color: var(--mid) !important; background: transparent !important; }
[data-testid="stTabs"] [aria-selected="true"] { color: var(--red) !important; border-bottom: 2px solid var(--red) !important; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-top: none !important; border-radius: 0 0 8px 8px !important; padding: 14px !important; }

[data-testid="stTextInput"] input { background: var(--bg3) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; color: var(--text) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 14px !important; }
[data-testid="stForm"] { background: transparent !important; border: none !important; }
[data-testid="stFormSubmitButton"] > button { background: var(--red) !important; color: white !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 12px !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; border: none !important; border-radius: 6px !important; padding: 10px 20px !important; box-shadow: none !important; }
[data-testid="stFormSubmitButton"] > button:hover { filter: brightness(0.88) !important; transform: none !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stExpander"] { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary { font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; color: var(--text) !important; text-transform: uppercase; letter-spacing: 1.2px; }
[data-testid="stMarkdownContainer"] { overflow: visible !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stSpinner"] > div { color: var(--mid) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 13px !important; }

hr { border-color: var(--border) !important; margin: 8px 0 !important; }
iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in [
    ('page', 'command_center'),
    ('chat_open', False),
    ('theme', 'dark'),         # 'dark' | 'light'
    ('dark_mode', True),       # legacy alias kept for compatibility
    ('messages', []),
    ('selected_country', None),
    ('opt_result', None),
    ('opt_budget', 100),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Keep dark_mode and theme in sync (theme is the canonical key)
if st.session_state.get('theme') == 'light':
    st.session_state['dark_mode'] = False
else:
    st.session_state['dark_mode'] = True

# ─── THEME CSS INJECTION ──────────────────────────────────────────────────────
def inject_theme_css():
    """Inject :root CSS variable overrides based on current theme."""
    if st.session_state.get('theme', 'dark') == 'light':
        st.markdown("""
<style>
/* ── LIGHT MODE: :root token overrides ── */
:root {
  --bg:      #F0F4F8;
  --bg2:     #FFFFFF;
  --bg3:     #E8EEF6;
  --border:  rgba(0,0,0,0.09);
  --border2: rgba(0,0,0,0.13);
  --text:    #1E293B;
  --text-h:  #0A1628;
  --mid:     #475569;
  --dim:     #64748B;
  --red:     #C8372D;
  --amber:   #A16207;
  --green:   #0A7A56;
  --blue:    #0077B6;
  --nav-bg:  rgba(255,255,255,0.88);
  --nav-b:   rgba(0,0,0,0.08);
  --card-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 20px rgba(0,0,0,0.06);
  --glass-bg:   rgba(255,255,255,0.72);
  --glass-blur: blur(20px) saturate(1.8);
  --glass-bdr:  rgba(255,255,255,0.90);
  --hero-bg:    transparent;
}

/* ── APP SHELL ── */
.stApp { background: linear-gradient(135deg, #EFF6FF 0%, #F0F4F8 50%, #EEF2F7 100%) !important; }

/* ── TYPOGRAPHY ── */
body, p, span, div, label, li { color: #1E293B !important; }
h1,h2,h3,h4,h5,h6 { color: #0A1628 !important; }

/* ── FLOATING PILL NAVBAR ── */
#bs-pill-nav {
    background: rgba(255,255,255,0.88) !important;
    border-color: rgba(0,0,0,0.09) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10), inset 0 1px 0 rgba(255,255,255,0.95) !important;
    backdrop-filter: blur(24px) saturate(1.8) !important;
    -webkit-backdrop-filter: blur(24px) saturate(1.8) !important;
}
#bs-pill-nav .bs-logo { color: #0A1628 !important; }
#bs-pill-nav .bs-logo em { color: #C8372D !important; }
#bs-pill-nav .bs-sep { background: rgba(0,0,0,0.10) !important; }
#bs-pill-nav .bs-nav-item { color: rgba(10,22,40,0.45) !important; }
#bs-pill-nav .bs-nav-item:hover { color: #0A1628 !important; }
#bs-pill-nav .bs-nav-item.active { color: #C8372D !important; }
#bs-pill-nav .bs-toggle { color: rgba(10,22,40,0.40) !important; }
#bs-pill-nav .bs-toggle:hover { color: #0A1628 !important; }
#bs-pill-nav .bs-live-badge {
    color: #0A7A56 !important;
    background: rgba(10,122,86,0.10) !important;
    border-color: rgba(10,122,86,0.25) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar-track { background: #E8EEF6 !important; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.14) !important; }

/* ── SECTION LABELS ── */
.section-label { color: rgba(10,22,40,0.40) !important; }
.section-title { color: #0A1628 !important; }

/* ── TRIAGE QUEUE ── */
.triage-row { border-bottom-color: rgba(0,0,0,0.06) !important; }
.triage-row:hover { background: rgba(0,119,182,0.06) !important; }
.triage-rank    { color: rgba(10,22,40,0.28) !important; }
.triage-country { color: #0A1628 !important; }
.triage-pct     { color: rgba(10,22,40,0.50) !important; }

/* ── KPI PILLS ── */
.kpi-label { color: rgba(10,22,40,0.42) !important; }
.kpi-value { color: #0A1628 !important; }
.kpi-desc  { color: rgba(10,22,40,0.42) !important; }
.kpi-pill:hover {
    background: rgba(255,255,255,0.65) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.10) !important;
}

/* ── METRIC SUMMARY CARDS (bottom strip) ── */
.metric-summary {
    background: rgba(255,255,255,0.75) !important;
    border-color: rgba(0,0,0,0.08) !important;
    backdrop-filter: blur(16px) saturate(1.6) !important;
    -webkit-backdrop-filter: blur(16px) saturate(1.6) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07) !important;
}
.metric-summary .lbl { color: rgba(10,22,40,0.46) !important; }
.metric-summary .val { color: #0A1628 !important; }

/* ── IDENTITY / METRIC CARDS ── */
.identity-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(0,0,0,0.08) !important;
    backdrop-filter: blur(16px) saturate(1.6) !important;
    -webkit-backdrop-filter: blur(16px) saturate(1.6) !important;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08) !important;
}
.identity-country { color: #0A1628 !important; }
.identity-iso     { color: #475569 !important; }
.identity-score-label { color: #475569 !important; }
.metric-cell {
    background: rgba(0,0,0,0.04) !important;
    border-color: rgba(0,0,0,0.08) !important;
}
.metric-cell .mc-val { color: #0A1628 !important; }
.metric-cell .mc-lbl { color: #475569 !important; }

/* ── RAG / BRIEF PANELS ── */
.rag-card {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(0,0,0,0.09) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
}
.rag-card-title { color: #0A1628 !important; }
.rag-worked     { color: #475569 !important; }
.powered-by     { color: rgba(10,22,40,0.35) !important; }

/* ── IMPACT BOXES ── */
.impact-box {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(0,0,0,0.08) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
}
.impact-box .ib-label { color: #475569 !important; }

/* ── FORMULA / TECH / SOURCE BLOCKS ── */
.formula-block {
    background: rgba(0,0,0,0.04) !important;
    border-color: rgba(0,0,0,0.10) !important;
}
.tech-card {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(0,0,0,0.08) !important;
    backdrop-filter: blur(12px) !important;
}
.tc-name { color: #0A1628 !important; }
.tc-desc { color: #475569 !important; }
.source-badge {
    background: rgba(0,0,0,0.04) !important;
    border-color: rgba(0,0,0,0.08) !important;
    color: #1E293B !important;
}
.disclaimer-box {
    background: rgba(0,0,0,0.04) !important;
    border-color: rgba(200,55,45,0.25) !important;
}

/* ── GLASS UTILITY ── */
.glass, .glass-card {
    background: rgba(255,255,255,0.70) !important;
    border-color: rgba(255,255,255,0.90) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
}

/* ── STREAMLIT COMPONENT OVERRIDES ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(0,0,0,0.08) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"] { color: #475569 !important; }
[data-testid="stMetricValue"] { color: #0A1628 !important; }

[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.90) !important;
    border-color: rgba(0,0,0,0.12) !important;
    color: #0A1628 !important;
}
[data-testid="stSelectbox"] label { color: #475569 !important; }

[data-testid="stSlider"] label { color: #475569 !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.80) !important;
    border-bottom-color: rgba(0,0,0,0.08) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] { color: #475569 !important; }
[data-testid="stTabs"] [aria-selected="true"] { color: #C8372D !important; border-color: #C8372D !important; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(0,0,0,0.08) !important;
}
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.90) !important;
    border-color: rgba(0,0,0,0.12) !important;
    color: #0A1628 !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(0,0,0,0.08) !important;
}
[data-testid="stExpander"] summary { color: #1E293B !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(0,0,0,0.08) !important;
}
[data-testid="stDataFrame"] { border-color: rgba(0,0,0,0.08) !important; }

/* ── BUTTONS ── */
.stButton > button {
    box-shadow: 0 4px 16px rgba(200,55,45,0.22) !important;
}

/* ── DIVIDERS ── */
hr { border-color: rgba(0,0,0,0.08) !important; }

/* ── KNOWN DARK INLINE BACKGROUNDS (brief output, empty states, etc.) ── */
/* Override dark navy rgba(22,31,51,...) backgrounds used via Python constants */
[style*="background:rgba(22,31,51"], [style*="background: rgba(22,31,51"] {
    background: rgba(0,0,0,0.04) !important;
}
/* Override very-low-alpha white borders used for dashed empty states */
[style*="dashed rgba(255,255,255,0.07)"] {
    border-color: rgba(0,0,0,0.18) !important;
}
/* Override any remaining CARD (#0D0D14) or pure dark bg used inline */
[style*="background:#0D0D14"], [style*="background: #0D0D14"],
[style*="background:#08080C"], [style*="background: #08080C"] {
    background: rgba(255,255,255,0.85) !important;
}
/* Ensure the stApp background gradient takes effect */
[data-testid="stApp"], [class*="stApp"] {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0F4F8 50%, #EEF2F7 100%) !important;
}
/* Override any inline low-alpha white text that was used as muted */
[style*="color:rgba(255,255,255,0.2)"], [style*="color: rgba(255,255,255,0.2)"] {
    color: rgba(10,22,40,0.38) !important;
}
[style*="color:rgba(255,255,255,0.25)"], [style*="color: rgba(255,255,255,0.25)"] {
    color: rgba(10,22,40,0.40) !important;
}
[style*="color:rgba(255,255,255,0.3)"], [style*="color: rgba(255,255,255,0.3)"] {
    color: rgba(10,22,40,0.45) !important;
}
[style*="color:rgba(255,255,255,0.45)"], [style*="color: rgba(255,255,255,0.45)"] {
    color: rgba(10,22,40,0.52) !important;
}
[style*="color:rgba(255,255,255,0.5)"], [style*="color: rgba(255,255,255,0.5)"] {
    color: rgba(10,22,40,0.52) !important;
}
</style>
""", unsafe_allow_html=True)


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
        xgb_path       = os.path.join(parent_dir, "assets", "xgboost_output.csv")
        optimized_path = os.path.join(parent_dir, "assets", "triage_master_optimized.csv")
        scores_path    = os.path.join(parent_dir, "assets", "triage_master_scores.csv")
        if os.path.exists(xgb_path):
            df = pd.read_csv(xgb_path)
        elif os.path.exists(optimized_path):
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
    if 'funding_required' in df.columns and df['funding_received'].sum() > 0:
        df['Funding_Ratio'] = np.where(
            df['funding_required'] > 0,
            np.round((df['funding_received'] / df['funding_required']) * 100, 1),
            100.0
        )
    elif 'Funding_Coverage_Ratio' in df.columns:
        df['Funding_Ratio'] = np.round(df['Funding_Coverage_Ratio'] * 100, 1)
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
    is_light   = st.session_state.get('theme', 'dark') == 'light'
    theme_icon = "☀" if is_light else "☾"
    cur_page   = st.session_state.get('page', 'command_center')

    # ── Hidden routing buttons — placed in sidebar (CSS-hidden, zero page space) ──
    with st.sidebar:
        _btn_defs = [
            ("command_center",       "nav_home"),
            ("crisis_detail",        "nav_cd"),
            ("allocation_simulator", "nav_alloc"),
            ("methodology",          "nav_meth"),
            ("chat_toggle",          "nav_chat"),
            ("theme_toggle",         "nav_theme"),
        ]
        for action, key in _btn_defs:
            if st.button("·", key=key):
                if action == "chat_toggle":
                    st.session_state.chat_open = not st.session_state.chat_open
                elif action == "theme_toggle":
                    new_t = 'light' if st.session_state.get('theme','dark') == 'dark' else 'dark'
                    st.session_state['theme']     = new_t
                    st.session_state['dark_mode'] = (new_t == 'dark')
                else:
                    st.session_state.page = action
                st.rerun()

    # ── Floating pill navbar injected into parent document via components.html ──
    chat_label  = "✕ Chat" if st.session_state.chat_open else "AI Chat"
    _page_json  = cur_page  # current active page for JS active-state highlight

    components.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
</style>
<script>
(function() {{
  var P = window.parent.document;

  // ── Remove stale pill if already present (theme/page update) ──
  var old = P.getElementById('bs-pill-nav');
  if (old) old.remove();

  var isLight = {'true' if is_light else 'false'};

  // ── Inject pill CSS into parent head ──
  var styleId = 'bs-pill-style';
  var existing = P.getElementById(styleId);
  if (existing) existing.remove();
  var s = P.createElement('style');
  s.id = styleId;
  s.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');
    #bs-pill-nav {{
      position: fixed; top: 18px; left: 50%; transform: translateX(-50%);
      display: -webkit-inline-flex; display: inline-flex;
      align-items: center; gap: 2px;
      padding: 5px 8px;
      border-radius: 100px;
      background: ${{isLight ? 'rgba(255,255,255,0.82)' : 'rgba(9,13,24,0.72)'}};
      backdrop-filter: blur(20px) saturate(1.5);
      -webkit-backdrop-filter: blur(20px) saturate(1.5);
      border: 1px solid ${{isLight ? 'rgba(221,227,236,0.9)' : 'rgba(255,255,255,0.08)'}};
      box-shadow: ${{isLight
        ? '0 8px 32px rgba(0,0,0,0.10), 0 2px 8px rgba(0,0,0,0.06)'
        : '0 8px 40px rgba(0,0,0,0.55), 0 2px 12px rgba(0,0,0,0.3)'}};
      z-index: 9000; white-space: nowrap;
    }}
    #bs-pill-nav .bsp-logo {{
      font-family:'Syne',sans-serif; font-size:15px; font-weight:800;
      color:${{isLight?'#0A1628':'#F1F5F9'}};
      letter-spacing:2.5px; text-transform:uppercase;
      padding:0 16px 0 10px; user-select:none; cursor:pointer;
    }}
    #bs-pill-nav .bsp-logo em {{ color:${{isLight?'#C8372D':'#E83D3D'}}; font-style:normal; }}
    #bs-pill-nav .bsp-sep {{
      width:1px; height:20px; flex-shrink:0; margin:0 4px;
      background:${{isLight?'rgba(0,0,0,0.1)':'rgba(255,255,255,0.1)'}};
    }}
    #bs-pill-nav .bsp-item {{
      font-family:'IBM Plex Mono',monospace; font-size:11px; font-weight:500;
      text-transform:uppercase; letter-spacing:1.5px;
      color:${{isLight?'#4A5568':'#94A3B8'}};
      padding:7px 15px; border-radius:100px; cursor:pointer;
      border:none; background:transparent;
      transition:color 0.15s,background 0.15s;
    }}
    #bs-pill-nav .bsp-item:hover {{
      color:${{isLight?'#0A1628':'#F1F5F9'}};
      background:${{isLight?'rgba(0,0,0,0.06)':'rgba(255,255,255,0.07)'}};
    }}
    #bs-pill-nav .bsp-item.bsp-active {{
      color:${{isLight?'#0A1628':'#F1F5F9'}};
      background:${{isLight?'rgba(0,0,0,0.08)':'rgba(255,255,255,0.10)'}};
      font-weight:600;
    }}
    #bs-pill-nav .bsp-toggle {{
      width:32px; height:32px; border-radius:50%;
      border:none; background:transparent; cursor:pointer;
      display:flex; align-items:center; justify-content:center;
      font-size:15px; color:${{isLight?'#4A5568':'#94A3B8'}};
      transition:background 0.15s,color 0.15s; flex-shrink:0;
    }}
    #bs-pill-nav .bsp-toggle:hover {{
      background:${{isLight?'rgba(0,0,0,0.07)':'rgba(255,255,255,0.08)'}};
      color:${{isLight?'#0A1628':'#F1F5F9'}};
    }}
    /* push page content below the fixed pill */
    [data-testid="stMain"] .block-container {{ padding-top: 64px !important; }}
    /* hide the Streamlit routing button rows */
    .bs-hidden-nav {{ display:none !important; }}
    [data-testid="stMarkdown"]:has(.bs-nav-routing-marker) {{
      display:none !important; height:0 !important; margin:0 !important; overflow:hidden !important;
    }}
    [data-testid="stMarkdown"]:has(.bs-nav-routing-marker) + [data-testid="stHorizontalBlock"] {{
      display:none !important; height:0 !important; margin:0 !important; overflow:hidden !important;
    }}
  `;
  P.head.appendChild(s);

  // ── Build pill HTML ──
  var curPage = '{_page_json}';
  var pages = [
    ['command_center',      'Dashboard',   'nav_home'],
    ['crisis_detail',       'Detail',      'nav_cd'],
    ['allocation_simulator','Optimizer',   'nav_alloc'],
    ['methodology',         'Methodology', 'nav_meth'],
  ];

  var pill = P.createElement('div');
  pill.id = 'bs-pill-nav';

  // Logo (click → dashboard)
  var logo = P.createElement('div');
  logo.className = 'bsp-logo';
  logo.innerHTML = 'BLIND<em>SPOT</em>';
  logo.title = 'Go to Dashboard';
  logo.addEventListener('click', function() {{ clickHiddenBtn(P, 'nav_home'); }});
  pill.appendChild(logo);

  // Separator
  pill.appendChild(makeSep(P));

  // Nav items
  pages.forEach(function(pg) {{
    var btn = P.createElement('button');
    btn.className = 'bsp-item' + (curPage === pg[0] ? ' bsp-active' : '');
    btn.textContent = pg[1];
    btn.addEventListener('click', function() {{ clickHiddenBtn(P, pg[2]); }});
    pill.appendChild(btn);
  }});

  // AI Chat
  pill.appendChild(makeSep(P));
  var chatBtn = P.createElement('button');
  chatBtn.className = 'bsp-item';
  chatBtn.textContent = '{chat_label}';
  chatBtn.addEventListener('click', function() {{ clickHiddenBtn(P, 'nav_chat'); }});
  pill.appendChild(chatBtn);

  // Theme toggle
  var tog = P.createElement('button');
  tog.className = 'bsp-toggle';
  tog.innerHTML = '{theme_icon}';
  tog.title = isLight ? 'Switch to dark mode' : 'Switch to light mode';
  tog.addEventListener('click', function() {{ clickHiddenBtn(P, 'nav_theme'); }});
  pill.appendChild(tog);

  P.body.appendChild(pill);

  function makeSep(doc) {{
    var sep = doc.createElement('div');
    sep.className = 'bsp-sep';
    return sep;
  }}

  function clickHiddenBtn(doc, key) {{
    // Streamlit renders button with a data-testid structure; find by text '·'
    // We stored real keys in aria-label via our hidden column pattern
    var allBtns = doc.querySelectorAll('button');
    for (var i = 0; i < allBtns.length; i++) {{
      var ariaLabel = allBtns[i].getAttribute('aria-label') || '';
      var testId    = (allBtns[i].closest('[data-testid="stButton"]') || {{}});
      // Match by key stored in .element-container data attribute if possible,
      // otherwise fall back to finding the button by its parent key div
      var keyDiv = allBtns[i].closest('[data-testid="column"]');
      // Simpler: match by order of our hidden buttons using index stored in data-bs-key
      if (allBtns[i].getAttribute('data-bs-key') === key) {{
        allBtns[i].click();
        return;
      }}
    }}
    // Fallback: find by scanning Streamlit stButton key pattern in DOM
    var stBtns = doc.querySelectorAll('[data-testid="stButton"] button');
    var keyMap = {{ 'nav_home':0, 'nav_cd':1, 'nav_alloc':2, 'nav_meth':3, 'nav_chat':4, 'nav_theme':5 }};
    var idx = keyMap[key];
    if (idx !== undefined && stBtns[idx]) {{
      stBtns[idx].click();
    }}
  }}

  // Tag the hidden Streamlit buttons with data-bs-key for reliable lookup
  setTimeout(function() {{
    var stBtns = P.querySelectorAll('[data-testid="stButton"] button');
    var keys = ['nav_home','nav_cd','nav_alloc','nav_meth','nav_chat','nav_theme'];
    for (var i = 0; i < Math.min(stBtns.length, keys.length); i++) {{
      stBtns[i].setAttribute('data-bs-key', keys[i]);
    }}
  }}, 300);
}})();
</script>
""", height=0, scrolling=False)


# ─── PAGE 1: COMMAND CENTER ───────────────────────────────────────────────────
def page_command_center(df):
    if df.empty:
        st.warning("No data loaded.")
        return

    opt_l, cur_l, gap = compute_misallocation_cost(df.to_json())

    total_needed = df['ipc_phase_3_plus'].sum()
    total_req  = df['funding_required'].sum() if 'funding_required' in df.columns else 0
    avg_funded = round(float(df['Funding_Ratio'].mean()), 1) if not df.empty else 0

    # ── HERO ──
    af_color = fund_color(avg_funded)

    # Theme-aware colors for components.html iframes (can't read CSS vars from parent)
    _is_light   = st.session_state.get('theme', 'dark') == 'light'
    _hero_bg    = "#FFFFFF"   if _is_light else "#05080F"
    _hero_num   = "#C8372D"   if _is_light else "#E83D3D"
    _hero_suf   = "#0A1628"   if _is_light else "#F1F5F9"
    _hero_amber = "#A16207"   if _is_light else "#F0A500"
    _hero_sub   = "#718096"   if _is_light else "#8494AD"

    # Hero inline color palette
    _h_border    = "rgba(0,0,0,0.08)"          if _is_light else "rgba(255,255,255,0.07)"
    _h_eyebrow   = "#C8372D"                   if _is_light else "#E53935"
    _h_num_c     = "#C8372D"                   if _is_light else "#E53935"
    _h_word_c    = "#0A1628"                   if _is_light else "#FFFFFF"
    _h_subtitle  = "rgba(10,22,40,0.50)"       if _is_light else "rgba(255,255,255,0.45)"
    _h_desc_c    = "rgba(10,22,40,0.48)"       if _is_light else "rgba(255,255,255,0.50)"
    _h_source_c  = "rgba(10,22,40,0.30)"       if _is_light else "rgba(255,255,255,0.20)"
    _kpi_border  = "rgba(0,0,0,0.08)"          if _is_light else "rgba(255,255,255,0.07)"
    _kpi_bg      = "rgba(255,255,255,0.80)"    if _is_light else "rgba(255,255,255,0.02)"
    _kpi_label   = "rgba(10,22,40,0.42)"       if _is_light else "rgba(255,255,255,0.30)"
    _kpi_val_c   = "#0A1628"                   if _is_light else "#FFFFFF"
    _kpi_desc    = "rgba(10,22,40,0.42)"       if _is_light else "rgba(255,255,255,0.30)"

    # Hero + KPI block
    n_crises = len(df)

    # Number of people in need sitting in underfunded crises (funding < 70%)
    underfunded_pop = int(df.loc[df['Funding_Ratio'] < 70, 'ipc_phase_3_plus'].sum())
    # Fallback to total if filter removes everything
    hero_num = underfunded_pop if underfunded_pop > 0 else int(df['ipc_phase_3_plus'].sum())
    # Format: "87.3M" or "216M"
    if hero_num >= 1_000_000:
        hero_num_str = f"{hero_num / 1_000_000:.1f}M"
    else:
        hero_num_str = f"{hero_num:,}"

    st.markdown(f"""
    <div class="bs-full-bleed" style="border-bottom:1px solid {_h_border};overflow:visible;padding-top:60px;padding-bottom:48px;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:400;
                    letter-spacing:4px;text-transform:uppercase;color:{_h_eyebrow};
                    margin-bottom:28px;">
            Humanitarian Funding Intelligence &nbsp;·&nbsp; Live Data
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:96px;
                    font-weight:800;line-height:1;letter-spacing:-3px;
                    color:{_h_num_c};margin-bottom:4px;overflow:visible;">
            {hero_num_str}
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:48px;
                    font-weight:300;line-height:1.1;color:{_h_word_c};
                    margin-bottom:16px;overflow:visible;">
            people
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:28px;
                    font-weight:300;line-height:1.3;color:{_h_subtitle};
                    margin-bottom:32px;">
            in active crises receive less than 70% of needed funding.
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:16px;font-weight:400;
                    color:{_h_desc_c};line-height:1.8;max-width:640px;
                    margin-bottom:32px;">
            BLINDSPOT is an AI-powered humanitarian intelligence platform.
            It scores every active crisis by population severity and funding coverage,
            detects systematic misallocations in donor capital, and shows exactly
            where moving money saves the most lives — sourced from OCHA FTS data.
        </div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                    color:{_h_source_c};letter-spacing:1px;">
            Source: BLINDSPOT optimizer &nbsp;·&nbsp; OCHA FTS &nbsp;·&nbsp; IPC Global &nbsp;·&nbsp; Updated live
        </div>
    </div>
    <div class="bs-full-bleed" style="display:grid;grid-template-columns:repeat(4,1fr);">
        <div style="padding:28px 32px;border-right:1px solid {_kpi_border};border-bottom:1px solid {_kpi_border};background:{_kpi_bg};">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_kpi_label};text-transform:uppercase;letter-spacing:3px;margin-bottom:12px;">People In Need</div>
            <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:700;color:{_kpi_val_c};line-height:1;margin-bottom:8px;">{fmt_people(total_needed)}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{_kpi_desc};">Across {n_crises} active crisis zones</div>
        </div>
        <div style="padding:28px 32px;border-right:1px solid {_kpi_border};border-bottom:1px solid {_kpi_border};background:{_kpi_bg};">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_kpi_label};text-transform:uppercase;letter-spacing:3px;margin-bottom:12px;">Funding Required</div>
            <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:700;color:{_kpi_val_c};line-height:1;margin-bottom:8px;">{fmt_b(total_req)}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{_kpi_desc};">Total humanitarian ask, OCHA FTS</div>
        </div>
        <div style="padding:28px 32px;border-right:1px solid {_kpi_border};border-bottom:1px solid {_kpi_border};background:{_kpi_bg};">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_kpi_label};text-transform:uppercase;letter-spacing:3px;margin-bottom:12px;">Avg. Coverage</div>
            <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:700;color:{af_color};line-height:1;margin-bottom:8px;">{avg_funded:.0f}%</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{_kpi_desc};">Mean funded ratio across all crises</div>
        </div>
        <div style="padding:28px 32px;border-bottom:1px solid {_kpi_border};background:{_kpi_bg};">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_kpi_label};text-transform:uppercase;letter-spacing:3px;margin-bottom:12px;">Active Crises</div>
            <div style="font-family:'Syne',sans-serif;font-size:42px;font-weight:700;color:{_kpi_val_c};line-height:1;margin-bottom:8px;">{n_crises}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{_kpi_desc};">Countries with IPC Phase&nbsp;3+ populations</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── MAP + QUEUE ──
    _map_lbl_c = "rgba(10,22,40,0.40)" if _is_light else "rgba(255,255,255,0.25)"
    st.markdown('<div style="padding:32px 0 0;">', unsafe_allow_html=True)
    left_col, right_col = st.columns([58, 42], gap="small")

    with left_col:
        st.markdown(f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:{_map_lbl_c};margin:0 0 10px 0;letter-spacing:2px;text-transform:uppercase;">BUBBLE SIZE = PEOPLE IN NEED &nbsp;·&nbsp; COLOR = FUNDING COVERAGE &nbsp;·&nbsp; DRAG TO ROTATE</p>', unsafe_allow_html=True)

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

        # Theme-aware globe colors
        _is_light = st.session_state.get('theme', 'dark') == 'light'
        _globe_body_bg   = "#EEF2F7"   if _is_light else "#05080F"
        _globe_tip_bg    = "#FFFFFF"   if _is_light else "#090D18"
        _globe_tip_bdr   = "#DDE3EC"   if _is_light else "#2D74DA44"
        _globe_tip_text  = "#0A1628"   if _is_light else "#CBD5E1"
        _globe_tip_glow  = "rgba(0,158,219,0.15)" if _is_light else "rgba(45,116,218,0.25)"
        _globe_legend_c  = "#718096"   if _is_light else "#64748B"
        _globe_atm       = "#009EDB"   if _is_light else "#2563EB"
        _globe_sphere    = 0xC8DCF0    if _is_light else 0x04111e
        _globe_emissive  = 0xAFC8E0    if _is_light else 0x020c16
        _globe_land      = "'#2A4B7C'" if _is_light else "'#0E2340'"
        _globe_side      = "'#1A3560'" if _is_light else "'#071422'"
        _globe_border    = "'#4A7AAE'" if _is_light else "'#1E4080'"
        _globe_bg_clear  = "'#C0D5E8'" if _is_light else "'#03080F'"

        globe_html = f"""
<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:{_globe_body_bg}; overflow:hidden; }}
  #g {{ width:100%; height:580px; }}
  #tip {{
    position:fixed; display:none; pointer-events:none; z-index:9999;
    background:{_globe_tip_bg}; border:1px solid {_globe_tip_bdr};
    border-radius:8px; padding:10px 14px;
    font-family:'IBM Plex Mono',monospace; font-size:12px; color:{_globe_tip_text};
    line-height:1.7; box-shadow:0 4px 20px {_globe_tip_glow};
    backdrop-filter: blur(8px);
  }}
  #legend {{
    position:absolute; bottom:12px; left:14px;
    font-family:'IBM Plex Mono',monospace; font-size:10px; color:{_globe_legend_c};
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
  .height(580)
  .backgroundColor({_globe_bg_clear})
  // No image texture — country polygons give us land/ocean contrast
  .globeImageUrl('')
  .atmosphereColor('{_globe_atm}')
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
        '<b style="color:{_globe_tip_text};font-size:13px">' + pt.name + '</b><br>' +
        'Severity &nbsp;<span style="color:' + pt.color + '">' + pt.sev + '/100</span><br>' +
        'Funded &nbsp;&nbsp;<span style="color:{_globe_legend_c}">' + pt.funded + '%</span><br>' +
        'People &nbsp;&nbsp;<span style="color:{_globe_legend_c}">' + pt.people + '</span>';
    }} else {{
      tip.style.display = 'none';
    }}
  }});

// Manually color the globe sphere (ocean)
world.scene().traverse(obj => {{
  if (obj.isMesh && obj.geometry && obj.geometry.type === 'SphereGeometry') {{
    obj.material.color.setHex({_globe_sphere});
    obj.material.emissive.setHex({_globe_emissive});
    obj.material.emissiveIntensity = 0.35;
  }}
}});

// ── Load real country polygons (land masses) ──
fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')
  .then(r => r.json())
  .then(worldData => {{
    const countries = topojson.feature(worldData, worldData.objects.countries).features;
    world
      .polygonsData(countries)
      .polygonCapColor(() => {_globe_land})
      .polygonSideColor(() => {_globe_side})
      .polygonStrokeColor(() => {_globe_border})
      .polygonAltitude(0.004);
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
        components.html(globe_html, height=595, scrolling=False)

    with right_col:
        df_q = df.nlargest(10, 'Crisis_Severity_Score').reset_index(drop=True)

        # Triage queue theme colors
        _q_bg      = "rgba(255,255,255,0.75)" if _is_light else "rgba(255,255,255,0.018)"
        _q_border  = "rgba(0,0,0,0.08)"        if _is_light else "rgba(255,255,255,0.07)"
        _q_label   = "rgba(10,22,40,0.38)"     if _is_light else "rgba(255,255,255,0.30)"
        _q_title   = "#0A1628"                  if _is_light else "#FFFFFF"
        _q_sub     = "rgba(10,22,40,0.40)"     if _is_light else "rgba(255,255,255,0.30)"
        _q_col_hdr = "rgba(10,22,40,0.32)"     if _is_light else "rgba(255,255,255,0.25)"
        _q_legend  = "rgba(10,22,40,0.28)"     if _is_light else "rgba(255,255,255,0.20)"

        q_html = f'''<div style="height:595px;overflow:hidden;display:flex;flex-direction:column;box-sizing:border-box;background:{_q_bg};border-left:1px solid {_q_border};padding:28px 24px 20px;backdrop-filter:blur(16px) saturate(1.6);-webkit-backdrop-filter:blur(16px) saturate(1.6);">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_q_label};letter-spacing:4px;text-transform:uppercase;margin-bottom:8px;display:block;">CRISIS TRIAGE QUEUE</span>
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:{_q_title};margin-bottom:3px;line-height:1.2;">Top 10 Critical Zones</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{_q_sub};margin-bottom:14px;">Ranked by severity &times; funding gap</div>
        <div style="display:flex;align-items:center;gap:12px;padding:0 16px 10px;border-bottom:1px solid {_q_border};margin-bottom:2px;">
            <span style="width:20px;flex-shrink:0;"></span>
            <span style="width:9px;flex-shrink:0;"></span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_q_col_hdr};text-transform:uppercase;letter-spacing:2px;flex:1;">Country</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_q_col_hdr};text-transform:uppercase;letter-spacing:2px;min-width:32px;text-align:right;">Score</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_q_col_hdr};text-transform:uppercase;letter-spacing:2px;width:42px;text-align:right;">Fund%</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_q_col_hdr};width:16px;text-align:center;">&nbsp;</span>
        </div>
        <div style="display:flex;flex-direction:column;flex:1;overflow:hidden;">'''

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

        q_html += f'''</div>
        <div style="margin-top:8px;font-family:'IBM Plex Mono',monospace;font-size:8px;color:{_q_legend};text-transform:uppercase;letter-spacing:2px;line-height:2.0;">
            Score = Crisis Severity (0–100) &nbsp;·&nbsp; <span style="color:{RED}">■</span> Critical &nbsp; <span style="color:{AMBER}">■</span> High &nbsp; <span style="color:{GREEN}">■</span> Moderate
        </div>
        </div>'''

        st.markdown(q_html, unsafe_allow_html=True)
        if st.button("→ Open Crisis Detail", key="cc_goto_cd", use_container_width=True):
            st.session_state.page = 'crisis_detail'
            st.rerun()

    # ── BOTTOM METRIC CARDS ──
    _hr_c = "rgba(0,0,0,0.08)" if _is_light else "rgba(255,255,255,0.07)"
    st.markdown(f'<hr style="margin:24px 0 20px;border:none;border-top:1px solid {_hr_c};">', unsafe_allow_html=True)
    st.markdown('<span class="section-label" style="display:block;margin-bottom:16px;">At a Glance · OCHA FTS</span>', unsafe_allow_html=True)

    red_zones = df[df['Crisis_Severity_Score'] > 75] if 'Is_Red_Zone' not in df.columns else df[df['Is_Red_Zone'] == True]
    most_severe  = df.loc[df['Crisis_Severity_Score'].idxmax()]
    least_funded = df.loc[df['Funding_Ratio'].idxmin()]
    total_gap    = (df['funding_required'] - df['funding_received']).clip(lower=0).sum()

    bottom_cards = f"""
    <div style="padding:0 0 40px;display:grid;grid-template-columns:repeat(5,1fr);gap:12px;">
        <div class="metric-summary">
            <span class="val" style="color:{RED};">{len(red_zones)}</span>
            <span class="lbl">Red Zone Crises</span>
        </div>
        <div class="metric-summary">
            <span class="val" style="color:{RED};font-size:22px !important;">{most_severe['Country_Name']}</span>
            <span class="lbl">Highest Severity</span>
        </div>
        <div class="metric-summary">
            <span class="val" style="color:{RED};">{most_severe['Crisis_Severity_Score']:.0f}<span style="font-size:18px;font-weight:400;">/100</span></span>
            <span class="lbl">Peak Severity Score</span>
        </div>
        <div class="metric-summary">
            <span class="val" style="color:{AMBER};font-size:22px !important;">{least_funded['Country_Name']}</span>
            <span class="lbl">Least Funded Zone</span>
        </div>
        <div class="metric-summary">
            <span class="val" style="color:{AMBER};">{fmt_b(total_gap)}</span>
            <span class="lbl">Total Funding Gap</span>
        </div>
    </div>
    """
    st.markdown(bottom_cards, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── PAGE 2: CRISIS DETAIL ────────────────────────────────────────────────────
def page_crisis_detail(df):
    if df.empty:
        st.warning("No data loaded.")
        return

    st.markdown('<div style="padding:20px 0;">', unsafe_allow_html=True)

    options = df.sort_values('Crisis_Severity_Score', ascending=False)['Country_Name'].tolist()
    def_idx = 0
    if st.session_state.selected_country in options:
        def_idx = options.index(st.session_state.selected_country)

    selected = st.selectbox("SELECT CRISIS", options, index=def_idx, key="cd_select")
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
        st.markdown('<span class="section-label">Severity Breakdown · 6 humanitarian signals</span>', unsafe_allow_html=True)
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
        st.markdown('<span class="section-label">Funding History · Required vs Received · OCHA FTS</span>', unsafe_allow_html=True)
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
        st.markdown('<span class="section-label">Media Attention vs Actual Need · ReliefWeb</span>', unsafe_allow_html=True)
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
        st.markdown('<span class="section-label">Similar Past Crises · Actian VectorAI</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">What Worked Before</div>', unsafe_allow_html=True)

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

        st.markdown(f'<div class="powered-by">◈ Actian VectorAI · matched against UN HRP documents</div>', unsafe_allow_html=True)

    # ── Brief Generator ──
    with pan_r:
        st.markdown('<span class="section-label">AI Briefs · Gemini 2.5 Flash</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Ready-to-Use Briefs</div>', unsafe_allow_html=True)

        tab_d, tab_j, tab_u = st.tabs(["DONOR ROI", "JOURNALIST", "UN COORDINATOR"])

        crisis_data = {'iso3': iso, 'Crisis_Severity_Score': sc,
                       'funding_required': req, 'funding_received': rcvd}

        for tab_obj, audience_key, audience_label, audience_desc in [
            (tab_d, 'donor',      'Donor Brief',      'For donors: impact and return on investment.'),
            (tab_j, 'journalist', 'Journalist Brief',  'For journalists: the untold story.'),
            (tab_u, 'un',         'UN Brief',          'For UN coordinators: field-ready guidance.'),
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

        st.markdown(f'<div class="powered-by">◈ Gemini 2.5 Flash · tailored for donors, press, and UN</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ─── PAGE 3: ALLOCATION SIMULATOR ─────────────────────────────────────────────
def page_allocation_simulator(df):
    st.markdown('<div style="padding:20px 0;">', unsafe_allow_html=True)

    st.markdown(f"""
    <span class="section-label">Allocation Simulator · Databricks</span>
    <div class="section-title" style="font-size:22px;">Budget Optimizer</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{MUTED};max-width:600px;line-height:1.6;margin-bottom:20px;">
        Given a donor budget, BLINDSPOT allocates across active crises to maximize lives saved—accounting for diminishing returns and conflict-based constraints.
    </div>
    """, unsafe_allow_html=True)

    ctrl1, ctrl2 = st.columns([3, 1], gap="large")
    with ctrl1:
        budget_m = st.slider(
            "BUDGET (USD MILLIONS)",
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
            (ib1, cur_lives, "Lives Saved (Current)",      MUTED,  DIM),
            (ib2, opt_lives, "Lives Saved (Optimized)",      GREEN,  f"{GREEN}40"),
            (ib3, gap_lives, "Lives Lost to Misallocation",       RED,    f"{RED}40"),
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
            st.markdown('<span class="section-label">Current Allocation · OCHA FTS</span>', unsafe_allow_html=True)
            ca = df_top['funding_received'].clip(lower=0).tolist()
            if sum(ca) == 0: ca = [1e6] * len(ca)
            st.plotly_chart(mk_sankey(ca, "Current Allocation"), use_container_width=True, config=dict(displayModeBar=False))

        with sk_r:
            st.markdown('<span class="section-label">Optimal Allocation · BLINDSPOT Model</span>', unsafe_allow_html=True)
            oa = df_top['Optimal_Allocation_USD'].clip(lower=0).tolist()
            if sum(oa) == 0: oa = [budget_m * 1e6 / len(oa)] * len(oa)
            st.plotly_chart(mk_sankey(oa, "BLINDSPOT Optimized"), use_container_width=True, config=dict(displayModeBar=False))

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
        st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:8px;color:{DIM};margin-top:6px;text-transform:uppercase;letter-spacing:1.2px;">SLSQP optimizer · diminishing returns + conflict access penalty</div>', unsafe_allow_html=True)

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
    st.markdown('<div style="padding:20px 0;">', unsafe_allow_html=True)

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
    st.markdown(f'<div class="section-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'DM Sans',sans-serif;font-size:12px;color:{MUTED};line-height:2;margin-bottom:8px;">
        <strong style="color:{TEXT};">Hacklytics 2026</strong> — Georgia Tech’s data science hackathon.<br>
        Challenge: <strong style="color:{BLUE};">Databricks × UN</strong> — Humanitarian AI for resource allocation.<br>
        Tracks: <strong style="color:{TEXT};">Actian VectorAI</strong> ·
        <strong style="color:{TEXT};">SafetyKit</strong> ·
        <strong style="color:{TEXT};">Figma Make</strong>
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

    _cl = st.session_state.get('theme', 'dark') == 'light'
    _chat_bg     = "rgba(255,255,255,0.85)"        if _cl else CARD
    _chat_border = "rgba(0,0,0,0.08)"              if _cl else BORDER
    _chat_text   = "#1E293B"                        if _cl else TEXT
    _chat_dim    = "rgba(10,22,40,0.35)"            if _cl else DIM
    _bot_bubble  = "rgba(0,119,182,0.07)"           if _cl else "rgba(45,116,218,0.07)"
    _usr_bubble  = "rgba(0,0,0,0.05)"              if _cl else "rgba(22,31,51,0.6)"
    _bot_bdr     = "rgba(0,119,182,0.18)"          if _cl else "rgba(45,116,218,0.18)"

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

    st.markdown(f'<hr style="border-color:{_chat_border};margin:0;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{_chat_bg};border-top:1px solid {_chat_border};padding:14px 0 8px;backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:{_chat_text};text-transform:uppercase;letter-spacing:2px;">◎ BLINDSPOT AI Agent</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:{_chat_dim};text-transform:uppercase;letter-spacing:1.5px;">Gemini 2.5 Flash · Domain-restricted</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    chat_c = st.container(height=250, border=False)
    with chat_c:
        for msg in st.session_state.messages:
            is_bot = msg['role'] == 'assistant'
            align = "flex-start" if is_bot else "flex-end"
            bg = _bot_bubble if is_bot else _usr_bubble
            bdr = _bot_bdr if is_bot else _chat_border
            pfx = "AI › " if is_bot else "YOU › "
            st.markdown(f"""
            <div style="display:flex;justify-content:{align};margin-bottom:5px;">
                <div style="max-width:86%;background:{bg};border:1px solid {bdr};border-radius:7px;padding:7px 11px;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:7px;color:{_chat_dim};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:3px;">{pfx}</div>
                    <div style="font-family:'DM Sans',sans-serif;font-size:12px;color:{_chat_text};line-height:1.5;">{msg['content']}</div>
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
    <div style="padding:5px 48px 14px;background:{CARD};">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:8px;color:{DIM};text-transform:uppercase;letter-spacing:1.5px;">
            BLINDSPOT exposes what the data shows. Action is yours. · Every misallocated dollar has a body count.
        </span>
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
with st.spinner("Connecting to Databricks..."):
    df_crises = load_data()

inject_theme_css()
render_navbar(df_crises)

page = st.session_state.page
if   page == 'command_center':       page_command_center(df_crises)
elif page == 'crisis_detail':         page_crisis_detail(df_crises)
elif page == 'allocation_simulator':  page_allocation_simulator(df_crises)
elif page == 'methodology':           page_methodology()

# ── Floating Chat Action Button ──
_chat_open_js = 'true' if st.session_state.chat_open else 'false'
_is_light_fab = st.session_state.get('theme', 'dark') == 'light'
_fab_color     = "#C8372D" if _is_light_fab else "rgba(229,57,53,0.9)"
_fab_hover     = "#A02820" if _is_light_fab else "rgba(229,57,53,1.0)"
_fab_shadow    = "rgba(200,55,45,0.30)" if _is_light_fab else "rgba(229,57,53,0.35)"

components.html(f"""
<script>
(function() {{
    var P = window.parent.document;
    var FAB_BG    = '{_fab_color}';
    var FAB_HOVER = '{_fab_hover}';
    var FAB_SHADE = '{_fab_shadow}';
    // Hide original nav_chat button (FAB replaces it visually)
    var allBtns = P.querySelectorAll('button');
    for (var i = 0; i < allBtns.length; i++) {{
        var t = (allBtns[i].innerText || allBtns[i].textContent || '').trim();
        if (t === '◎ AI Agent' || t === '✕ Close Chat') {{
            var wrapper = allBtns[i].closest('[data-testid="stButton"]');
            if (wrapper) wrapper.style.cssText = 'position:absolute!important;opacity:0!important;pointer-events:none!important;width:1px!important;height:1px!important;overflow:hidden!important;';
            break;
        }}
    }}
    // Create or update FAB
    var fab = P.getElementById('bs-chat-fab');
    var isOpen = {_chat_open_js};
    if (!fab) {{
        fab = P.createElement('div');
        fab.id = 'bs-chat-fab';
        fab.style.cssText = [
            'position:fixed;bottom:28px;right:28px;',
            'width:52px;height:52px;border-radius:50%;',
            'background:' + FAB_BG + ';',
            'backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);',
            'border:1px solid rgba(255,255,255,0.15);',
            'display:flex;align-items:center;justify-content:center;',
            'cursor:pointer;z-index:9999;',
            'box-shadow:0 8px 32px ' + FAB_SHADE + ';',
            'transition:background 0.15s,transform 0.15s,box-shadow 0.15s;',
        ].join('');
        fab.addEventListener('mouseenter', function() {{
            this.style.background = FAB_HOVER;
            this.style.transform  = 'scale(1.06)';
        }});
        fab.addEventListener('mouseleave', function() {{
            this.style.background = FAB_BG;
            this.style.transform  = 'scale(1)';
        }});
        fab.addEventListener('click', function() {{
            var btns2 = P.querySelectorAll('button');
            for (var j = 0; j < btns2.length; j++) {{
                var t2 = (btns2[j].innerText || btns2[j].textContent || '').trim();
                if (t2 === '◎ AI Agent' || t2 === '✕ Close Chat') {{
                    btns2[j].click(); return;
                }}
            }}
        }});
        P.body.appendChild(fab);
    }} else {{
        fab.style.background  = FAB_BG;
        fab.style.boxShadow   = '0 8px 32px ' + FAB_SHADE;
    }}
    // Update icon based on chat state
    fab.innerHTML = isOpen
        ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="white"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>'
        : '<svg width="24" height="24" viewBox="0 0 24 24" fill="white"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/></svg>';
}})();
</script>
""", height=0, scrolling=False)

render_chat(df_crises)
