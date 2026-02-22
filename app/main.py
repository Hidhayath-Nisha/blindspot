import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from models.allocation_optimizer import run_allocation_optimizer
    from app.genai_briefs import generate_safety_brief_prompts
    from vector_rag.actian_search import ActianVectorDB
except Exception as e:
    st.error(f"Module import error: {e}")

st.set_page_config(page_title="TRIAGE: Humanitarian AI", page_icon="globe", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    .modebar {display: none !important;}

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem;
        max-width: 100% !important;
    }
    section[data-testid="stMain"] > div {
        padding-top: 0 !important;
    }

    .stApp {
        background: #F4F8FE !important;
    }
    
    p, span, div, h1, h2, h3, h4, h5, h6, label {
        color: #1A2C4E !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    
    .big-font {font-size:34px !important; font-weight: 800; color: #1565C0 !important;}
    .alert-font {font-size:28px !important; font-weight: 700; color: #BF360C !important;}
    
    .metric-card {
        background: #FFFFFF !important;
        border: 1px solid #D6E4F7 !important; 
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 2px 12px rgba(21, 101, 192, 0.07) !important; 
        transition: all 0.25s ease;
        text-align: center;
        margin-bottom: 16px;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 24px rgba(21, 101, 192, 0.13) !important;
    }

    [data-testid="stChatMessage"],
    [data-testid="stChatMessageContent"],
    .stChatMessage {
        background-color: #FFFFFF !important;
        color: #1A2C4E !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #EEF5FF !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span {
        color: #1A2C4E !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #D6E4F7 !important;
        border-radius: 12px !important;
    }
    .stTextInput input {
        background: #FFFFFF !important;
        color: #1A2C4E !important;
        border: 1px solid #B3CDEF !important;
        border-radius: 8px !important;
    }
    .stFormSubmitButton button {
        background: linear-gradient(135deg, #1565C0, #0D47A1) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #D6E4F7 !important;
        border-radius: 8px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(21, 101, 192, 0.3) !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(21, 101, 192, 0.5) !important;
        transform: translateY(-2px);
    }
    
    .stSlider > div > div > div > div {
        background: #1565C0 !important;
    }
    
    hr {
        border-color: #D6E4F7 !important;
        margin: 16px 0 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    df = pd.DataFrame()
    
    api_token = os.environ.get("DATABRICKS_TOKEN", "")
    hostname = os.environ.get("DATABRICKS_SERVER_HOSTNAME", "")
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
            pass
            
    if df.empty:
        optimized_path = os.path.join(parent_dir, "assets", "triage_master_optimized.csv")
        scores_path = os.path.join(parent_dir, "assets", "triage_master_scores.csv")
        
        if os.path.exists(optimized_path):
            df = pd.read_csv(optimized_path)
        elif os.path.exists(scores_path):
            df = pd.read_csv(scores_path)
        else:
            st.error("No triage scoring data found. Run Databricks pipelines first.")
            return pd.DataFrame({'iso3': [], 'Crisis_Severity_Score': [], 'Funding_Coverage_Ratio': []})
            
    fallback_geo = {
        'AFG': [33.9, 67.7], 'UKR': [48.3, 31.1], 'SDN': [12.8, 30.2], 
        'SYR': [34.8, 38.9], 'YEM': [15.5, 48.5], 'COD': [-4.0, 21.7],
        'ETH': [9.1, 40.4], 'HTI': [18.9, -72.2], 'SOM': [5.1, 46.1],
        'MMR': [21.9, 95.9], 'PSE': [31.9, 35.2], 'VEN': [6.4, -66.5],
        'ZAF': [-30.5, 22.9], 'SSD': [6.8, 31.3], 'NGA': [9.0, 8.6]
    }
    
    lats, lons = [], []
    for iso in df['iso3']:
        if iso in fallback_geo:
            lats.append(fallback_geo[iso][0])
            lons.append(fallback_geo[iso][1])
        else:
            lats.append(0.0)
            lons.append(0.0)
            
    df['Lat'] = lats
    df['Lon'] = lons
    df['Country_Name'] = df['iso3']
    
    if 'funding_required' in df.columns:
        df['Funding_Ratio'] = np.where(df['funding_required'] > 0, 
                                      np.round((df['funding_received'] / df['funding_required']) * 100, 1), 
                                      100)
    else:
        df['Funding_Ratio'] = df['Funding_Coverage_Ratio'] * 100
        
    return df


with st.spinner("Connecting to Databricks Unity SQL..."):
    df_crises = load_data()

header_img_path = os.path.join(parent_dir, "assets", "header.webp")
if os.path.exists(header_img_path):
    st.image(header_img_path, use_container_width=True)

st.markdown("""
<div style='
    background: linear-gradient(90deg, #0D47A1 0%, #1565C0 40%, #1976D2 100%);
    padding: 16px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
'>
    <div>
        <div style='font-size:28px; font-weight:900; color:#fff; letter-spacing:-1px; font-family:Segoe UI,Inter,sans-serif;'>TRIAGE</div>
        <div style='font-size:10px; color:rgba(255,255,255,0.80); letter-spacing:5px; font-weight:600; text-transform:uppercase; margin-top:2px;'>Global AI Capital Allocation Engine</div>
    </div>
    <div style='font-size:12px; color:rgba(255,255,255,0.75); text-align:right;'>
        Powered by Databricks &nbsp;|&nbsp; Live Data
    </div>
</div>
""", unsafe_allow_html=True)


top_left, top_right = st.columns([2.5, 1])

with top_left:
    df_map = df_crises.copy()
    df_map['Size'] = df_map['Crisis_Severity_Score'] * 2.5 
    
    fig = px.scatter_geo(
        df_map,
        lat="Lat",
        lon="Lon",
        hover_name="Country_Name",
        hover_data={"Crisis_Severity_Score": True, "Funding_Ratio": True, "Size": False, "Lat": False, "Lon": False},
        size="Size",
        color="Funding_Ratio",
        color_continuous_scale=px.colors.sequential.Blues[::-1], 
        projection="orthographic", 
        template="plotly_white"
    )
    
    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}, 
        height=550, 
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            bgcolor="rgba(232, 244, 253, 0.3)",
            showocean=True, 
            oceancolor="rgba(173, 216, 255, 0.5)", 
            showland=True, 
            landcolor="rgba(220, 230, 242, 0.8)",  
            showcountries=True, 
            countrycolor="rgba(100, 149, 200, 0.4)"
        ),
        coloraxis_colorbar=dict(title="Funding %", dtick=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with top_right:
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0; font-weight:400; color:#5A7299 !important;">Active Tracked Crises</h4>
        <p class="big-font" style="margin:0;">{len(df_crises)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    avg_severity = df_crises['Crisis_Severity_Score'].mean() if not df_crises.empty else 0
        
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0; font-weight:400; color:#5A7299 !important;">Average Global Severity</h4>
        <p class="alert-font" style="margin:0;">{avg_severity:.1f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    most_severe = df_crises.loc[df_crises['Crisis_Severity_Score'].idxmax()] if not df_crises.empty else None
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0; font-weight:400; color:#5A7299 !important;">Highest Severity Zone</h4>
        <p class="alert-font" style="margin:0; font-size:24px !important;">{most_severe['Country_Name']}</p>
        <p style="margin:0; color:#5A7299;">Score: {most_severe['Crisis_Severity_Score']}</p>
    </div>
    """, unsafe_allow_html=True)


bot_left, bot_right = st.columns([1, 1])

with bot_left:
    st.markdown("<h3 style='margin-bottom: 5px; color: #1A2C4E;'>AI Capital Allocation Engine</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #5A7299; margin-bottom: 20px;'>Enter a deployment budget and see exactly which crisis zones the algorithm would fund first, ranked by severity.</p>", unsafe_allow_html=True)

    opt_col1, opt_col2 = st.columns([2, 1])
    with opt_col1:
        donor_budget = st.slider("Deployment Capital (USD Millions)", min_value=10, max_value=500, value=100, step=10)
    with opt_col2:
        st.write("")
        run_sim = st.button("Know where to fund!", use_container_width=True)

    if run_sim:
        st.divider()
        st.markdown("<h4 style='color:#1565C0; margin-bottom:15px; margin-top:8px;'>Top Priority Allocations (Severity-Weighted)</h4>", unsafe_allow_html=True)
        
        total_budget = donor_budget * 1e6
        df_sorted = df_crises.sort_values(by="Crisis_Severity_Score", ascending=False).head(10)
        total_severity_top10 = df_sorted['Crisis_Severity_Score'].sum()
        allocations = []
        
        for idx, row in df_sorted.iterrows():
            proportion = row['Crisis_Severity_Score'] / total_severity_top10
            allocation = total_budget * proportion
            allocations.append({
                "Crisis Zone": row['Country_Name'],
                "Severity Score": round(row['Crisis_Severity_Score'], 2),
                "Recommended Allocation": f"${allocation / 1e6:.1f}M"
            })
            
        if allocations:
            df_alloc = pd.DataFrame(allocations)
            st.dataframe(df_alloc, use_container_width=True, hide_index=True)
            
        st.markdown(f"<p style='font-size:13px; color:#5A7299; margin-top:10px;'>${donor_budget}M distributed proportionally across the 10 highest severity zones. Based entirely on factual severity score data.</p>", unsafe_allow_html=True)


with bot_right:
    st.markdown("<h3 style='margin-bottom: 8px; color: #1A2C4E;'>Intelligence Assistant</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #5A7299; font-size:13px; margin-bottom:12px;'>Ask questions about the crisis data, funding priorities, or geopolitical context.</p>", unsafe_allow_html=True)
    
    bot_icon_path = os.path.join(parent_dir, "assets", "bot-icon.jpg")
    if not os.path.exists(bot_icon_path):
        bot_icon_path = None
    
    chat_container = st.container(height=310, border=False)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": f"Connected to Databricks cluster. {most_severe['Country_Name']} is the highest severity zone (Score: {most_severe['Crisis_Severity_Score']}). How can I assist with your deployment analysis?"})

    with chat_container:
        for message in st.session_state.messages:
            avatar_to_use = bot_icon_path if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=avatar_to_use):
                st.markdown(message["content"])

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            prompt = st.text_input("Ask an analytical question...", label_visibility="collapsed")
        with col2:
            submit = st.form_submit_button("Send")


if submit and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        response_text = "**Error:** No `GEMINI_API_KEY` found in the `.env` file."
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()
    else:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            df_summary = df_crises[['iso3', 'Crisis_Severity_Score', 'funding_required', 'funding_received']].head(5).to_string()
            
            sys_prompt = f"""You are TRIAGE AI, an analytical assistant for a Sovereign Wealth fund. 
Strict rules:
1. ONLY answer questions related to the humanitarian crisis map, capital allocation, geopolitical risk, and the data provided.
2. Do not generate code, stories, or answer general knowledge questions.
3. If unrelated: "System Alert: I am authorized only to analyze humanitarian triage and capital allocation data."
4. Be professional, succinct, and quantitative. No emojis.

Current Context:
Most severe crisis: {most_severe['Country_Name']} (Severity: {most_severe['Crisis_Severity_Score']})
Data Summary:
{df_summary}
"""
            response = client.models.generate_content(
                model="gemini-2.5-flash", 
                contents=f"{sys_prompt}\n\nUser Question: {prompt}"
            )
            
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            st.rerun()
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                error_msg = "**System Alert: API Quota Exhausted.** The Gemini API daily limit has been reached. The Databricks allocation engine continues to operate."
            else:
                error_msg = f"**System Alert:** {error_str}"
                
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.rerun()
