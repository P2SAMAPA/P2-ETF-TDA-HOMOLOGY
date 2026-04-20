import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant TDA Homology", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .alert-card { background: #dc3545; border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .normal-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files: return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f: return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
st.sidebar.divider()
st.sidebar.markdown("### 🧬 TDA Parameters")
st.sidebar.markdown(f"- Lookback: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- Max Homology Dim: **{config.MAX_DIM}**")

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🧬 P2Quant TDA Homology Engine</div>', unsafe_allow_html=True)
st.markdown('<div>Persistent Homology on ETF Return/Correlation Clouds – Early Regime Warnings</div>', unsafe_allow_html=True)

with st.expander("📘 How to Interpret TDA Metrics", expanded=False):
    st.markdown("""
    - **Betti‑0**: Number of connected components.
    - **Betti‑1**: Number of 1‑dimensional holes (loops) in the data cloud — **key indicator of market structure complexity**.
    - **Betti‑2**: Number of 2‑dimensional voids (rare).
    - **Max Persistence**: Lifetime of the most persistent topological feature. Spikes often precede regime breaks.
    - **Warning**: Triggered when Betti‑1 changes rapidly OR max persistence exceeds 2σ.
    """)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_tda']
alerts = daily.get('top_alerts', {})

tab1, tab2 = st.tabs(["📋 Daily TDA Dashboard", "📆 Shrinking Windows"])

with tab1:
    if alerts:
        st.markdown("### 🚨 Active Topological Warnings")
        for uni, alert in alerts.items():
            st.markdown(f"""
            <div class="alert-card">
                <h2>{uni}</h2>
                <p>Betti‑1: {alert['betti_1']} | Max Persistence: {alert['max_persistence']:.4f}</p>
                <p>{alert['warning']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No topological warnings detected. Market structure appears stable.")
    
    st.markdown("### Universe TDA Metrics")
    cols = st.columns(3)
    for i, (uni, metrics) in enumerate(daily['universes'].items()):
        with cols[i % 3]:
            alert = metrics['tda_warning']
            card_class = "alert-card" if alert else "normal-card"
            st.markdown(f"""
            <div class="{card_class}">
                <h3>{uni}</h3>
                <p>Betti: {metrics['betti_numbers']}</p>
                <p>Max Persistence: {metrics['max_persistence']:.4f}</p>
                <p>{'⚠️ Warning' if alert else '✅ Stable'}</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### Betti‑1 Evolution Across Historical Windows")
    shrinking = data.get('shrinking_windows', {})
    if shrinking:
        df = pd.DataFrame(shrinking).T
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['betti_1'], mode='lines+markers', name='Betti‑1'))
        fig.add_trace(go.Scatter(x=df.index, y=df['max_persistence'], mode='lines+markers', name='Max Persistence', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'), height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
    else:
        st.info("No shrinking windows data.")
