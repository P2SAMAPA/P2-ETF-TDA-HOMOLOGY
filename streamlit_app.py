"""
Streamlit Dashboard for TDA Homology Engine.
Displays topological metrics and ETF selection signals.
"""

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
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .alert-card { background: #dc3545; border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .pick-card { background: #f8f9fa; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .signal-tag { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: bold; }
    .defensive { background: #28a745; color: white; }
    .momentum { background: #007bff; color: white; }
    .safe_haven { background: #ffc107; color: black; }
    .neutral { background: #6c757d; color: white; }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-mid { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .return-positive { color: #28a745; font-weight: 600; }
    .return-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def style_tag(style):
    return f'<span class="signal-tag {style}">{style.upper()}</span>'

def confidence_badge(conf):
    if conf >= 0.75:
        return f'<span class="confidence-high">{conf:.2f} (High)</span>'
    elif conf >= 0.5:
        return f'<span class="confidence-mid">{conf:.2f} (Moderate)</span>'
    else:
        return f'<span class="confidence-low">{conf:.2f} (Low)</span>'

def return_badge(ret):
    if ret >= 0:
        return f'<span class="return-positive">+{ret*100:.2f}%</span>'
    else:
        return f'<span class="return-negative">{ret*100:.2f}%</span>'

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
st.sidebar.divider()
st.sidebar.markdown("### 🧬 TDA Parameters")
st.sidebar.markdown(f"- Lookback: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- Max Homology Dim: **{config.MAX_DIM}**")
st.sidebar.markdown(f"- Return Lookback: **{config.RETURN_LOOKBACK_DAYS} days**")

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🧬 P2Quant TDA Homology Engine</div>', unsafe_allow_html=True)
st.markdown('<div>Persistent Homology – Market Structure & ETF Selection Signals</div>', unsafe_allow_html=True)

with st.expander("📘 How to Interpret TDA Signals", expanded=False):
    st.markdown("""
    ### Topological Metrics
    - **Betti‑1 Trend**: Rising → fragmentation (defensive). Falling → simplification (momentum).
    - **Max Persistence Spike**: Regime break → safe havens.

    ### Confidence Score
    | Confidence | Meaning | Action |
    |------------|---------|--------|
    | 0.80–1.00 | Strong signal | Higher conviction |
    | 0.50–0.79 | Moderate | Use as tilt |
    | 0.00–0.49 | Weak | Rely on other engines |
    
    Confidence of **0.50** = model uncertain; defer to other signals.
    
    ### Top Picks
    Selected from style‑specific candidates based on **{}-day return** (highest first).
    """.format(config.RETURN_LOOKBACK_DAYS))

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_tda']
global_signal = data.get('global_signal', {})
top_picks_all = daily.get('top_picks', {})
universes_data = daily.get('universes', {})

# --- Hero Section: Global Regime ---
st.markdown("## 🌐 Global Market Regime")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Regime", global_signal.get('regime', 'unknown').replace('_', ' ').title())
with col2:
    st.metric("Confidence", f"{global_signal.get('confidence', 0):.2f}")
with col3:
    style = global_signal.get('recommended_style', 'neutral')
    st.markdown(f"**Recommended Style:** {style_tag(style)}", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"## 📈 Top 3 ETF Picks by Universe (Ranked by {config.RETURN_LOOKBACK_DAYS}-Day Return)")

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        pick_data = top_picks_all.get(key, {})
        picks = pick_data.get('picks', [])
        reg = pick_data.get('regime', 'neutral')
        conf = pick_data.get('confidence', 0.5)
        style = pick_data.get('recommended_style', 'neutral')
        
        if picks:
            st.markdown(f"**Regime:** {reg.replace('_', ' ').title()} | **Confidence:** {confidence_badge(conf)} | **Style:** {style_tag(style)}")
            cols = st.columns(3)
            for i, pick in enumerate(picks[:3]):
                with cols[i]:
                    ret = pick['return_21d']
                    st.markdown(f"""
                    <div class="pick-card">
                        <h3>#{i+1} {pick['ticker']}</h3>
                        <p style="font-size: 1.5rem;">{return_badge(ret)}</p>
                        <p>{config.RETURN_LOOKBACK_DAYS}-day return</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No picks available for this universe.")

# --- Topological Metrics Table ---
st.markdown("---")
st.markdown("### 🔬 Topological Metrics by Universe")
rows = []
for uni, metrics in universes_data.items():
    rows.append({
        'Universe': uni,
        'Betti‑0': metrics['betti_numbers'][0],
        'Betti‑1': metrics['betti_numbers'][1],
        'Betti‑2': metrics['betti_numbers'][2] if len(metrics['betti_numbers'])>2 else 0,
        'Max Persistence': f"{metrics['max_persistence']:.4f}"
    })
df_metrics = pd.DataFrame(rows)
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# --- Shrinking Windows ---
if data.get('shrinking_windows'):
    st.markdown("---")
    st.markdown("### 📆 Historical Regime Evolution")
    shrink = data['shrinking_windows']
    df_shrink = pd.DataFrame(shrink).T
    st.dataframe(df_shrink[['regime', 'confidence', 'recommended_style']], use_container_width=True)
