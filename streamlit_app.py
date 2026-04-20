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
    .signal-tag { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: bold; }
    .defensive { background: #28a745; color: white; }
    .momentum { background: #007bff; color: white; }
    .safe_haven { background: #ffc107; color: black; }
    .neutral { background: #6c757d; color: white; }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-mid { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
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
st.markdown('<div>Persistent Homology – Market Structure & ETF Selection Signals</div>', unsafe_allow_html=True)

with st.expander("📘 How to Interpret TDA Signals", expanded=False):
    st.markdown("""
    ### Topological Metrics
    - **Betti‑0**: Number of connected components in the data cloud.
    - **Betti‑1**: Number of 1‑dimensional holes (loops) — **key indicator of market complexity**.  
      *Rising Betti‑1* → market fragmentation / stress.  
      *Falling Betti‑1* → structure simplifying, trends emerging.
    - **Max Persistence**: Lifetime of the most persistent topological feature.  
      *Spikes* often precede regime breaks (e.g., 2008, 2020).

    ### Confidence Score Interpretation
    | Confidence | Meaning | Recommended Action |
    |------------|---------|---------------------|
    | **0.80 – 1.00** | Strong topological signal | Higher conviction in the recommended ETF |
    | **0.50 – 0.79** | Moderate signal | Use as a tilt; confirm with other engines (BSTS, HRP) |
    | **0.00 – 0.49** | Weak / no clear edge | Default to neutral or rely on other signals |
    
    A confidence of **0.50** is the midpoint — the model is uncertain and suggests using other engines.
    """)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_tda']
global_signal = data.get('global_signal', {})
top_picks = daily.get('top_picks', {})
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
st.markdown("## 📈 Top 3 ETF Picks by Universe")

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

# Generate top 3 picks per universe based on signal confidence and style
for tab, key in zip(tabs, universe_keys):
    with tab:
        universe_data = universes_data.get(key, {})
        signal_info = universe_data.get('signal', {})
        top_pick = top_picks.get(key, {})
        
        # For now, we only have one top pick; we'll display it prominently
        # and note that additional picks would come from style alternatives
        if top_pick:
            card_class = "alert-card" if key in daily.get('alerts', {}) else "hero-card"
            st.markdown(f"""
            <div class="{card_class}">
                <h2>🥇 {top_pick['ticker']}</h2>
                <p>Regime: {top_pick['regime'].replace('_', ' ').title()} | Confidence: {confidence_badge(top_pick['confidence'])}</p>
                <p>Style: {style_tag(top_pick['recommended_style'])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show second and third picks based on alternative styles
            st.markdown("### Alternative Picks (Other Styles)")
            alt_styles = []
            if top_pick['recommended_style'] != 'defensive':
                alt_styles.append(('Defensive', 'defensive', 0.6))
            if top_pick['recommended_style'] != 'momentum':
                alt_styles.append(('Momentum', 'momentum', 0.6))
            if top_pick['recommended_style'] != 'safe_haven':
                alt_styles.append(('Safe Haven', 'safe_haven', 0.5))
            
            # Map styles to tickers for this universe
            style_map = {
                'defensive': {'FI_COMMODITIES': 'TLT', 'EQUITY_SECTORS': 'XLP', 'COMBINED': 'XLP'},
                'momentum': {'FI_COMMODITIES': 'HYG', 'EQUITY_SECTORS': 'SPY', 'COMBINED': 'SPY'},
                'safe_haven': {'FI_COMMODITIES': 'GLD', 'EQUITY_SECTORS': 'GLD', 'COMBINED': 'GLD'}
            }
            
            cols = st.columns(len(alt_styles))
            for i, (label, style, base_conf) in enumerate(alt_styles[:2]):  # show up to 2 alternatives
                ticker = style_map.get(style, {}).get(key, 'N/A')
                with cols[i]:
                    st.markdown(f"""
                    <div style="background: #f8f9fa; border-radius: 12px; padding: 1rem; text-align: center;">
                        <h3>{label}</h3>
                        <h4>{ticker}</h4>
                        <p>Style: {style_tag(style)}</p>
                        <p>Confidence: {confidence_badge(base_conf)} (estimated)</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No recommendation available.")

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
