"""
Streamlit Dashboard for TDA Homology Engine (Return‑Chasing).
"""

import streamlit as st
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant TDA Homology", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .regime-simplification { background: #28a745; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.9rem; }
    .regime-fragmentation { background: #dc3545; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; }
    .regime-regime_break { background: #ffc107; color: black; padding: 0.2rem 0.8rem; border-radius: 20px; }
    .regime-neutral { background: #6c757d; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; }
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

def regime_badge(regime):
    badge_class = f"regime-{regime}"
    return f'<span class="{badge_class}">{regime.replace("_"," ").title()}</span>'

def regime_text(regime):
    """Plain text regime name for table columns."""
    return regime.replace("_", " ").title()

def render_mode_tab(mode_data, mode_name):
    if not mode_data:
        st.warning(f"No {mode_name} data.")
        return
    regime = mode_data.get('regime', 'neutral')
    confidence = mode_data.get('confidence', 0)
    boost = mode_data.get('boost_factor', 1.0)

    st.markdown(f"### {mode_name}")
    st.markdown(f"Regime: {regime_badge(regime)} | Confidence: {confidence:.2f} | Boost: {boost:.2f}", unsafe_allow_html=True)

    top = mode_data.get('top_picks', [])
    if top:
        pick = top[0]
        st.markdown(f"""
        <div class="hero-card">
            <div style="font-size: 1.2rem; opacity: 0.8;">🧬 TOP PICK (Return‑Chasing TDA)</div>
            <div class="hero-ticker">{pick['ticker']}</div>
            <div>Adj Return: {pick['adjusted_score']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Top 3 Picks")
        rows = [{"Ticker": p['ticker'], "Adj Return": f"{p['adjusted_score']:.4f}",
                 "Return (21d)": f"{p['return_21d']*100:.2f}%"} for p in top]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        all_scores = mode_data.get('all_scores', [])
        if all_scores:
            st.markdown("### All ETFs")
            df_all = pd.DataFrame(all_scores)
            df_all['Return (21d)'] = df_all['return_21d'].apply(lambda x: f"{x*100:.2f}%")
            df_all['Adj Return'] = df_all['adjusted_score'].apply(lambda x: f"{x:.4f}")
            df_all = df_all[['ticker', 'Return (21d)', 'Adj Return']].sort_values('Adj Return', ascending=False)
            st.dataframe(df_all, use_container_width=True, hide_index=True)

def render_shrinking_tab(shrinking_data):
    if not shrinking_data:
        st.warning("No shrinking data.")
        return
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🔄 SHRINKING CONSENSUS</div>
        <div class="hero-ticker">{shrinking_data['ticker']}</div>
        <div>{shrinking_data['conviction']:.0f}% conviction · {shrinking_data['num_windows']} windows</div>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 All Windows"):
        rows = []
        for w in shrinking_data.get('windows', []):
            rows.append({
                'Window': f"{w['window_start']}-{w['window_end']}",
                'Top Pick': w['ticker'],
                'Regime': regime_text(w.get('regime', 'neutral'))
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🧬 P2Quant TDA Homology</div>', unsafe_allow_html=True)
st.markdown('<div>Persistent Homology – Return‑Chasing Topological Regime Signals</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

universes_data = data.get('universes', {})
tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, keys):
    uni = universes_data.get(key, {})
    if not uni:
        with tab:
            st.info(f"No data for {key}.")
        continue
    with tab:
        d, g, s = st.tabs(["📅 Daily (504d)", "🌍 Global (2008‑YTD)", "🔄 Shrinking Consensus"])
        with d:
            render_mode_tab(uni.get('daily'), "Daily")
        with g:
            render_mode_tab(uni.get('global'), "Global")
        with s:
            render_shrinking_tab(uni.get('shrinking'))
