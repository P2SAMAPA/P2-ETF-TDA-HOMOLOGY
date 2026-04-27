"""
Configuration for P2-ETF-TDA-HOMOLOGY engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-tda-homology-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Columns ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- TDA Parameters ---
LOOKBACK_WINDOW = 252
MAX_DIM = 2
MIN_OBSERVATIONS = 100
BETTI_ALERT_THRESHOLD = 0.5

# --- Return‑Seeking TDA Interpretation ---
# Regime factor: amplification for return ranking
REGIME_BOOST = {
    "simplification": 1.5,    # strong boost for momentum
    "neutral": 1.0,           # no adjustment
    "fragmentation": 0.7,     # reduce, but still positive returns are okay
    "regime_break": 0.3       # heavily reduce
}

# --- Daily and Global modes ---
DAILY_LOOKBACK = 504
GLOBAL_TRAIN_START = "2008-01-01"

# --- Shrinking Windows (fixed) ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
