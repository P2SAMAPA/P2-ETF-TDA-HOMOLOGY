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

# --- TDA Parameters ---
LOOKBACK_WINDOW = 252                 # Rolling window for correlation/return matrix
TDA_METHOD = "ripser"                 # "ripser" or "gudhi"
MAX_DIM = 2                           # Compute H0, H1, H2 persistent homology
N_LANDSCAPES = 5                      # Number of persistence landscapes to extract
MIN_OBSERVATIONS = 100                # Minimum observations for TDA

# --- Return Selection Parameters ---
RETURN_LOOKBACK_DAYS = 21             # Period for ranking ETFs within a style

# --- Early Warning Signals ---
BETTI_ALERT_THRESHOLD = 0.5           # Normalized Betti number change threshold
PERSISTENCE_ALERT_PERCENTILE = 90     # Alert when max persistence exceeds percentile

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
