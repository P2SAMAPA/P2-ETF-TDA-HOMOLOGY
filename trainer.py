"""
Main training script for TDA-HOMOLOGY engine.
Computes persistent homology and generates ETF selection signals.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from tda_model import TDAHomologyAnalyzer
import push_results

# Style-to-candidate ETFs mapping per universe
STYLE_CANDIDATES = {
    'defensive': {
        'FI_COMMODITIES': ['TLT', 'LQD', 'VCIT'],
        'EQUITY_SECTORS': ['XLP', 'XLU', 'XLV'],
        'COMBINED': ['XLP', 'XLU', 'TLT', 'LQD']
    },
    'momentum': {
        'FI_COMMODITIES': ['HYG', 'VNQ', 'GLD'],
        'EQUITY_SECTORS': ['SPY', 'QQQ', 'XLK', 'IWF'],
        'COMBINED': ['SPY', 'QQQ', 'HYG', 'IWF']
    },
    'safe_haven': {
        'FI_COMMODITIES': ['GLD', 'SLV', 'TLT'],
        'EQUITY_SECTORS': ['GLD', 'XLP', 'XLU'],
        'COMBINED': ['GLD', 'SLV', 'TLT']
    },
    'neutral': {
        'FI_COMMODITIES': ['LQD', 'VCIT', 'HYG'],
        'EQUITY_SECTORS': ['SPY', 'XLV', 'XLI'],
        'COMBINED': ['SPY', 'LQD', 'XLV']
    }
}

def compute_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Compute recent returns for ranking."""
    prices = df_wide.set_index('Date')[tickers]
    returns = prices.pct_change(config.RETURN_LOOKBACK_DAYS).iloc[-1]
    return returns

def select_top_etfs_by_return(universe: str, style: str, returns: pd.Series, n: int = 3) -> list:
    """Select top N ETFs from style candidates based on return."""
    candidates = STYLE_CANDIDATES.get(style, {}).get(universe, [])
    available = [t for t in candidates if t in returns.index]
    if not available:
        # fallback to all tickers in universe
        available = config.UNIVERSES[universe]
    sorted_etfs = returns[available].sort_values(ascending=False)
    top = sorted_etfs.head(n)
    return [{'ticker': t, 'return_21d': float(v)} for t, v in top.items()]

def run_tda_analysis():
    print(f"=== P2-ETF-TDA-HOMOLOGY Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    analyzer = TDAHomologyAnalyzer(max_dim=config.MAX_DIM, n_landscapes=config.N_LANDSCAPES)
    
    all_results = {}
    top_picks_all = {}
    alerts = {}
    
    # Combined universe for global TDA signal
    returns_combined = data_manager.prepare_returns_matrix(df_master, config.ALL_TICKERS)
    if len(returns_combined) >= config.MIN_OBSERVATIONS:
        recent_combined = returns_combined.iloc[-config.LOOKBACK_WINDOW:]
        analyzer.rolling_tda(returns_combined.iloc[-504:], window=config.LOOKBACK_WINDOW)
        global_signal = analyzer.compute_regime_signal()
    else:
        global_signal = {'regime': 'unknown', 'confidence': 0.0, 'recommended_style': 'neutral'}
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue
        recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]
        
        # Local TDA metrics
        point_cloud = analyzer.compute_point_cloud(recent_returns, method='correlation')
        pers = analyzer.compute_persistence(point_cloud, is_distance=True)
        
        # Compute 21-day returns for ranking
        returns_21d = compute_returns_matrix(df_master, tickers)
        
        # Select top 3 ETFs for the recommended style
        style = global_signal['recommended_style']
        top_picks = select_top_etfs_by_return(universe_name, style, returns_21d, n=3)
        
        all_results[universe_name] = {
            'betti_numbers': pers['betti_numbers'],
            'max_persistence': pers['max_persistence'],
            'signal': global_signal,
            'top_picks': top_picks
        }
        
        top_picks_all[universe_name] = {
            'regime': global_signal['regime'],
            'confidence': global_signal['confidence'],
            'recommended_style': style,
            'picks': top_picks
        }
        
        if global_signal['regime'] in ['fragmentation', 'regime_break']:
            alerts[universe_name] = top_picks_all[universe_name]
    
    # Shrinking windows
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue
        win_returns = data_manager.prepare_returns_matrix(df_window, config.ALL_TICKERS)
        if len(win_returns) < config.MIN_OBSERVATIONS:
            continue
        win_analyzer = TDAHomologyAnalyzer(max_dim=config.MAX_DIM)
        win_analyzer.rolling_tda(win_returns.iloc[-config.LOOKBACK_WINDOW:], window=config.LOOKBACK_WINDOW)
        win_signal = win_analyzer.compute_regime_signal()
        shrinking_results[window_label] = {
            'start_year': start_year,
            'regime': win_signal['regime'],
            'confidence': win_signal['confidence'],
            'recommended_style': win_signal['recommended_style']
        }
    
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "max_dim": config.MAX_DIM,
            "return_lookback_days": config.RETURN_LOOKBACK_DAYS
        },
        "global_signal": global_signal,
        "daily_tda": {
            "universes": all_results,
            "top_picks": top_picks_all,
            "alerts": alerts
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_tda_analysis()
