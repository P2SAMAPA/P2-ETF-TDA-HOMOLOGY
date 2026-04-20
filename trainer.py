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

# Style-to-ETF mapping per universe
STYLE_ETF_MAP = {
    'defensive': {
        'FI_COMMODITIES': 'TLT',
        'EQUITY_SECTORS': 'XLP',
        'COMBINED': 'XLP'
    },
    'momentum': {
        'FI_COMMODITIES': 'HYG',
        'EQUITY_SECTORS': 'SPY',
        'COMBINED': 'SPY'
    },
    'safe_haven': {
        'FI_COMMODITIES': 'GLD',
        'EQUITY_SECTORS': 'GLD',
        'COMBINED': 'GLD'
    },
    'neutral': {
        'FI_COMMODITIES': 'LQD',
        'EQUITY_SECTORS': 'SPY',
        'COMBINED': 'SPY'
    }
}

def select_etf_from_signal(signal: dict, universe: str) -> str:
    style = signal.get('recommended_style', 'neutral')
    return STYLE_ETF_MAP.get(style, {}).get(universe, config.ALL_TICKERS[0])

def run_tda_analysis():
    print(f"=== P2-ETF-TDA-HOMOLOGY Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    analyzer = TDAHomologyAnalyzer(max_dim=config.MAX_DIM, n_landscapes=config.N_LANDSCAPES)
    
    all_results = {}
    top_picks = {}
    alerts = {}
    
    # Combined universe for global TDA (used for signal)
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
        
        # Local TDA for this universe
        point_cloud = analyzer.compute_point_cloud(recent_returns, method='correlation')
        pers = analyzer.compute_persistence(point_cloud, is_distance=True)
        
        # Determine top pick based on global signal (or local if you prefer)
        top_etf = select_etf_from_signal(global_signal, universe_name)
        
        all_results[universe_name] = {
            'betti_numbers': pers['betti_numbers'],
            'max_persistence': pers['max_persistence'],
            'signal': global_signal,
            'top_pick': top_etf
        }
        
        top_picks[universe_name] = {
            'ticker': top_etf,
            'regime': global_signal['regime'],
            'confidence': global_signal['confidence'],
            'recommended_style': global_signal['recommended_style']
        }
        
        if global_signal['regime'] in ['fragmentation', 'regime_break']:
            alerts[universe_name] = top_picks[universe_name]
    
    # Shrinking windows (simplified)
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
            "max_dim": config.MAX_DIM
        },
        "global_signal": global_signal,
        "daily_tda": {
            "universes": all_results,
            "top_picks": top_picks,
            "alerts": alerts
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_tda_analysis()
