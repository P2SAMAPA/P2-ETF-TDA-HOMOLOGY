"""
Main training script for TDA-HOMOLOGY engine.
Computes persistent homology on ETF universes and generates early-warning signals.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from tda_model import TDAHomologyAnalyzer
import push_results

def run_tda_analysis():
    print(f"=== P2-ETF-TDA-HOMOLOGY Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    analyzer = TDAHomologyAnalyzer(max_dim=config.MAX_DIM, n_landscapes=config.N_LANDSCAPES)
    
    all_results = {}
    top_alerts = {}
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue
        recent_returns = returns.iloc[-config.LOOKBACK_WINDOW:]
        
        # Single window TDA for today
        point_cloud = analyzer.compute_point_cloud(recent_returns, method='correlation')
        pers = analyzer.compute_persistence(point_cloud)
        
        # Rolling TDA for history (last 2 years)
        tda_history = analyzer.rolling_tda(returns.iloc[-504:], window=config.LOOKBACK_WINDOW)
        tda_warning = analyzer.compute_early_warning(tda_history)
        
        # Alert if Betti-1 changes significantly or persistence spikes
        latest = tda_warning.iloc[-1] if len(tda_warning) > 0 else pd.Series()
        alert = False
        if len(latest) > 0:
            betti_change = latest.get('betti_1_change', 0)
            pers_z = latest.get('persistence_z', 0)
            if betti_change > config.BETTI_ALERT_THRESHOLD or pers_z > 2.0:
                alert = True
        
        all_results[universe_name] = {
            'betti_numbers': pers['betti_numbers'],
            'max_persistence': pers['max_persistence'],
            'tda_warning': alert,
            'latest_metrics': latest.to_dict() if len(latest) > 0 else {},
            'history': tda_history.reset_index().to_dict(orient='list') if len(tda_history) > 0 else {}
        }
        
        if alert:
            top_alerts[universe_name] = {
                'universe': universe_name,
                'betti_1': pers['betti_numbers'][1],
                'max_persistence': pers['max_persistence'],
                'warning': 'Regime shift likely — topological structure changing'
            }
    
    # Shrinking windows (simplified: only store top alert per window)
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue
        window_returns = data_manager.prepare_returns_matrix(df_window, config.ALL_TICKERS)
        if len(window_returns) < config.MIN_OBSERVATIONS:
            continue
        pc = analyzer.compute_point_cloud(window_returns.iloc[-config.LOOKBACK_WINDOW:], method='correlation')
        pers = analyzer.compute_persistence(pc)
        shrinking_results[window_label] = {
            'start_year': start_year,
            'betti_1': pers['betti_numbers'][1],
            'max_persistence': pers['max_persistence']
        }
    
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "lookback_window": config.LOOKBACK_WINDOW,
            "max_dim": config.MAX_DIM,
            "betti_alert_threshold": config.BETTI_ALERT_THRESHOLD
        },
        "daily_tda": {
            "universes": all_results,
            "top_alerts": top_alerts
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_tda_analysis()
