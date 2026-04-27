"""
Main training script – Daily, Global, and Shrinking Windows with Consensus.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from tda_model import TDAHomologyAnalyzer
import push_results


def select_etfs_by_return(returns, tickers, n=3, boost_factor=1.0):
    """Rank ETFs by 21‑day return multiplied by regime boost."""
    ret_21d = returns.iloc[-21:].mean() * 252
    scores = ret_21d * boost_factor
    sorted_tickers = scores.sort_values(ascending=False)
    top = [{'ticker': t, 'return_21d': float(ret_21d[t]), 'adjusted_score': float(scores[t])}
           for t in sorted_tickers.head(n).index]
    return top


def run_mode(returns, mode_name, macro_df=None):
    """Run TDA on a data slice and return regime, top picks, table."""
    if len(returns) < config.MIN_OBSERVATIONS:
        return None

    analyzer = TDAHomologyAnalyzer(max_dim=config.MAX_DIM)
    analyzer.rolling_tda(returns.iloc[-config.LOOKBACK_WINDOW:], window=config.LOOKBACK_WINDOW)
    regime_info = analyzer.compute_regime()

    tickers = [c for c in returns.columns if c in config.ALL_TICKERS]
    boost = regime_info['boost_factor']
    top3 = select_etfs_by_return(returns[tickers], tickers, n=3, boost_factor=boost)

    # Full table
    ret_21d = returns[tickers].iloc[-21:].mean() * 252
    scores = ret_21d * boost
    table = [{'ticker': t, 'return_21d': float(ret_21d[t]), 'adjusted_score': float(scores[t])}
             for t in tickers]

    return {
        'regime': regime_info['regime'],
        'confidence': regime_info['confidence'],
        'boost_factor': boost,
        'top_picks': top3,
        'all_scores': table,
        'training_start': str(returns.index[0].date()),
        'training_end': str(returns.index[-1].date())
    }


def run_shrinking_windows(df_master, tickers, macro_df=None):
    """Fixed shrinking windows with consensus on top ETF."""
    windows = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        sd = pd.Timestamp(f"{start_year}-01-01")
        ed = pd.Timestamp(f"{start_year+2}-12-31")
        mask = (df_master['Date'] >= sd) & (df_master['Date'] <= ed)
        window_df = df_master[mask].copy()
        if len(window_df) < config.MIN_OBSERVATIONS:
            continue
        returns = data_manager.prepare_returns_matrix(window_df, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        analyzer = TDAHomologyAnalyzer(max_dim=config.MAX_DIM)
        analyzer.rolling_tda(returns.iloc[-config.LOOKBACK_WINDOW:], window=config.LOOKBACK_WINDOW)
        regime_info = analyzer.compute_regime()
        boost = regime_info['boost_factor']
        top = select_etfs_by_return(returns, tickers, n=1, boost_factor=boost)
        windows.append({
            'window_start': start_year,
            'window_end': start_year+2,
            'ticker': top[0]['ticker'] if top else 'N/A',
            'regime': regime_info['regime'],
            'boost_factor': boost
        })

    if not windows:
        return None

    # Consensus
    vote = {}
    for w in windows:
        vote[w['ticker']] = vote.get(w['ticker'], 0) + 1
    pick = max(vote, key=vote.get)
    conviction = vote[pick] / len(windows) * 100
    return {'ticker': pick, 'conviction': conviction, 'num_windows': len(windows), 'windows': windows}


def main():
    import os
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_master = data_manager.load_master_data()
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns_all = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_all) < config.MIN_OBSERVATIONS:
            continue

        universe_out = {}

        # Daily
        daily_ret = returns_all.iloc[-config.DAILY_LOOKBACK:]
        daily_out = run_mode(daily_ret, 'Daily', macro.loc[daily_ret.index])
        if daily_out:
            universe_out['daily'] = daily_out
            print(f"  Daily top: {daily_out['top_picks'][0]['ticker']}")

        # Global
        global_ret = returns_all.loc[returns_all.index >= config.GLOBAL_TRAIN_START]
        global_out = run_mode(global_ret, 'Global', macro.loc[global_ret.index])
        if global_out:
            universe_out['global'] = global_out
            print(f"  Global top: {global_out['top_picks'][0]['ticker']}")

        # Shrinking
        shrinking = run_shrinking_windows(df_master, tickers, macro)
        if shrinking:
            universe_out['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
