# P2-ETF-TDA-HOMOLOGY

**Topological Data Analysis – Return‑Chasing Regime Signals for ETF Selection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-TDA-HOMOLOGY/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-TDA-HOMOLOGY/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--tda--homology--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-tda-homology-results)

## Overview

`P2-ETF-TDA-HOMOLOGY` computes persistent homology (Betti numbers, max persistence) on rolling ETF correlation matrices to detect market regime shifts. A **return‑chasing scoring** amplifies recent 21‑day returns based on the current topological regime:

- **Simplification (Betti‑1 falling)** → strong boost (×1.5) — favors momentum.
- **Neutral** → no adjustment (×1.0) — pure return ranking.
- **Fragmentation (Betti‑1 rising)** → reduced boost (×0.7).
- **Regime Break (persistence spike)** → heavily reduced boost (×0.3).

ETFs are ranked by their **adjusted score** (21‑day return × regime boost).

## Methodology

1. **Rolling TDA** – Persistent homology on 252‑day correlation windows.
2. **Regime Classification** – Betti‑1 trend and max persistence z‑score.
3. **Return‑Chasing Scoring** – `Score = 21‑day annualised return × regime_boost`.
4. **Three Training Modes** – Daily, Global, Shrinking Windows Consensus.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
