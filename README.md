# P2-ETF-TDA-HOMOLOGY

**Topological Data Analysis for Market Structure & Regime Detection**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-TDA-HOMOLOGY/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-TDA-HOMOLOGY/actions/workflows/daily_run.yml)

## Overview
Uses persistent homology (Rips filtration) on rolling ETF return/correlation point clouds to extract topological features (Betti numbers, persistence diagrams). These features capture higher‑order market structure and provide early‑warning signals for regime shifts.

## Methodology
1. Construct point cloud from correlation distance matrix over rolling 252‑day window.
2. Compute persistent homology up to dimension 2 using Ripser.
3. Extract Betti‑0 (components), Betti‑1 (holes/loops), and max persistence.
4. Monitor changes in Betti‑1 and spikes in persistence as regime alerts.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)

## Dashboard
- Daily Betti numbers and persistence per universe
- Red alert cards when topological warning active
- Shrinking windows to see historical evolution

## Usage
```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
