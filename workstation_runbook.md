# High-Growth Screener: Operator Manual & Runbook
**Target Audience**: Human Traders, Portfolio Managers, and AI Coding Assistants  
**Location**: `d:\Projects\early screener\workstation_runbook.md`  

---

## 1. System Objective & Architecture

This system is a quantitative stock discovery workstation designed for the Indian equity market (NSE/BSE). It isolates high-relative-strength breakout candidates (momentum markup) and early-stage microcap turnarounds (governance-backed accumulation).

### Component Map
* **Database (`database/screener.db`)**:
  * `stocks`: Tickers, sectors, and fundamentals (MCAP, ROCE, Promoter %, CFO).
  * `price_history`: Daily OHLCV candles (5 years lookback).
  * `technical_features`: Rolling averages, VCP tightness, and Pivot High prices.
  * `multibagger_dna`: Snapshotted technical states of 279,846 historical breakouts.
* **Ingestion Pipeline**:
  * `pipeline/build_db.py`: Fetches price history candles from Upstox.
  * `pipeline/fetch_fundamentals.py`: Scrapes fundamentals from Screener.in (fallback to yfinance).
* **Quant Engines**:
  * `engine/feature_engineer.py`: Generates Bollinger/SMA VCP calculations and Pivot Highs.
  * `engine/similarity_engine.py`: Matches setups against historical DNA using 3D Cosine Similarity.
  * `engine/breakout_model.py`: Trains an XGBoost model on path-dependent labels.
  * `engine/backtest_ledger.py`: Backtests the strategy with chronological cooldown rules.

---

## 2. Daily Operational Runbook

Follow these commands to keep the workstation operational:

### Step 1: End-of-Day Data Ingestion
Run this command daily after market close (5:00 PM IST or later) to update prices and scrape fundamentals:
```bash
python pipeline/build_db.py
```
*Note: If the Screener.in scraper is rate-limited, it automatically falls back to Yahoo Finance.*

### Step 2: Model Maintenance (Optional)
The XGBoost model does not need daily training. Retrain only if you change the target parameter `BREAKOUT_LABEL_THRESHOLD` in `config.py`:
```bash
python engine/breakout_model.py
```

### Step 3: Launching the Dashboard
Launch the Streamlit visual terminal to review candidates and backtest metrics:
```bash
streamlit run app.py
```

---

## 3. Strategic Execution Blueprint (For Humans)

To trade this strategy with positive mathematical expectancy, you must follow these rules:

```
                  [Nifty 50 Monthly 10 EMA Check]
                                 │
                   ┌─────────────┴─────────────┐
             🟢 Above EMA                 🟡 Below EMA
             (Bull Market)                (Bear Market)
                   │                           │
          Position Size: 1.0%         Position Size: 0.5%
          (Max Risk per trade)        (Sized down by 50%)
```

### 1. Selection & Confluence
* **Confluence Threshold**: Target candidates with a **Confluence Score $\ge 70\%$**.
* **Sector Clustering**: Prioritize setups inside sectors displaying concurrent breakouts.

### 2. The Execution (Pivot High Entry)
* **Rule**: **Never buy at yesterday's close price or at the market open.**
* **Action**: Identify the **Pivot High Price** printed by the tool. Place a **Buy Limit GTT (Good Till Triggered) Order** exactly at that price. This prevents entering consolidations that fail to break out.

### 3. Sizing & Exits (Standard Mode vs. Microcap Mode)
* **Standard Mode (Midcaps)**:
  * Initial Stop-Loss: **-7.5%**
  * Target Profit (50% position): **+15.0%** (2:1 Risk-Reward Ratio)
* **Microcap Mode (Turnarounds)**:
  * Initial Stop-Loss: **-15.0%** (wider for microcap volatility)
  * Target Profit (50% position): **+15.0%**
* **The Runner (Remaining 50%)**: Once the first half is sold at +15%, move the stop-loss to entry price (breakeven). Exit ONLY when the daily candle closes below the **50-day Moving Average**.

---

## 4. Time Horizon Configuration Guide (1-Day vs. Swing vs. Long-Term)

The workstation can be configured to target three distinct trading timeframes. Adjust your parameters and execution rules as follows:

### 1-Day Holding Period (Intraday Scalping)
* **Goal**: Capture quick 2% to 4% momentum spikes and close all positions before 3:30 PM IST.
* **Workstation Configuration**:
  1. Toggle **Enable Intraday Workstation** `ON` in the Streamlit sidebar.
  2. Input your **Total Trading Capital** and **Max Risk per Trade** (recommended: 1.0% = ₹5,000 for a ₹5L account).
  3. Keep the **Nifty Intraday VWAP Kill-Switch** card visible. Do not enter any long trades if Nifty is trading below its daily VWAP (indicated by a red warning).
* **Execution**: Place orders when the **Live 15-Min Breakout Scanner** alerts you that a stock has crossed its Pivot High on a volume spike. Stop-loss is set tight (1.5% below pivot).

### 15-to-20-Day Holding Period (Swing Trading - Default Mode)
* **Goal**: Ride mid-cap breakouts to a clean 15% target.
* **Workstation Configuration**:
  1. In the sidebar, set **Select Model Target Scenario** to `🎯 15% (Target Predictor - Default)`.
  2. Set **Initial Stop-Loss** to `7.5%` and **Swing Target Profit** to `15.0%`.
* **Execution**: Place pre-market **Buy GTT Limit** orders at the printed **Pivot High** price. Take profit on 50% of the position at +15%, move the remaining 50% stop to breakeven, and exit on Day 20 if targets aren't reached (Time Exit).

### Long-Term Holding Period (Months to Years Compounding)
* **Goal**: Buy high-conviction, high-ROCE compounders right as their multi-year markup begins.
* **Workstation Configuration**:
  1. In the sidebar, set **Select Model Target Scenario** to `🚀 50% (Multibagger - Strict / Selective)`. Click *Retrain XGBoost Classifier* to build the selective model.
  2. Check the **Structural Growth Compounder** tab.
  3. Turn **Enforce Quality (ROCE/ROE >= 18% & Debt/Equity <= 0.5)** `ON`.
* **Execution**: Buy breakouts using GTT orders. Take swing profit on 50% of the shares at +30% to cover risk. Hold the remaining 50% runner indefinitely, exiting ONLY when the daily candle closes below the **150-day or 200-day Moving Average** (printed on the chart).

---

## 5. AI Developer Hand-off Notes (For Future AIs)

When editing or maintaining this repository, you must adhere to these quantitative guardrails:

> [!WARNING]
> **Lookahead Bias Danger**: The database only holds *current* fundamental snapshots. Joining current fundamentals to past historical dates in vector searches or F-Score backtests introduces lookahead bias. The Cosine Similarity engine must remain **3D Technical-Only** (`volatility_contraction_score`, `volume_surge_score`, `momentum_score`).

* **Model Training Alignment**: The XGBoost classifier is trained on path-dependent labels. Do not use simple closing return shifts (`shift(-20)`), as this ignores stopped-out trades.
* **In-Flight Filter**: Always filter out active setups from the last 20 days of the training set (`days_to_target == 0`) to prevent the model from learning incomplete trades as failures.
