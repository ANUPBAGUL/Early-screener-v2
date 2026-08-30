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

### 3. Volatility-Adjusted Sizing & Exits
Instead of using rigid parameters, the system dynamically calculates stop-losses and risk limits:
* **Dynamic Stop-Loss (Dynamic SL)**: Stop-loss is set dynamically based on the stock's actual VCP (standard deviation) score:
  $$\text{Dynamic SL (\%)} = \text{VCP Score} \times 1.5 \quad (\text{Bounded between 5.0\% and 15.0\%})$$
* **Rupee Risk Equalization**: Use the built-in **Position Sizer** in the Detailed Viewer to calculate the exact share count to buy based on your capital and risk target (e.g. 1.0% max risk). If the stop-loss is hit, your rupee loss remains perfectly constant.
* **PbD Profile Shape Restrictions (P, b, D)**:
  * **P-Profile (Accumulation)**: POC is in the upper 35% of the range. Bullish setup; standard position sizing.
  * **b-Profile (Distribution)**: POC is in the lower 35% of the range. Bearish structure; **avoid entry** or scale down risk to a minimum of **0.1%**.
  * **D-Profile (Neutral Balance)**: POC is in the middle. Sideways range; trade range boundaries.
* **Trend Extension Warnings (Anti-FOMO)**:
  * If a stock trades $> 20\%$ above its 50-day SMA, it is flagged as **`⚠️ Over-Extended`**. Reduce sizer risk to **0.25%** and wait for a pullback to the moving average.
* **The Runner (Remaining 50%)**: Take profit on the first 50% of the position at the timeframe target. Move the stop-loss of the remaining 50% runner to breakeven, exiting only if the daily candle closes below the **50-day or 200-day Moving Average**.

---

## 4. Time Horizon Discovery Dashboard Guide

The **⏳ Time Horizon Discovery** dashboard tab automatically filters the scanned universe into three watchlists based on their expected breakout target velocities (using historical analogue matching data):

### ⚡ Intraday Watchlist (1-Day Holding Period)
* **Expected Velocity**: **1 to 3 days** (quick momentum scalps).
* **Execution Rules**:
  * Set a tight **1.5% Stop-Loss** and target a **+3.0% profit target**.
  * Use the **Intraday Workstation** live scanner to get 15-minute alerts when these trigger above their Pivot High.
  * Exit 100% of the position before market close (3:30 PM IST) to avoid overnight gap risk.

### 🎯 Swing Watchlist (7-to-30-Day Holding Period)
* **Expected Velocity**: **7 to 30 days** (standard swing cycles).
* **Execution Rules**:
  * Buy using GTT Limit orders exactly at the printed **Pivot High** price.
  * Apply the **Dynamic Stop-Loss** and set a **15.0% swing profit target**.
  * Take profit on 50% of the position at the target, move the rest to breakeven, and exit on Day 30 if target isn't met.

### 💎 Long-Term Watchlist (30-to-250-Day Holding Period)
* **Expected Velocity**: **30 to 250 days** (structural trend compounders).
* **Execution Rules**:
  * Targets high-quality compounders (ROCE $\ge 18\%$ and Debt-to-Equity $\le 0.5$) breaking out of a multi-month base.
  * Set a **30.0% swing exit target** on 50% of the shares to lock in returns.
  * Leave the remaining 50% runner uncapped, exiting only when the daily candle closes below the **50-day or 200-day Moving Average**.

---

## 5. AI Developer Hand-off Notes (For Future AIs)

When editing or maintaining this repository, you must adhere to these quantitative guardrails:

> [!WARNING]
> **Lookahead Bias Danger**: The database only holds *current* fundamental snapshots. Joining current fundamentals to past historical dates in vector searches or F-Score backtests introduces lookahead bias. The Cosine Similarity engine must remain **3D Technical-Only** (`volatility_contraction_score`, `volume_surge_score`, `momentum_score`).

* **Model Training Alignment**: The XGBoost classifier is trained on path-dependent labels. Do not use simple closing return shifts (`shift(-20)`), as this ignores stopped-out trades.
* **In-Flight Filter**: Always filter out active setups from the last 20 days of the training set (`days_to_target == 0`) to prevent the model from learning incomplete trades as failures.
