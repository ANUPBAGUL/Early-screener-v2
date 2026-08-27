# Implementation Plan: High-Growth & Multibagger Screener

## Goal Description
The primary goal of this system is to identify outlier stocks within the top 700 Indian equities that have the potential for explosive growth—specifically targeting moves like **50% in 1 month or 1000% in 1 year**. 

While adapting the "Market DNA" concept from the previous review, this screener shifts focus from general portfolio risk-management to **extreme momentum and anomaly detection**. It aims to answer the question: *What exact market, sector, and company conditions (DNA) existed right before historical stocks experienced massive, parabolic runs, and which stocks exhibit that exact DNA today?*

---

## User Review Required

> [!IMPORTANT]
> **1. Defining "Explosive DNA" Features**
> To find 50% monthly or 1000% yearly gainers, traditional valuation metrics (like low P/E) often fail. We must focus on:
> * **Extreme Volume Anomalies:** Sudden, massive spikes in traded volume compared to historical averages.
> * **Volatility Contraction (VCP):** Periods of extreme price tightness followed by breakouts.
> * **Sector Momentum:** Is the stock part of a sector experiencing a massive macro tailwind?
> * **Do you agree with prioritizing these momentum/breakout technicals alongside fundamental catalysts?**
>
> **2. Handling Highly Imbalanced Data**
> A 50% move in 1 month is statistically rare (an outlier). Standard machine learning models struggle with this because they optimize for the average. We will need to formulate this as an **Anomaly Detection** or **Imbalanced Classification** problem (e.g., training XGBoost specifically to maximize recall on the top 1% of historical movers).
>
> **3. Upstox API Credentials**
> We will still need your Upstox API Key, Secret, and Redirect URI to fetch the EOD data for the top 700 stocks. 

---

## Proposed Changes

We will build the codebase in the `d:\Projects\early screener` directory.

### Component 1: Data Ingestion & Storage

#### [NEW] [database/schema.sql](file:///d:/Projects/early%20screener/database/schema.sql)
SQLite database schema optimized for time-series and breakout analysis:
* `stocks`: Top 700 symbols.
* `price_history`: Daily OHLCV data.
* `technical_features`: Pre-calculated metrics (Volume surges, ATR, Bollinger Band width, moving average slopes).
* `multibagger_dna`: Vector representations of the stock's state (macro, sector, and technical momentum).

#### [NEW] [pipeline/upstox_client.py](file:///d:/Projects/early%20screener/pipeline/upstox_client.py)
Handles Upstox OAuth login and fetches bulk historical OHLCV data.

#### [NEW] [pipeline/feature_engineer.py](file:///d:/Projects/early%20screener/pipeline/feature_engineer.py)
The core logic for defining the "Setup". It will compute:
* Volume relative to 50-day average.
* Price consolidation metrics (volatility contraction).
* Distance from 52-week highs.

### Component 2: The "Outlier" Engine

#### [NEW] [engine/similarity_engine.py](file:///d:/Projects/early%20screener/engine/similarity_engine.py)
Instead of finding general analogues, this engine scans historical data for **known multibaggers** (e.g., historical periods where a stock actually *did* go up 1000%) and computes their "State Vector" just prior to the breakout. It then scores current stocks based on cosine similarity to those historical pre-breakout states.

#### [NEW] [engine/breakout_model.py](file:///d:/Projects/early%20screener/engine/breakout_model.py)
An XGBoost classifier trained exclusively to predict the probability of a $>50\%$ move in 1 month, using cost-sensitive learning to heavily penalize missing a true breakout.

### Component 3: The Screener Dashboard

#### [NEW] [app.py](file:///d:/Projects/early%20screener/app.py)
A high-performance **Streamlit** dashboard tailored for aggressive growth hunting:
* **The "Launchpad" Screener**: Ranks the top 700 stocks daily based on their combined Similarity Score (to historical multibaggers) and XGBoost probability.
* **Breakout Radar**: Visualizes stocks currently experiencing maximum volume contraction or expansion.
* **Historical Evidence Viewer**: When you click a flagged stock, it shows *why* it was flagged by displaying the historical stock (and date) that had the most similar pre-breakout DNA.

---

## Verification Plan

### Automated Verification
* Define historical test cases: Feed the engine data for known massive movers (e.g., a specific stock in a specific year that gained 500%) up to the day before the run, and verify the model flags it.

### Manual Verification
* Run the Streamlit dashboard locally.
* Review the top 10 recommended stocks daily and manually analyze their charts to ensure the technical setups align with high-growth breakout patterns (VCP, momentum).
