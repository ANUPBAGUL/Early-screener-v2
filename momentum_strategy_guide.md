# Momentum Strategy Reference Manual: Hunting Multibaggers Scientifically

This document details the quantitative, technical, and fundamental strategy implemented in the High-Growth & Multibagger Screener. Rather than attempting to "predict the next price" in isolation, this system is designed as a **Scenario & Market Memory Engine** that identifies setups matching the exact "DNA footprint" of historical stocks right before they achieved explosive rallies ($\ge 50\%$ in a month or $10x$ in a year).

---

## 1. The Core Philosophy: Scenario Probability over Price Pointing

Instead of predicting an exact price target, this system compares today's stock state to a historical database to output a probability distribution of outcomes. To avoid the sequencing contradiction of applying strict momentum filters to distressed assets, the pipeline splits into two separate execution branches:

```
[Current Stock State]
   ├── Path 1: Momentum Engine (General & Structural Compounders)
   │     └── Stage 2 Uptrend ──> VCP Consolidation ──> Volume Surge ──> Cosine Similarity & XGBoost
   └── Path 2: Value Engine (Turnarounds & Cyclical Deep Value)
         └── Fundamental Rules ──> Solvency/Asset Filters ──> Volume Spike Reversal (>= 1.5x)
```

---

## 2. Phase 1: The Broader Market Regime Filter (The "Guardrail")

Research shows that **90.77% of successful breakouts** occur when the broader index is in an uptrend. Running a long momentum screener during a bear market or major index correction results in a high percentage of failed breakouts ("fakeouts") that hit stop-losses.

### The Nifty 50 Monthly 10 EMA Filter
* **Calculation**: The system queries monthly candles for the Nifty 50 Index (`NSE_INDEX|Nifty 50`) and computes a 10-period Exponential Moving Average (EMA).
* **Rule**: 
  - **Bull Market (Nifty Close $\ge$ 10 EMA)**: Breakouts are active.
  - **Bear Market (Nifty Close $<$ 10 EMA)**: The "Kill Switch" triggers. Breakout signals are suppressed by default.
* **Why it works**: Monthly EMAs filter out short-term noise, showing the true institutional direction of the index.

---

## 3. Phase 2: Structural Uptrend Check (Minervini Stage 2 Template)

For momentum-based assets, the stock must prove it is in a structural uptrend before its volatility contraction or volume is checked. We enforce Mark Minervini's exact **Trend Template** to isolate Stage 2 accumulation phases:

1. **SMA Stack**: The current stock price must be above its 50-day, 150-day, and 200-day Simple Moving Averages (SMAs).
2. **SMA Ordering**: The moving averages must be stacked sequentially:
   $$\text{SMA 50} > \text{SMA 150} > \text{SMA 200}$$
3. **200 SMA Slope**: The 200-day SMA must be trending upward (current 200 SMA must be higher than its value 20 trading days ago).
4. **Historical Proximity**: The stock price must be within 25% of its 52-week High (ensuring it is near overhead resistance breakouts, not languishing in a downtrend).

*Note: Stocks are routed through this check depending on their core quantitative archetype (Path 1 vs. Path 2), preventing the elimination of distressed turnaround value candidates.*

---

## 4. Phase 3: Technical Anomaly & Setup Detection (VCP & Volume Surge)

For stocks routed through the Momentum Path, once verified to be in a Stage 2 uptrend, the screener scans for two critical footprints left by institutional accumulation:

### A. Volatility Contraction Pattern (VCP)
VCP represents a period where supply is being systematically absorbed by strong hands. Each contraction represents a shakeout of weak hands.
* **Quantification**: Calculated as the 20-day standard deviation of closing prices divided by the close price:
  $$\text{VCP Score} = \left( \frac{\sigma_{\text{close, 20}}}{\text{Close Price}} \right) \times 100$$
* **Constraint**: A lower score indicates tighter price consolidation (typically $< 5\%$), setting up the stock like a coiled spring.

### B. Institutional Volume Surge
An increase in price must be accompanied by an increase in volume, indicating that large institutions are buying shares in bulk.
* **Quantification**: Calculated as today's volume relative to its 50-day average:
  $$\text{Volume Surge Score} = \frac{\text{Volume}}{\text{Average Volume}_{50}}$$
* **Constraint**: A high score (typically $\ge 3.0x$) indicates a massive influx of institutional capital.

---

## 5. Phase 4: Institutional Size & Liquidity Guards

To ensure technical setups are backed by real institutions and not retail cornering, the system enforces strict liquidity limits across an expanded universe of **1,200 stocks** (covering Nifty Total Market Index and active NSE corporate equities):

* **Size Constraint**: Filters for stocks with a Market Cap between **₹300 Crores and ₹100,000 Crores** depending on the selected screening archetype.
  - *Why*: Targeting small-to-midcap sweet spots (₹300Cr - ₹10,000Cr) maximizes multi-bagger potential, while allowing up to ₹100,000Cr for turnaround situations.
* **Volume Constraint**: Filters for stocks with a 50-day Average Daily Volume of **$\ge 200,000$ shares**, guaranteeing enough trading depth for entry and exit.

---

## 6. Phase 5: Quantitative Multibagger Archetypes

For true multibagger potential, technical patterns must be backed by a strong operational or cyclical engine. The system implements three distinct screening modes corresponding to the core multibagger archetypes outlined in corporate finance:

### A. The Turnaround Multibagger
* **Description**: Distressed or out-of-favor companies emerging from losses, reducing debt, or undergoing structural reforms.
* **Filtering Rules**:
  - Market Cap: ₹1,000 Cr to ₹100,000 Cr
  - Solvency: Debt-to-Equity $\le$ 0.75
  - Operational Acceleration: Trailing Revenue Growth $\ge$ 15%
  - Profitability: ROCE / ROE $\ge$ 15%

### B. The Cyclical Deep Value Play
* **Description**: Commodity, shipping, and heavy industrial companies at the trough of the global cycle undergoing violent operational inflections.
* **Filtering Rules**:
  - Asset Price: Price-to-Book (P/B) $\le$ 3.0 (liquidating discount floor)
  - Inflection Point: YoY Quarterly Profit Growth $\ge$ 100% (variance surge)
  - Solvency under Duress: Interest Coverage Ratio $\ge$ 4.0
  - Return Normalization: ROCE / ROE $\ge$ 15%

### C. The Structural Growth Compounder
* **Description**: High-margin, moat-driven companies operating in secular growth industries (AI, railways, green energy) that compound returns over decades.
* **Filtering Rules**:
  - Capital Efficiency: ROCE / ROE $\ge$ 20%
  - Balance Sheet Purity: Debt-to-Equity $\le$ 0.10 (Virtually Debt-Free)
  - Secular Growth: Consistent Annual Growth $\ge$ 15%

---

## 7. The Mathematical Engine: 6D Hybrid Cosine Similarity

Instead of relying on basic technical rules, the system represents each stock's state as a vector in a **6D Hybrid Space** combining technical charts and corporate balance sheets:
$$\vec{V} = [\text{Volatility Contraction}, \text{Volume Surge}, \text{Momentum}, \text{Debt-to-Equity}, \text{Price-to-Book}, \text{ROCE}]$$

### Z-Score Normalization
To prevent high-scale dimensions from dominating the matching, the engine normalizes both the target stock and DNA candidate vectors relative to the active universe stats:
$$Z_i = \frac{X_i - \mu_i}{\sigma_i}$$
* **Missing Value Imputation**: Missing metrics default to the universe mean ($\mu_i$), resulting in a neutral Z-score of `0`.

### Cosine Similarity Calculation
Computes the cosine similarity between today's 6D Z-score normalized stock vector and all vectors in the DNA library:
$$\text{Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|} \times 100$$

* **Historical Match**: Returns the closest counterpart from the 3,798-breakout library (e.g. matching CPCL in April 2023) that matches *both* the chart setup and corporate capital structure, showing what return that analogue subsequently achieved.

---

## 8. Algorithmic Quality Filters, Screener.in Fundamentals & Rotation Indicators

### A. Screener.in Verified Data Engine
To prevent data contamination and stale balance sheets from global scrapers, the system extracts standalone and consolidated financial statements directly from **Screener.in**:
* **Capital Efficiency**: ROCE and ROE (%) parsed from official BSE/NSE filings.
* **Solvency**: Debt-to-Equity ratio extracted from balance sheet borrowings and total equity.
* **Operational Inflection**: YoY Quarterly Sales Growth and Net Profit Growth.
* **Coverage**: Interest Coverage Ratio ($\frac{\text{Operating Profit}}{\text{Interest Expense}}$).
* **Data Quality Score (`📊`)**: Ranks data reliability on a `0` to `3` scale (`🟢 Verified`, `🟡 Good`, `🟥 Partial`, `⬛ No Data`).

### B. Pre-Earnings Blackout Guard (`⚠️`)
* **Rule**: Tracks trading days to next corporate earnings announcement (`days_to_earnings`).
* **Protection**: Flags stocks with earnings in $\le 5$ trading days (`⚠️ EARNINGS 3d`). The sidebar toggle **"Hide Earnings Risk (≤5 days)"** suppresses these high-risk setups to prevent overnight gap-down stop loss destruction.

### C. Archetype-Specific Confluence Scoring (%)
Rather than applying a generic 60/40 blend across all assets, the system dynamically weights the confluence score based on the underlying investment thesis:
* **Momentum & Structural Paths (General & Structural)**: 80% XGBoost Breakout Probability + 20% F-Score. Technical breakout probability is the primary edge.
* **Value & Recovery Paths (Turnarounds & Cyclicals)**: 30% XGBoost Breakout Probability + 70% F-Score. Fundamental balance sheet inflection is the primary edge.

$$\text{Confluence Score} = (\text{XGBoost Prob} \times w_{\text{xgb}}) + (\text{F-Score} \times 10 \times w_{\text{fscore}}) + \text{Sector Bonus}$$

### D. Sector Co-Breakout Clustering (Institutional Rotation Index)
* **Trigger**: If a sector has **$\ge 2$ concurrent breakouts** on the same day, it is flagged as an active **Sector Co-Breakout Cluster**.
* **Impact**: Flagged stocks inside a cluster receive a **+10% Confluence Score bonus** (capped at 100%), highlighting institutional sector rotation.

---

## 9. The Predictive Engine: XGBoost & Realistic DNA Returns

The similarity score is paired with an **XGBoost Classifier** model trained on 1.37+ million historical candles:

* **Realistic 20-Day Close-to-Close Return**: Replaced theoretical peak-high returns with the realized holding-period return ($\frac{\text{Close}_{t+20} - \text{Close}_t}{\text{Close}_t}$).
* **Configurable Target Threshold (`BREAKOUT_LABEL_THRESHOLD`)**:
  - `0.20` (+20%): **📈 Swing Growth** mode (broad signals, high frequency).
  - `0.30` (+30%): **🎯 Target Predictor** mode (default — directly predicts if setup reaches your +30% swing profit target).
  - `0.50` (+50%): **🚀 Multibagger** mode (selective & conservative).
* **Dynamic Temporal Retraining**: The model splits training data dynamically at **6 months prior to the current system date**. Signals from the last 6 months are evaluated in a held-out **Out-of-Sample (OOS) Win-Rate Ledger**.

---

## 10. Volatility-Adjusted Risk Management (The Hybrid Exit Strategy)

The mathematics of momentum trading rely on asymmetric risk: keeping losses small and letting winners run. To remove the rigidity of fixed percentage parameters, the system implements a **Volatility-Adjusted Risk Management Workspace**:

1. **Dynamic Initial Stop-Loss**: The system discards hardcoded percentages (e.g. 7.5%) and calculates a custom stop-loss based on the stock's actual VCP (standard deviation) score:
   $$\text{Dynamic SL (\%)} = \text{VCP Score} \times 1.5 \quad (\text{Bounded between 5.0\% and 15.0\%})$$
   This gives stable stocks a tight stop-loss (avoiding excess drag) and volatile stocks a wider stop-loss (avoiding shakeouts).
2. **The Swing Exit (50% of position)**: Sell 50% of the shares when the stock hits the timeframe target (e.g. +15.0% for swing). This locks in profits and covers the risk of the whole trade.
3. **The Multibagger Runner (50% of position)**: The remaining 50% shifts to a trailing stop-loss (exiting only if the daily close drops below the **50-day or 200-day SMA**) to let compounders run uncapped.
4. **Rupee Risk Equalization (Position Sizer)**: To prevent highly volatile stocks from creating outsized portfolio drawdowns, the position sizer equalizes the rupee risk across all trades:
   $$\text{Shares to Buy} = \frac{\text{Total Capital} \times \text{Risk \%}}{\text{Entry Price} \times \text{Dynamic SL \%}}$$
   This ensures that if the trade hits its dynamic stop-loss, you lose exactly the risk amount (e.g. 1.0% of capital) regardless of the stock's volatility profile.

---

## 11. Empirical Strategy Verification (The Backtest Ledger)

To validate the strategy's real-world accuracy without subjective bias, the system runs a **concrete path-dependent backtest**:

* **Chronological Simulation**: Checks stop-loss on `low` first (pessimistic) before target on `high`.
* **Out-of-Sample (OOS) Win-Rate**: Displays separate hit-rates for full history vs. held-out 6-month test data to detect model overfitting.
* **Profit Factor & Kelly Criterion**: Measures total gross gains vs. losses and optimal capital allocation percentage.

---

## 12. Intraday Trading Workstation Suite

For intraday execution and market-hours monitoring, the application includes a dedicated **Intraday Workstation**:

### A. Sidebar Toggle Control (`⚡ Intraday Workstation Mode`)
* **OFF**: Displays the EOD research screener view with all 4 tabs intact.
* **ON**: Expands the Intraday Workstation suite at the top of the dashboard.

### B. 🎯 Pivot High (₹) Intraday Trigger Price
Calculated as the maximum high over the last 15-day consolidation handle:
$$\text{Pivot High} = \max(\text{High}_{15\text{d}})$$
A stock breaching this price intraday on volume triggers the true volatility explosion.

### C. 🧮 Fixed-Risk Position Sizing & Margin Calculator
User inputs **Total Trading Capital (₹)** and **Risk Per Trade (%)**:
$$\text{Risk Amount (₹)} = \text{Trading Capital} \times \left(\frac{\text{Risk \%}}{100}\right)$$
$$\text{Intraday SL (₹)} = \text{🎯 Pivot High} \times (1 - 0.015)$$
$$\text{Shares to Buy} = \left\lfloor \frac{\text{Risk Amount}}{\text{Pivot High} - \text{Intraday SL}} \right\rfloor$$
$$\text{Capital Required (₹)} = \text{Shares to Buy} \times \text{🎯 Pivot High}$$

### D. 📌 Today's Intraday Watchlist Card & CSV Export
Aggregates the top 15 highest-conviction setups across all archetypes (`Confluence >= 70%`, `Stage 2 == 1`, `Earnings > 5d`) with exact position sizing metrics and a **📥 Download Intraday Watchlist (CSV)** button for broker terminals (Upstox, Zerodha, Dhan).

### E. 🛑 Nifty 50 Intraday VWAP Kill-Switch
Fetches 15-minute live candles for Nifty 50 from Upstox API. If $\text{Nifty LTP} < \text{Nifty 15-Min VWAP}$, the kill-switch triggers: `🔴 Nifty Bearish — Intraday Breakouts Suppressed`.

### F. ⚡ Upstox 15-Min + 60-Min Live Trigger Scanner & Audio Alert
Market-hours scanner monitoring *only* the 15 watchlist stocks. Requires `LTP >= Pivot High`, `15-Min Vol >= 1.5x Avg`, and `60-Min Price > 60-Min VWAP`. Fires a **🚨 LIVE BREAKOUT ALERT** with an HTML5 audio chime sound (`🔊`).

---

## 13. Auction Market Theory: Volume Profile POC & PbD Shapes

To isolate institutional accumulation zones and distinguish breakouts from weak structures, the system implements a rolling 30-day Volume Profile calculation using a vectorized histogram:

### A. Point of Control (POC)
The price level where the absolute maximum trading volume occurred over the last 30 trading days:
$$\text{POC} = \text{Price Bin midpoint with } \max(\text{Volume})$$

### B. Volume Area Density
The percentage of total 30-day volume traded within a tight $\pm 3.0\%$ window around the POC:
$$\text{Volume Area Density (\%)} = \frac{\text{Volume in } [0.97 \times \text{POC}, 1.03 \times \text{POC}]}{\text{Total 30-Day Volume}} \times 100$$
High density ($\ge 50\%$) signals a significant high-volume accumulation base (Balance Area).

### C. PbD Profile Shape Classification
We mathematically classify Nill's three core chart profiles by evaluating where the POC sits relative to the 30-day price range:
* **P-Profile (Accumulation)**: POC is in the upper 35% of the range. Bullish accumulation; standard markup breakouts.
* **b-Profile (Distribution)**: POC is in the lower 35% of the range. Bearish structure representing post-liquidation consolidation. **Breakouts fail here and are blocked/suppressed in strategy suggested risk metrics.**
* **D-Profile (Balance)**: POC is in the center. sideways balance; play range boundaries or wait for momentum.

---

## 14. Trend Extension Guardrails (Anti-FOMO)

To prevent entering trades at the peak of a parabolic markup phase (chasing price), the system tracks the distance between the current close and the 50-Day Moving Average:
$$\text{Extension Ratio} = \frac{\text{Current Close}}{\text{50-Day SMA}}$$

* **Extension Check**: If the Extension Ratio is $> 1.20$ (trading $> 20\%$ above the 50D SMA), the setup is flagged as **`⚠️ Over-Extended`**.
* **Capital Risk Control**: The position sizer dynamically drops the recommended trade risk parameter to **0.25%** to protect capital from immediate mean-reversion pullbacks.

