# Momentum Strategy Reference Manual: Hunting Multibaggers Scientifically

This document details the quantitative, technical, and fundamental strategy implemented in the High-Growth & Multibagger Screener. Rather than attempting to "predict the next price" in isolation, this system is designed as a **Scenario & Market Memory Engine** that identifies setups matching the exact "DNA footprint" of historical stocks right before they achieved explosive rallies ($\ge 50\%$ in a month or $10x$ in a year).

---

## 1. The Core Philosophy: Scenario Probability over Price Pointing

As outlined in the expert review of stock prediction ideas, predicting an exact price target (e.g., *"Stock XYZ will reach ₹1,420"*) is statistically indefensible due to market noise. 

Instead, this system asks: **"Given the technical, fundamental, and market regime conditions of this stock today, what outcomes have historically followed under similar conditions, and with what probability?"**

It builds a multi-dimensional state vector (Market DNA) for each stock and compares it to a historical database to output a probability distribution of future outcomes.

```
[Current Stock State] ──> [Database Cosine Search] ──> [XGBoost ML Classifier] ──> [1:4 Risk-Reward Trade Plan]
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

Before a stock's volatility contraction or volume is checked, the stock must prove it is in a structural uptrend. We enforce Mark Minervini's exact **Trend Template** to isolate Stage 2 accumulation phases:

1. **SMA Stack**: The current stock price must be above its 50-day, 150-day, and 200-day Simple Moving Averages (SMAs).
2. **SMA Ordering**: The moving averages must be stacked sequentially:
   $$\text{SMA 50} > \text{SMA 150} > \text{SMA 200}$$
3. **200 SMA Slope**: The 200-day SMA must be trending upward (current 200 SMA must be higher than its value 20 trading days ago).
4. **Historical Proximity**: The stock price must be within 25% of its 52-week High (ensuring it is near overhead resistance breakouts, not languishing in a downtrend).

*Stocks that do not pass these criteria are discarded before any similarity calculations are run.*

---

## 4. Phase 3: Technical Anomaly & Setup Detection (VCP & Volume Surge)

Once a stock is verified to be in a Stage 2 uptrend, the screener scans for two critical footprints left by institutional accumulation:

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

To ensure technical setups are backed by real institutions and not retail cornering, the system enforces strict liquidity limits:

* **Size Constraint**: Filters for stocks with a Market Cap between **₹300 Crores and ₹10,000 Crores**. 
  - *Why*: Companies larger than ₹10,000 Cr rarely multiply by 10x or 100x due to the law of large numbers. Companies under ₹300 Cr lack institutional interest and suffer from manipulation.
* **Volume Constraint**: Filters for stocks with a 50-day Average Daily Volume of **$\ge 200,000$ shares**, guaranteeing enough trading depth for entry and exit.

---

## 6. Phase 5: Fundamental Quality Constraints

For true multibagger potential, technical patterns must be backed by a strong operational engine.

1. **Capital Efficiency (ROCE / ROE $\ge 18\%$)**: Ensures the business generates high returns on its capital, enabling it to self-finance hyper-growth without shareholder dilution.
2. **Balance Sheet Safety (Debt-to-Equity $\le 0.5$)**: Filters out highly leveraged companies (avoiding debt traps like Yes Bank or DHFL).
3. **Valuation (PEG Ratio $\le 1.0$)**: Identifies Growth At a Reasonable Price (GARP).

---

## 7. The Mathematical Engine: Cosine Similarity Matching

Instead of relying on basic heuristic rules, the system represents each stock's current state as a vector in a 3D feature space:
$$\vec{V} = [\text{Volatility Contraction}, \text{Volume Surge}, \text{Momentum}]$$

### The DNA Library
* We scanned 429,954 historical daily records for our stocks and identified **3,798 true breakout setups** where a stock subsequently went up $\ge 50\%$ in 20 trading days.
* The feature vectors on those breakout days are saved in the database as the **Multibagger DNA Library**.

### Cosine Similarity Calculation
The engine standardizes the features using z-score stats to balance dimensions, and computes the cosine similarity between today's stock vector and all vectors in the library:
$$\text{Similarity}(\vec{A}, \vec{B}) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

* **Historical Match**: The system returns the closest historical counterpart from the 3,798-breakout library, showing you exactly which stock and date matches today's setup (e.g., matching Tata Steel in April 2023) and what return that analogue achieved.

---

## 8. The Predictive Engine: XGBoost Breakout Probability

The similarity score is paired with an **XGBoost Classifier** model trained on all historical records.

* **Labeling**: Target is `1` if subsequent 20-day returns $\ge 50\%$, and `0` otherwise.
* **Training Details**: The model is trained using cost-sensitive learning to maximize recall on class 1, outputting a probability (0-100%) that today's setup will result in a $>50\%$ move in 1 month.

---

## 9. Phase 6: Strict Risk Management (The Hybrid Exit Strategy)

The mathematics of momentum trading rely on asymmetric risk: keeping losses small and letting winners run. To allow the stock to compound into a true multibagger without selling too early, the system implements a **Hybrid Exit Strategy**:

1. **Initial Stop-Loss (7.5%)**: Enforces a strict stop-loss level at `Close Price * 0.925` for the entire position at start. If the breakout fails, the trade is cut immediately.
2. **The Swing Exit (50% of position)**: Sell 50% of the shares when the stock hits a **30% profit target** (`Close Price * 1.30`). Booking half the position at 30% mathematically locks in a 1:4 risk-reward ratio and covers the risk of the entire trade.
3. **The Multibagger Runner (50% of position)**: The remaining 50% is left completely uncapped with no profit target. Instead of a take-profit order, you shift to a trailing stop-loss, cutting the trade if it daily closes below the **50-day SMA** or **150-day SMA**. This allows the stock to run for months or years, giving it the time it needs to potentially multiply 10x or 100x.
