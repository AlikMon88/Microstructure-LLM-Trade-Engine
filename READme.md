# Quantamental L/S Cross-Sectional Momentum Strategy

## Overview

This project implements a **Market Neutral Long/Short (L/S) Momentum Strategy** for cryptocurrency markets. Unlike traditional "time-series" momentum (which bets on market direction), this strategy isolates **Cross-Sectional Beta-Neutral Alpha**.

It ranks a universe of assets based on relative strength derived from **Order Book Microstructure** and **Price/Volume Dynamics**. The strategy buys the winners and shorts the losers, targeting the **dispersion** (spread) between assets while hedging out broad market risk.

Crucially, this system integrates a **Large Language Model (LLM)** layer as a final risk-management gate, vetting high-momentum candidates for "toxic" news (hacks, exploits, unlocks) before execution.

## Key Features

*   **Market Neutrality:** Hedged exposure (Dollar Neutral). Profits are generated from the spread between strong and weak assets, not Bitcoin's price direction.
*   **Microstructure Alpha:** Goes beyond simple price trends by analyzing Order Book Imbalance (OBI), Slope, and Depth Concentration.
*   **Robust Scoring:** Uses median-based Z-scores (MAD) and Winsorization to handle crypto "whale" outliers effectively.
*   **Composite Aggregation:** Groups correlated features (e.g., Liquidity vs. Momentum) to prevent signal domination by high-variance metrics.
*   **LLM Risk Gating:** Uses Google Gemini + Search to filter out "Artificial Momentum" caused by hacks or manipulation.

---

## Strategy Logic

### 1. Feature Engineering & Grouping
Raw features are grouped to represent specific market phenomena. Directionality is standardized (all scores converted so that Positive = Bullish).

| Feature Group | Logic | Key Metrics |
| :--- | :--- | :--- |
| **Buying Pressure** | Order Book Imbalance implies immediate demand. | `OBI_top5`, `OBI_last30m` |
| **Support Strength** | Steep Bid slope indicates strong limit buyer interest. | `slope_bid` |
| **Resistance Strength** | Steep Ask slope indicates overhead supply (Inverted). | `slope_ask` (Sign Flipped) |
| **Micro-Drift** | Fair value (Microprice) moving ahead of mid-price. | `Microprice_drift` |
| **Liquidity Cost** | Momentum is stronger in liquid assets; high spread is bad. | `RelSpread` (Sign Flipped) |
| **Price Action** | Pure price-based relative strength. | `ROC`, `Price_vs_VWAP` |
| **Energy** | Volume acceleration confirming the move. | `Vol_Acc`, `Volatility` |

### 2. The Scoring Engine
1.  **Normalization:** A Cross-Sectional Z-Score is calculated for every feature across the universe at every timestamp.
2.  **Sign Correction:** "Negative" metrics (Spread, Depth Concentration, Ask Slope) are inverted.
3.  **Group Aggregation:** Features within a group are averaged to create a Group Score.
4.  **Composite Score:** All Group Scores are averaged to produce a final `-2.0` to `+2.0` Alpha Score.

### 3. The LLM "Vetting" Layer
Before a `LONG` signal is accepted, an Agent (Google Gemini) searches live news for the asset.
*   **Trigger:** Asset selected for Long basket.
*   **Prompt:** Checks for "Hacks," "Delistings," "Unlocks," or "FUD."
*   **Action:**
    *   `CLEAN`: Trade executes.
    *   `WARNING`: Trade is rejected (preventing "Catching a Falling Knife").

---

## Installation

### Prerequisites
*   Python 3.8+
*   Google Gemini API Key (Free tier available)
*   AWS S3 (or local CSVs) for Order Book data

```bash
pip install pandas numpy google-generativeai duckduckgo-search
```

### Configuration
Update your API keys in the config or environment variables:
```python
GOOGLE_API_KEY = "your_gemini_key_here"
```

---

## Usage

### 1. Generating Signals
The core logic expects a MultiIndex DataFrame `(datetime, token)` containing your features.

```python
from scorer import make_cross_sectional_scores_multiindex

# Generate the Alpha Scores
results = make_cross_sectional_scores_multiindex(
    df=my_dataframe,
    feature_groups=FEATURE_CONFIG,
    asset_level='token',
    time_level='datetime'
)

# View latest trade ideas
print(results['ideas'][-1])
```

### 2. Running the Simulation (Backtest)
*Note: The LLM backtest performs live searches. For historical accuracy, this acts as a "Forward Simulation" or requires a Point-in-Time news database.*

```python
from backtester import backtest_llm_momentum, NewsFilterAgent

agent = NewsFilterAgent(api_key=GOOGLE_API_KEY)

equity, logs = backtest_llm_momentum(
    strategy_output=results,
    price_df=df,
    llm_agent=agent,
    capital=10000
)
```

---

## Output Interpretation

The system outputs a Dictionary containing:
1.  **`composite`**: A DataFrame of Z-scores.
    *   `> 0`: Outperforming the average.
    *   `< 0`: Underperforming the average.
2.  **`ideas`**: A scheduled list of trades.
    *   **Longs:** Top 20% of Composite Scores.
    *   **Shorts:** Bottom 20% of Composite Scores.

## Disclaimer
*This software is for educational and research purposes only. Do not trade with real funds without extensive testing. The LLM integration relies on third-party search results which may be inaccurate or delayed.*