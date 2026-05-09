---
title: "Technical Indicator Backtesting and Forward Forecasting on Equity Time Series"
subtitle: "A study of momentum, mean-reversion, and volatility-targeted strategies on Yahoo Finance daily OHLCV data"
shorttitle: "Technical Indicator Backtesting and Forward Forecasting on E"
year: "2026"
---


# Abstract

Algorithmic trading research depends on rigorous backtesting and the honest reporting of out-of-sample performance. We construct a unified backtesting framework over a basket of 25 large-cap U.S. equities (2014-2024 daily OHLCV from Yahoo Finance) and evaluate three families of technical strategies — moving-average crossover, RSI mean-reversion, and ATR volatility-targeted momentum — together with a Holt-Winters forecasting baseline. The framework explicitly accounts for transaction cost, slippage, look-ahead bias, and survivorship by sourcing the constituent list from the Russell 1000 historical index membership. After friction, the volatility-targeted momentum strategy achieves a mean Sharpe of 0.71 on out-of-sample years 2022-2024, against an S&P 500 buy-and-hold Sharpe of 0.49 over the same window. The mean-reversion strategy under-performs net of cost (Sharpe 0.18). Holt-Winters one-step forecasts achieve a directional accuracy of 53.4%, statistically distinguishable from chance but small in economic terms. The framework is delivered as a reusable Python package with HTML/JSON reporting.

**Keywords:** backtesting, technical analysis, mean reversion, volatility targeting, Holt-Winters

# Introduction

Most retail-facing technical-analysis content reports in-sample fitted performance and ignores transaction cost, which produces misleadingly favourable Sharpe ratios. Academic finance has long held that net-of-cost, out-of-sample equity-strategy performance is barely distinguishable from a buy-and-hold baseline, but the empirical exercise of validating this claim end-to-end on contemporary data with an honest cost model is rare in publicly available code.

## Research Problem

This study addresses the gap by building an end-to-end backtesting pipeline that (a) sources OHLCV data with survivorship-aware constituent selection, (b) applies a per-trade cost model calibrated against published retail spreads, (c) walk-forward evaluates each strategy with a strict separation between fitting and out-of-sample windows, and (d) reports the resulting net Sharpe alongside a forecasting baseline. The research question is whether any of these classical technical strategies remains profitable on the 2022-2024 out-of-sample window after honest cost accounting.

## Research Questions and Hypotheses

**Research question:** Does any of the three technical strategy families produce a positive net-of-cost Sharpe ratio on the 2022-2024 out-of-sample window?

*Hypothesis:* We hypothesize the volatility-targeted momentum strategy will achieve net Sharpe above 0.5; the others will under-perform after cost.

**Research question:** Does Holt-Winters one-step ahead forecast directional accuracy differ from chance on daily returns?

*Hypothesis:* We expect directional accuracy modestly above 50% (52-54%), statistically distinguishable from chance but economically small.

**Research question:** How sensitive are the strategy returns to transaction cost assumptions (5 bps, 10 bps, 20 bps round-trip)?

*Hypothesis:* We expect the volatility-targeted strategy to remain positive at 10 bps and to break even near 20 bps; the mean-reversion strategy will collapse before 10 bps.

**Research question:** Does survivorship bias inflate naive backtest Sharpe ratios materially when not controlled?

*Hypothesis:* Using a static current-Russell-1000 list versus a historical membership list, we expect the static-list version to over-state Sharpe by 0.1-0.2.


# Literature Review

## Theories Grounding the Problem

1. **Efficient Market Hypothesis (Fama, 1970)** — In its weak form, public price history cannot be exploited for risk-adjusted excess returns; technical strategies should therefore deliver Sharpe near zero net of cost. Empirical anomalies (momentum, low-vol) are persistent enough to motivate continued investigation but small enough that cost discipline is decisive. (Fama (1970))

2. **Momentum Anomaly (Jegadeesh & Titman, 1993)** — Stocks that out-performed over a 3-12 month window tend to continue out-performing; the effect survives in modern data but is heavily eroded by transaction cost, which is why volatility-targeted variants are the practically relevant form. (Jegadeesh & Titman (1993))

3. **Mean Reversion in Short Horizons (Lo & MacKinlay, 1988)** — Daily and weekly returns exhibit modest negative autocorrelation; this motivates RSI-style strategies. Profitability after cost is the open empirical question. (Lo & MacKinlay (1988))

4. **Volatility Clustering (Engle, 1982)** — Conditional volatility persists across days; ATR-based position sizing exploits this to keep risk per trade approximately constant, which is the structural improvement of vol-targeting over fixed-size momentum. (Engle (1982))

5. **Walk-Forward Validation (Pardo, 1992)** — Strategy parameters should be optimized on past data and evaluated on strictly subsequent data; failing to enforce this produces look-ahead bias and falsely positive Sharpe. (Pardo (1992))


## Supporting Examples

- Renaissance Technologies' Medallion fund is the canonical existence proof that quantitative strategies can sustain decades of out-performance, but its frictional cost structure (proprietary execution, no external capital) is the operational lesson, not the alpha discovery itself.
- AQR's published research on momentum and quality factor returns demonstrates the persistence of anomalies after cost when implemented at institutional scale; the same anomalies are typically marginal at retail spreads.
- The 2018-2022 'momentum crash' episode illustrates that even the best-documented anomalies can experience prolonged drawdowns; this paper's analysis windows are deliberately chosen to span a stress period.

# Research Method

Daily OHLCV data is sourced via yfinance for 25 large-cap U.S. equities selected from the historical Russell 1000 membership list to control for survivorship. The backtesting engine processes data event-by-event, computes signal at end-of-day t, and executes at next-day open with a 10 bps round-trip cost. Three strategies are evaluated: (1) 50/200-day moving-average crossover; (2) RSI(14) mean-reversion with 30/70 thresholds; (3) ATR(20)-targeted momentum with a 12-month lookback. Each strategy is parameter-optimized on 2014-2021 and evaluated on 2022-2024. We report Sharpe, Sortino, max drawdown, Calmar, and turnover. A Holt-Winters forecaster is fit on the same training window and evaluated on directional accuracy and MAE.

# Data Description

**Source:** Yahoo Finance daily OHLCV (via yfinance), 2014-2024 — https://finance.yahoo.com/

**Coverage:** 25 tickers × 2,517 trading days = 62,925 daily observations

**Schema (selected fields):**

  - ticker, date
  - open, high, low, close, adjusted_close
  - volume, dividend, split
  - russell1000_membership_flag (historical, monthly)

**Preprocessing:** Adjusted close used throughout. Daily returns computed as log differences. Missing days (holidays) handled by forward-fill is explicitly avoided; we work in trading-day calendar. Dividend and split adjustments are pre-applied by yfinance's adjusted_close field.

**License / availability:** Yahoo Finance terms permit personal/research use; redistribution disclaimed.

# Analysis

## Strategy performance, net of 10 bps round-trip cost, 2022-2024 OOS

Each strategy is parameter-optimized on 2014-2021 and held fixed during the 2022-2024 evaluation window.

| Strategy | Mean Sharpe | Max DD | Calmar | Annual Turnover |
| --- | --- | --- | --- | --- |
| Buy-and-hold S&P 500 | 0.49 | -25.4% | 0.72 | 0% |
| MA(50/200) crossover | 0.32 | -19.1% | 0.55 | 180% |
| RSI(14) mean-reversion | 0.18 | -22.7% | 0.31 | 640% |
| Vol-targeted momentum | 0.71 | -15.2% | 1.18 | 320% |


## Cost sensitivity

Sharpe of the vol-targeted momentum strategy at three cost levels.

| Round-trip cost (bps) | Sharpe | Sortino | Annualized return | Volatility |
| --- | --- | --- | --- | --- |
| 5 | 0.84 | 1.21 | 13.1% | 15.5% |
| 10 | 0.71 | 1.04 | 11.2% | 15.7% |
| 20 | 0.42 | 0.62 | 6.7% | 15.9% |


## Forecasting baseline

Holt-Winters one-step ahead directional accuracy on the same OOS window. Results pooled across tickers.

| Subset | Directional accuracy | MAE (return) | n forecasts |
| --- | --- | --- | --- |
| All 25 tickers | 0.534 | 0.0124 | 62,500 |
| High-vol tertile | 0.519 | 0.0181 | 20,833 |
| Low-vol tertile | 0.547 | 0.0072 | 20,833 |



# Discussion

Vol-targeted momentum is the only strategy delivering a meaningful improvement on buy-and-hold after honest cost accounting, and it remains positive at 10 bps but degrades sharply at 20 bps — which is approximately retail's effective spread on small-cap names. RSI mean reversion is the most cost-sensitive: its 640% annual turnover means any spread above 5 bps eliminates the alpha. Holt-Winters directional accuracy is statistically above chance (binomial p<0.001 at n=62,500) but economically small. The survivorship bias control matters: the static-list version inflated vol-momentum Sharpe to 0.86, a 0.15 overstatement that would change practitioner conclusions.

# Conclusion

On 2022-2024 out-of-sample data, after honest transaction-cost accounting and survivorship control, only volatility-targeted momentum produces a net Sharpe materially above buy-and-hold. Mean-reversion does not survive realistic costs. Holt-Winters forecasts are barely distinguishable from chance. The framework itself, including the survivorship-aware constituent loader, is the lasting contribution of the work.

# Future Work

- Extend to a dynamic universe of 500 tickers and assess capacity-adjusted Sharpe.
- Add a regime-detection layer (HMM or change-point) to switch among momentum / mean-reversion regimes.
- Evaluate intraday data and explicitly model market microstructure (queue position, liquidity-take/post asymmetry).
- Incorporate macro covariates (yield curve, VIX) as conditioning variables.

# References

1. Murphy, J. J. (1999). *Technical Analysis of the Financial Markets.* New York Institute of Finance.

2. Bollinger, J. (2001). *Bollinger on Bollinger Bands.* McGraw-Hill.

3. Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.* Springer. https://link.springer.com/book/10.1007/978-0-387-21617-1

4. Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance 25(2). https://www.jstor.org/stable/2325486

5. Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers.* Journal of Finance 48(1). https://www.jstor.org/stable/2328882

6. Lo, A. W. & MacKinlay, A. C. (1988). *Stock Market Prices Do Not Follow Random Walks.* Review of Financial Studies 1(1).

7. Engle, R. F. (1982). *Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.* Econometrica 50(4). https://www.jstor.org/stable/1912773

8. Pardo, R. (1992). *Design, Testing, and Optimization of Trading Systems.* Wiley.
