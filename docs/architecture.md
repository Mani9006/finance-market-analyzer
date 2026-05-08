# Architecture Documentation

## Financial Market Trend Analyzer

This document describes the architecture, design decisions, and module interactions of the Financial Market Trend Analyzer.

---

## Table of Contents

1. [Design Principles](#design-principles)
2. [System Overview](#system-overview)
3. [Module Architecture](#module-architecture)
4. [Data Flow](#data-flow)
5. [Class Diagrams](#class-diagrams)
6. [Design Decisions](#design-decisions)
7. [Extension Points](#extension-points)
8. [Error Handling Strategy](#error-handling-strategy)
9. [Testing Strategy](#testing-strategy)

---

## Design Principles

1. **Modularity** - Each domain concern is isolated in its own module with a clean public API.
2. **Composability** - Components like `IndicatorPipeline` and `BacktestEngine` can be chained and combined.
3. **Testability** - Pure functions for calculations; dependency injection for configurable behavior.
4. **Type Safety** - Type hints throughout; dataclasses for structured data.
5. **Defensive Programming** - Input validation at module boundaries with clear error messages.
6. **Zero External API Dependency** - All data is synthetic; no network calls required.

---

## System Overview

```
User Input (CLI) / Python API
         |
         v
+-----------------------------+
|  Data Generation Layer      |  market_data.py
|  - OHLCVGenerator           |
|  - MultiAssetGenerator      |
+-----------------------------+
         |
         v
+-----------------------------+
|  Analysis Layer             |
|  - indicators.py            |  Technical indicators
|  - forecaster.py            |  Time series forecasting
|  - backtester.py            |  Strategy backtesting
+-----------------------------+
         |
         v
+-----------------------------+
|  Output Layer               |
|  - visualizer.py            |  Chart generation
|  - reporter.py              |  Report generation
+-----------------------------+
```

---

## Module Architecture

### 1. `market_data.py` - Data Generation Layer

**Responsibility**: Generate synthetic financial market data.

**Classes**:
- `OHLCVGenerator` - Single-asset OHLCV data generator using geometric Brownian motion with extensions
- `MultiAssetGenerator` - Multi-asset correlated data generator using Cholesky decomposition

**Key Design**:
- Uses `np.random.default_rng` for reproducible random number generation
- SDE-based price generation: `dS = mu*S*dt + sigma*S*dW + mean_reversion + trend`
- OHLC derivation from close with intraperiod noise
- Supports CSV and Parquet serialization

**Input Validation**:
- `periods > 0`
- `start_price > 0`
- `volatility > 0`

---

### 2. `indicators.py` - Technical Indicators

**Responsibility**: Compute financial technical indicators on price/OHLCV data.

**Functions**:
- `sma()` - Simple Moving Average
- `ema()` - Exponential Moving Average
- `rsi()` - Relative Strength Index (Wilder smoothing)
- `macd()` - Moving Average Convergence Divergence
- `bollinger_bands()` - Bollinger Bands with bandwidth and %B
- `atr()` - Average True Range
- `stochastic()` - Stochastic Oscillator (%K, %D)

**Classes**:
- `IndicatorPipeline` - Compose multiple indicators into a single operation

**Key Design**:
- Pure functions operating on `pd.Series`/`pd.DataFrame`
- Consistent `_validate_series()` helper with minimum length checks
- Pipeline pattern for batch indicator computation
- All indicators return NaN for insufficient lookback data

---

### 3. `forecaster.py` - Time Series Forecasting

**Responsibility**: Fit models and generate forecasts with confidence intervals.

**Classes**:
- `ForecastResult` - Dataclass for forecast outputs
- `NaiveDriftForecaster` - Mean return extrapolation
- `PolynomialTrendForecaster` - Polynomial least-squares fitting
- `MovingAverageForecaster` - Rolling mean trend extrapolation

**Utility**:
- `walk_forward_validation()` - Cross-validation across time folds

**Key Design**:
- All forecasters implement `fit()` and `forecast()` methods
- Confidence intervals derived from residual standard deviation
- Metrics (MAE, RMSE, MAPE) computed on fitted values
- Walk-forward validation for model comparison

---

### 4. `backtester.py` - Strategy Backtesting

**Responsibility**: Simulate trading strategies on historical data.

**Classes**:
- `Trade` - Single trade record (entry/exit, PnL)
- `BacktestResult` - Comprehensive backtest statistics
- `BacktestEngine` - Event-driven backtesting simulation

**Built-in Strategies**:
- `buy_and_hold_signal()` - Simple benchmark
- `sma_crossover_signal()` - SMA crossover with configurable windows
- `rsi_strategy_signal()` - RSI mean-reversion

**Key Design**:
- Event-driven loop: signal -> entry/exit -> equity update
- Commission and slippage applied on each trade
- Open positions force-closed at end of data
- Comprehensive metrics: Sharpe, Sortino, max drawdown, profit factor

---

### 5. `visualizer.py` - Chart Generation

**Responsibility**: Generate matplotlib charts for analysis and reporting.

**Functions**:
- `plot_ohlc()` - Candlestick-style OHLC with volume
- `plot_with_indicators()` - Price with indicator overlays
- `plot_macd()` - Dedicated MACD chart
- `plot_rsi()` - RSI with overbought/oversold zones
- `plot_bollinger()` - Bollinger Band visualization
- `plot_forecast()` - Historical + forecast with confidence bands
- `plot_backtest_equity()` - Equity curve with trade markers
- `plot_drawdown()` - Drawdown chart
- `plot_correlation_heatmap()` - Asset correlation matrix
- `generate_all_charts()` - Batch chart generation

**Key Design**:
- Seaborn whitegrid style for professional appearance
- Consistent `_save_or_show()` pattern
- Optional `save_path` for headless/batch usage
- GridSpec layouts for multi-panel charts

---

### 6. `reporter.py` - Report Generation

**Responsibility**: Generate human-readable and machine-readable reports.

**Classes**:
- `HTMLReporter` - Styled HTML report builder with sections, tables, charts
- `JSONReporter` - JSON serialization for programmatic access

**Utility**:
- `generate_summary_report()` - End-to-end report from analysis results

**Key Design**:
- Fluent API: `add_section()`, `add_metrics()`, `add_dataframe()`
- Embedded CSS for self-contained HTML
- Chart embedding via `<img>` tags
- JSON serialization handles pandas types and dataclasses

---

### 7. `cli.py` - Command-Line Interface

**Responsibility**: Provide a unified CLI for all functionality.

**Subcommands**:
- `generate` - Data generation
- `indicators` - Indicator computation
- `forecast` - Forecasting
- `backtest` - Backtesting
- `report` - Full report generation

**Key Design**:
- argparse with typed arguments
- Central error handling with logging
- Modular command functions map 1:1 to subcommands

---

## Data Flow

```
[OHLCVGenerator] --(DataFrame)--> [indicators.py] --(Annotated DF)--> [visualizer.py]
                                         |
                                         v
                                [backtester.py] --(BacktestResult)--> [reporter.py]
                                         |
                                         v
                                [forecaster.py] --(ForecastResult)--> [reporter.py]
```

1. **Data Generation** produces a standard OHLCV DataFrame
2. **Indicators** enrich the DataFrame with computed columns
3. **Backtester** consumes the DataFrame to simulate trading
4. **Forecaster** fits models on the close price series
5. **Visualizer** generates charts from any of the above
6. **Reporter** aggregates all results into a final HTML report

---

## Class Diagrams

### Backtest Engine

```
+------------------+        +------------------+
| BacktestEngine   |<>------| Trade            |
+------------------+        +------------------+
| -initial_capital |        | -entry_time      |
| -commission      |        | -exit_time       |
| -slippage        |        | -entry_price     |
| -position_size   |        | -exit_price      |
+------------------+        | -position        |
| +run(df, signal) |        | -size            |
| -_build_result() |        | -pnl             |
+------------------+        | -return_pct      |
         |                  +------------------+
         v
+------------------+
| BacktestResult   |
+------------------+
| -strategy_name   |
| -initial_capital |
| -final_capital   |
| -total_trades    |
| -winning_trades  |
| -sharpe_ratio    |
| -max_drawdown    |
| -equity_curve    |
| -trades          |
+------------------+
```

### Forecaster Pattern

```
         +------------------+
         | <<abstract>>     |
         | BaseForecaster   |
         +------------------+
         | +fit(series)     |
         | +forecast(horiz) |
         +--------+---------+
                  |
     +------------+------------+
     |            |            |
+----v-----+ +----v------+ +---v-------+
| Naive    | | Polynomial| | Moving    |
| Drift    | | Trend     | | Average   |
+----------+ +-----------+ +-----------+
```

---

## Design Decisions

### Why Synthetic Data?
- No API keys or network dependencies required
- Reproducible results for testing and demos
- Fine-grained control over market conditions (trend, volatility, mean reversion)

### Why Pure Functions for Indicators?
- Easier to test and debug
- No hidden state or side effects
- Can be composed freely in pipelines

### Why Event-Driven Backtesting?
- More realistic than vectorized backtesting
- Proper handling of intraday signals
- Easier to extend with stop-losses, take-profits, etc.

### Why Multiple Forecasters?
- Different market regimes favor different models
- Baseline comparison is essential for forecasting
- Simple models are more robust than complex ones for short horizons

---

## Extension Points

1. **New Indicators**: Add functions to `indicators.py` and register in `IndicatorPipeline`
2. **New Forecasters**: Implement `fit()`/`forecast()` pattern, add to `cli.py` model_map
3. **New Strategies**: Write a signal function `fn(df, idx) -> int` and pass to `BacktestEngine.run()`
4. **New Charts**: Add plotting functions to `visualizer.py` following the `_save_or_show` pattern
5. **Real Data**: Subclass `OHLCVGenerator` with API client integration

---

## Error Handling Strategy

- **Type checking** at module boundaries (`isinstance` checks)
- **Value validation** for parameters (range checks, length checks)
- **RuntimeError** for invalid state (calling forecast before fit)
- **FileNotFoundError** for missing data files
- **Logging** at appropriate levels (DEBUG for internals, INFO for progress, WARNING for issues)
- **CLI** catches exceptions and prints user-friendly messages

---

## Testing Strategy

- **Unit tests** for each module (isolated with fixtures)
- **Property-based tests** for indicator ranges (e.g., RSI in [0, 100])
- **Integration tests** for CLI commands
- **Edge case coverage**: empty data, single value, constant series
- **Reproducibility tests**: seeded random generators produce identical output
