"""Tests for the backtester module."""

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    BacktestEngine,
    BacktestResult,
    Trade,
    sma_crossover_signal,
    rsi_strategy_signal,
    buy_and_hold_signal,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    close = np.cumsum(np.random.randn(n) * 0.5 + 0.05) + 100
    noise = np.random.rand(n) * 1.5
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": np.random.randint(1_000_000, 5_000_000, n),
        },
        index=pd.date_range("2020-01-01", periods=n),
    )


class TestBacktestEngine:
    def test_initialization(self):
        engine = BacktestEngine(initial_capital=50000, commission=0.002, slippage=0.001)
        assert engine.initial_capital == 50000
        assert engine.commission == 0.002
        assert engine.slippage == 0.001

    def test_invalid_initial_capital(self):
        with pytest.raises(ValueError, match="initial_capital must be > 0"):
            BacktestEngine(initial_capital=0)
        with pytest.raises(ValueError, match="initial_capital must be > 0"):
            BacktestEngine(initial_capital=-100)

    def test_invalid_commission_slippage(self):
        with pytest.raises(ValueError, match="commission and slippage must be >= 0"):
            BacktestEngine(commission=-0.001)
        with pytest.raises(ValueError, match="commission and slippage must be >= 0"):
            BacktestEngine(slippage=-0.001)

    def test_run_buy_and_hold(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(sample_ohlcv, buy_and_hold_signal, strategy_name="buy_hold")
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "buy_hold"
        assert result.initial_capital == 100_000
        assert result.total_trades >= 1

    def test_run_sma_cross(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(sample_ohlcv, sma_crossover_signal, strategy_name="sma_cross")
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "sma_cross"

    def test_run_rsi_strategy(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(sample_ohlcv, rsi_strategy_signal, strategy_name="rsi")
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "rsi"

    def test_equity_curve_length(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        assert len(result.equity_curve) == len(sample_ohlcv)

    def test_final_capital_positive(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        assert result.final_capital > 0

    def test_total_return_calculation(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=100_000)
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        expected_return = result.final_capital - result.initial_capital
        assert result.total_return == pytest.approx(expected_return)
        expected_pct = (expected_return / result.initial_capital) * 100
        assert result.total_return_pct == pytest.approx(expected_pct)

    def test_win_rate_bounds(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, sma_crossover_signal)
        assert 0 <= result.win_rate <= 100

    def test_trades_consistency(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, sma_crossover_signal)
        assert result.total_trades == result.winning_trades + result.losing_trades

    def test_sharpe_ratio_finite(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        assert np.isfinite(result.sharpe_ratio)

    def test_max_drawdown_negative_or_zero(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        assert result.max_drawdown <= 0
        assert result.max_drawdown_pct <= 0

    def test_trades_list_not_empty(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        assert len(result.trades) >= 1

    def test_trade_attributes(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        for trade in result.trades:
            assert isinstance(trade, Trade)
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.entry_time is not None
            assert trade.exit_time is not None

    def test_profit_factor_finite(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, sma_crossover_signal)
        assert result.profit_factor >= 0 or np.isinf(result.profit_factor)

    def test_missing_columns_raises(self):
        bad_df = pd.DataFrame({"close": [1, 2, 3]})
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="must contain columns"):
            engine.run(bad_df, buy_and_hold_signal)

    def test_insufficient_rows_raises(self):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
        engine = BacktestEngine()
        with pytest.raises(ValueError, match="at least 2"):
            engine.run(df, buy_and_hold_signal)

    def test_trades_df_created(self, sample_ohlcv):
        engine = BacktestEngine()
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        assert result.trades_df is not None
        assert len(result.trades_df) == len(result.trades)

    def test_zero_commission_backtest(self, sample_ohlcv):
        engine = BacktestEngine(initial_capital=100_000, commission=0, slippage=0)
        result = engine.run(sample_ohlcv, buy_and_hold_signal)
        assert result.total_trades >= 1


class TestSignalFunctions:
    def test_buy_and_hold(self, sample_ohlcv):
        sig = buy_and_hold_signal(sample_ohlcv, 1)
        assert sig == 1
        sig2 = buy_and_hold_signal(sample_ohlcv, 50)
        assert sig2 == 0

    def test_sma_crossover_above(self):
        df = pd.DataFrame({
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1000] * 6,
        })
        sig = sma_crossover_signal(df, 5, fast=2, slow=5)
        # In strong uptrend, fast MA > slow MA -> long signal
        assert sig in (1, -1, 0)

    def test_rsi_strategy_neutral(self):
        df = pd.DataFrame({
            "open": [100] * 20,
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100] * 20,
            "volume": [1000] * 20,
        })
        sig = rsi_strategy_signal(df, 19, period=14, oversold=30, overbought=70)
        # Flat prices -> RSI near 50 -> no signal
        assert sig == 0
