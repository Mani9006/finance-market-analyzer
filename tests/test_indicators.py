"""Tests for the indicators module."""

import numpy as np
import pandas as pd
import pytest

from src import indicators as ind


@pytest.fixture
def price_series() -> pd.Series:
    np.random.seed(42)
    return pd.Series(np.cumsum(np.random.randn(200) * 0.5 + 0.1) + 100, index=pd.date_range("2020-01-01", periods=200))


@pytest.fixture
def ohlc_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 200
    close = np.cumsum(np.random.randn(n) * 0.5 + 0.1) + 100
    noise = np.random.rand(n) * 2
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=pd.date_range("2020-01-01", periods=n),
    )


class TestSMA:
    def test_sma_basic(self, price_series):
        result = ind.sma(price_series, window=20)
        assert isinstance(result, pd.Series)
        assert result.isna().sum() == 19  # first 19 are NaN
        assert result.notna().sum() == 181

    def test_sma_values(self, price_series):
        result = ind.sma(price_series, window=20)
        expected = price_series.rolling(20).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_sma_window_larger_than_series_raises(self, price_series):
        with pytest.raises(ValueError, match="at least"):
            ind.sma(price_series.iloc[:10], window=20)

    def test_sma_invalid_input_type(self):
        with pytest.raises(TypeError, match="pandas Series"):
            ind.sma([1, 2, 3], window=2)

    def test_sma_all_nan_raises(self):
        with pytest.raises(ValueError, match="only NaN"):
            ind.sma(pd.Series([np.nan, np.nan]), window=1)


class TestEMA:
    def test_ema_basic(self, price_series):
        result = ind.ema(price_series, span=20)
        assert isinstance(result, pd.Series)
        assert result.isna().sum() == 19

    def test_ema_responds_faster_than_sma(self, price_series):
        ema = ind.ema(price_series, span=20)
        sma = ind.sma(price_series, window=20)
        # EMA should generally differ from SMA
        valid = ema.notna() & sma.notna()
        assert not ema[valid].equals(sma[valid])

    def test_ema_window_larger_than_series_raises(self, price_series):
        with pytest.raises(ValueError, match="at least"):
            ind.ema(price_series.iloc[:10], span=20)


class TestRSI:
    def test_rsi_range(self, price_series):
        result = ind.rsi(price_series, window=14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_strong_uptrend(self):
        # Strong uptrend -> RSI should be high
        np.random.seed(42)
        uptrend = pd.Series(np.linspace(100, 200, 200) + np.random.randn(200) * 2)
        result = ind.rsi(uptrend, window=14)
        assert result.dropna().iloc[-1] > 50

    def test_rsi_strong_downtrend(self):
        # Strong downtrend -> RSI should be low
        downtrend = pd.Series(np.linspace(200, 100, 50))
        result = ind.rsi(downtrend, window=14)
        assert result.dropna().iloc[-1] < 50

    def test_rsi_window_larger_than_series_raises(self, price_series):
        with pytest.raises(ValueError, match="at least"):
            ind.rsi(price_series.iloc[:10], window=14)


class TestMACD:
    def test_macd_returns_dict(self, price_series):
        result = ind.macd(price_series)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"macd", "signal", "histogram"}

    def test_macd_histogram_is_difference(self, price_series):
        result = ind.macd(price_series)
        expected_hist = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], expected_hist)

    def test_macd_fast_lt_slow_required(self, price_series):
        with pytest.raises(ValueError, match="fast span must be smaller"):
            ind.macd(price_series, fast=30, slow=20)

    def test_macd_fast_equals_slow_raises(self, price_series):
        with pytest.raises(ValueError, match="fast span must be smaller"):
            ind.macd(price_series, fast=20, slow=20)


class TestBollingerBands:
    def test_bb_returns_dict(self, price_series):
        result = ind.bollinger_bands(price_series, window=20)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"middle", "upper", "lower", "bandwidth", "percent_b"}

    def test_upper_gte_middle_gte_lower(self, price_series):
        bb = ind.bollinger_bands(price_series, window=20)
        valid = bb["upper"].notna()
        assert (bb["upper"][valid] >= bb["middle"][valid]).all()
        assert (bb["middle"][valid] >= bb["lower"][valid]).all()

    def test_percent_b_range(self, price_series):
        bb = ind.bollinger_bands(price_series, window=20)
        pb = bb["percent_b"].dropna()
        # percent_b can exceed 0-1 but should be finite
        assert np.isfinite(pb).all()


class TestATR:
    def test_atr_positive(self, ohlc_df):
        result = ind.atr(ohlc_df, window=14)
        assert isinstance(result, pd.Series)
        assert (result.dropna() > 0).all()

    def test_atr_missing_columns_raises(self):
        with pytest.raises(ValueError, match="must contain columns"):
            ind.atr(pd.DataFrame({"close": [1, 2, 3]}))


class TestStochastic:
    def test_stochastic_returns_dict(self, ohlc_df):
        result = ind.stochastic(ohlc_df, window=14)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"k", "d"}

    def test_k_d_range(self, ohlc_df):
        result = ind.stochastic(ohlc_df, window=14)
        for key in ["k", "d"]:
            s = result[key].dropna()
            assert (s >= 0).all() and (s <= 100).all()


class TestIndicatorPipeline:
    def test_pipeline_add_and_apply(self, ohlc_df):
        pipeline = ind.IndicatorPipeline()
        pipeline.add("sma_20", "sma", "close", window=20)
        pipeline.add("rsi_14", "rsi", "close", window=14)
        result = pipeline.apply(ohlc_df)
        assert "sma_20" in result.columns
        assert "rsi_14" in result.columns
        assert len(result) == len(ohlc_df)

    def test_pipeline_empty(self, ohlc_df):
        pipeline = ind.IndicatorPipeline()
        result = pipeline.apply(ohlc_df)
        assert list(result.columns) == list(ohlc_df.columns)
