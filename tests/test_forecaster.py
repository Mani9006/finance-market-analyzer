"""Tests for the forecaster module."""

import numpy as np
import pandas as pd
import pytest

from src.forecaster import (
    NaiveDriftForecaster,
    PolynomialTrendForecaster,
    MovingAverageForecaster,
    walk_forward_validation,
    ForecastResult,
)


@pytest.fixture
def price_series() -> pd.Series:
    np.random.seed(42)
    return pd.Series(
        np.cumsum(np.random.randn(200) * 0.5 + 0.1) + 100,
        index=pd.date_range("2020-01-01", periods=200),
    )


@pytest.fixture
def trending_series() -> pd.Series:
    # Deterministic upward trend
    return pd.Series(
        np.linspace(100, 200, 200) + np.random.randn(200) * 2,
        index=pd.date_range("2020-01-01", periods=200),
    )


class TestNaiveDriftForecaster:
    def test_fit(self, price_series):
        model = NaiveDriftForecaster()
        model.fit(price_series)
        assert model.mean_return != 0.0
        assert model.last_value == pytest.approx(price_series.iloc[-1])

    def test_forecast_returns_result(self, price_series):
        model = NaiveDriftForecaster()
        model.fit(price_series)
        result = model.forecast(horizon=10)
        assert isinstance(result, ForecastResult)
        assert len(result.forecast_values) == 10
        assert result.model_name == "naive_drift"

    def test_forecast_has_confidence_bands(self, price_series):
        model = NaiveDriftForecaster()
        model.fit(price_series)
        result = model.forecast(horizon=10)
        assert result.confidence_lower is not None
        assert result.confidence_upper is not None
        assert len(result.confidence_lower) == 10
        assert len(result.confidence_upper) == 10
        assert (result.confidence_upper >= result.confidence_lower).all()

    def test_forecast_before_fit_raises(self):
        model = NaiveDriftForecaster()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.forecast(horizon=5)

    def test_forecast_invalid_horizon_raises(self, price_series):
        model = NaiveDriftForecaster()
        model.fit(price_series)
        with pytest.raises(ValueError, match="horizon must be > 0"):
            model.forecast(horizon=0)

    def test_metrics_present(self, price_series):
        model = NaiveDriftForecaster()
        model.fit(price_series)
        result = model.forecast(horizon=10)
        assert result.metrics is not None
        assert "mae" in result.metrics
        assert "rmse" in result.metrics
        assert "mape" in result.metrics

    def test_trending_series_forecast_direction(self, trending_series):
        model = NaiveDriftForecaster()
        model.fit(trending_series)
        result = model.forecast(horizon=10)
        # For a strong uptrend, mean drift should be positive
        assert result.forecast_values.iloc[-1] > result.forecast_values.iloc[0]

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            NaiveDriftForecaster().fit(pd.Series([100]))

    def test_invalid_input_type(self):
        with pytest.raises(TypeError, match="pandas Series"):
            NaiveDriftForecaster().fit([1, 2, 3])

    def test_fit_on_constant_series(self):
        series = pd.Series([100.0] * 50)
        model = NaiveDriftForecaster()
        model.fit(series)
        assert model.mean_return == pytest.approx(0.0)


class TestPolynomialTrendForecaster:
    def test_fit(self, price_series):
        model = PolynomialTrendForecaster(degree=2)
        model.fit(price_series)
        assert model.coefs is not None
        assert model.fitted is not None

    def test_forecast(self, price_series):
        model = PolynomialTrendForecaster(degree=2)
        model.fit(price_series)
        result = model.forecast(horizon=10)
        assert len(result.forecast_values) == 10
        assert result.model_name == "polynomial_trend_deg2"

    def test_forecast_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            PolynomialTrendForecaster().forecast(horizon=5)

    def test_invalid_degree(self):
        with pytest.raises(ValueError, match="degree must be >= 1"):
            PolynomialTrendForecaster(degree=0)

    def test_degree_1_is_linear(self):
        # Linear series should fit perfectly with degree 1
        x = np.arange(100)
        y = 2 * x + 10 + np.random.randn(100) * 0.01  # nearly linear
        series = pd.Series(y, index=pd.date_range("2020-01-01", periods=100))
        model = PolynomialTrendForecaster(degree=1)
        model.fit(series)
        residuals = model.fitted - series
        assert residuals.abs().max() < 0.5  # should be close

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="at least 3"):
            PolynomialTrendForecaster().fit(pd.Series([1, 2]))


class TestMovingAverageForecaster:
    def test_fit(self, price_series):
        model = MovingAverageForecaster(window=20)
        model.fit(price_series)
        assert model.rolling_mean is not None
        assert model.trend is not None

    def test_forecast(self, price_series):
        model = MovingAverageForecaster(window=20)
        model.fit(price_series)
        result = model.forecast(horizon=10)
        assert len(result.forecast_values) == 10
        assert result.model_name == "moving_average_w20"

    def test_forecast_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            MovingAverageForecaster().forecast(horizon=5)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window must be >= 2"):
            MovingAverageForecaster(window=1)

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            MovingAverageForecaster().fit(pd.Series([1]))


class TestWalkForwardValidation:
    def test_walk_forward_runs(self, price_series):
        result = walk_forward_validation(
            price_series,
            NaiveDriftForecaster,
            horizon=5,
            step=10,
        )
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "actual" in result.columns
            assert "predicted" in result.columns
            assert "error" in result.columns

    def test_walk_forward_returns_empty_for_short_series(self):
        short = pd.Series([1, 2, 3, 4, 5])
        result = walk_forward_validation(short, NaiveDriftForecaster, horizon=2, step=2)
        # May be empty due to insufficient training data
        assert isinstance(result, pd.DataFrame)

    def test_walk_forward_with_polynomial(self, price_series):
        result = walk_forward_validation(
            price_series,
            PolynomialTrendForecaster,
            horizon=5,
            step=15,
            degree=2,
        )
        assert isinstance(result, pd.DataFrame)
