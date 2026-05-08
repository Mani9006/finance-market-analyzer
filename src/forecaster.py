"""Time series forecasting models for financial market data."""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast results."""

    model_name: str
    fitted_values: pd.Series
    forecast_values: pd.Series
    forecast_index: pd.DatetimeIndex
    confidence_lower: Optional[pd.Series] = None
    confidence_upper: Optional[pd.Series] = None
    residuals: Optional[pd.Series] = None
    metrics: Optional[dict[str, float]] = None


class NaiveDriftForecaster:
    """Simple drift-based forecaster using historical mean return."""

    def __init__(self):
        self.mean_return: float = 0.0
        self.std_return: float = 0.0
        self.last_value: float = 0.0
        self.fitted: Optional[pd.Series] = None

    def fit(self, series: pd.Series) -> "NaiveDriftForecaster":
        """Fit the model on a price series."""
        series = self._validate(series)
        returns = series.pct_change().dropna()
        self.mean_return = float(returns.mean())
        self.std_return = float(returns.std())
        self.last_value = float(series.iloc[-1])
        # Fitted = forward-shifted prediction from each prior point
        self.fitted = series.shift(1) * (1 + self.mean_return)
        logger.info("NaiveDrift fitted: mean_return=%.6f std=%.6f", self.mean_return, self.std_return)
        return self

    def forecast(
        self,
        horizon: int = 30,
        freq: str = "D",
        confidence: float = 0.95,
    ) -> ForecastResult:
        """Generate forecasts.

        Args:
            horizon: Number of periods to forecast.
            freq: Frequency string for forecast index.
            confidence: Confidence level for bands.

        Returns:
            ForecastResult with predictions and confidence bands.
        """
        if self.fitted is None:
            raise RuntimeError("Model must be fitted before forecasting")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        z = 1.96 if confidence >= 0.95 else 1.645
        forecasts = np.zeros(horizon)
        lower = np.zeros(horizon)
        upper = np.zeros(horizon)
        current = self.last_value

        for h in range(horizon):
            current = current * (1 + self.mean_return)
            se = self.std_return * math.sqrt(h + 1) * current
            forecasts[h] = current
            lower[h] = current - z * se
            upper[h] = current + z * se

        last_date = self.fitted.index[-1]
        forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        forecast_series = pd.Series(forecasts, index=forecast_index)
        lower_series = pd.Series(lower, index=forecast_index)
        upper_series = pd.Series(upper, index=forecast_index)
        residuals = self.fitted - self.fitted.shift(1)  # approximate

        metrics = self._compute_metrics(self.fitted.dropna(), residuals.dropna())

        return ForecastResult(
            model_name="naive_drift",
            fitted_values=self.fitted,
            forecast_values=forecast_series,
            forecast_index=forecast_index,
            confidence_lower=lower_series,
            confidence_upper=upper_series,
            residuals=residuals,
            metrics=metrics,
        )

    @staticmethod
    def _validate(series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series")
        if len(series) < 2:
            raise ValueError("Series must have at least 2 observations")
        return series

    @staticmethod
    def _compute_metrics(actual: pd.Series, residual: pd.Series) -> dict[str, float]:
        mae = float(residual.abs().mean()) if len(residual) > 0 else np.nan
        rmse = float(math.sqrt((residual**2).mean())) if len(residual) > 0 else np.nan
        mape = float((residual.abs() / actual.replace(0, np.nan)).mean() * 100) if len(actual) > 0 else np.nan
        return {"mae": mae, "rmse": rmse, "mape": mape}


class PolynomialTrendForecaster:
    """Polynomial trend forecaster using least-squares regression."""

    def __init__(self, degree: int = 2):
        if degree < 1:
            raise ValueError("degree must be >= 1")
        self.degree = degree
        self.coefs: Optional[np.ndarray] = None
        self.last_t: int = 0
        self.fitted: Optional[pd.Series] = None
        self._index: Optional[pd.DatetimeIndex] = None

    def fit(self, series: pd.Series) -> "PolynomialTrendForecaster":
        """Fit polynomial trend to series."""
        series = self._validate(series)
        x = np.arange(len(series), dtype=float)
        y = series.values.astype(float)
        self.coefs, *_ = np.polynomial.polynomial.polyfit(x, y, self.degree, full=True)
        self.last_t = len(series) - 1
        self._index = series.index
        fitted_vals = np.polynomial.polynomial.polyval(x, self.coefs)
        self.fitted = pd.Series(fitted_vals, index=series.index)
        logger.info("PolynomialTrend fitted: degree=%s", self.degree)
        return self

    def forecast(
        self,
        horizon: int = 30,
        freq: str = "D",
        confidence: float = 0.95,
    ) -> ForecastResult:
        """Generate forecasts based on polynomial trend."""
        if self.coefs is None or self.fitted is None:
            raise RuntimeError("Model must be fitted before forecasting")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        future_x = np.arange(self.last_t + 1, self.last_t + 1 + horizon, dtype=float)
        forecasts = np.polynomial.polynomial.polyval(future_x, self.coefs)

        # Simple residual-based confidence bands
        residuals = self.fitted.dropna().values - np.polynomial.polynomial.polyval(
            np.arange(len(self.fitted)), self.coefs
        )
        std_resid = float(np.std(residuals))
        z = 1.96 if confidence >= 0.95 else 1.645

        last_date = self._index[-1]  # type: ignore[index]
        forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        forecast_series = pd.Series(forecasts, index=forecast_index)
        lower_series = pd.Series(forecasts - z * std_resid * np.sqrt(np.arange(1, horizon + 1)), index=forecast_index)
        upper_series = pd.Series(forecasts + z * std_resid * np.sqrt(np.arange(1, horizon + 1)), index=forecast_index)

        res_series = pd.Series(residuals, index=self.fitted.index)
        metrics = NaiveDriftForecaster._compute_metrics(self.fitted, res_series)

        return ForecastResult(
            model_name=f"polynomial_trend_deg{self.degree}",
            fitted_values=self.fitted,
            forecast_values=forecast_series,
            forecast_index=forecast_index,
            confidence_lower=lower_series,
            confidence_upper=upper_series,
            residuals=res_series,
            metrics=metrics,
        )

    @staticmethod
    def _validate(series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series")
        if len(series) < 3:
            raise ValueError("Series must have at least 3 observations")
        return series


class MovingAverageForecaster:
    """Moving average crossover forecaster: predicts using rolling mean continuation."""

    def __init__(self, window: int = 20):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.rolling_mean: Optional[float] = None
        self.trend: Optional[float] = None
        self.fitted: Optional[pd.Series] = None
        self._index: Optional[pd.DatetimeIndex] = None

    def fit(self, series: pd.Series) -> "MovingAverageForecaster":
        """Fit using recent rolling mean and trend of the mean."""
        series = self._validate(series)
        roll = series.rolling(window=self.window, min_periods=self.window).mean()
        self.rolling_mean = float(roll.iloc[-1])
        # Trend = difference between last two rolling means
        valid = roll.dropna()
        if len(valid) >= 2:
            self.trend = float(valid.iloc[-1] - valid.iloc[-2])
        else:
            self.trend = 0.0
        self.fitted = roll
        self._index = series.index
        logger.info("MovingAverageForecaster fitted: window=%s trend=%.4f", self.window, self.trend)
        return self

    def forecast(
        self,
        horizon: int = 30,
        freq: str = "D",
        confidence: float = 0.95,
    ) -> ForecastResult:
        """Generate forecasts."""
        if self.rolling_mean is None or self.fitted is None:
            raise RuntimeError("Model must be fitted before forecasting")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")

        z = 1.96 if confidence >= 0.95 else 1.645
        # Extrapolate rolling mean with trend
        forecasts = np.array([self.rolling_mean + self.trend * (h + 1) for h in range(horizon)])

        # Confidence based on historical residual std
        residuals = self.fitted.dropna().diff().dropna()
        std_resid = float(residuals.std()) if len(residuals) > 0 else 1.0

        last_date = self._index[-1]  # type: ignore[index]
        forecast_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        forecast_series = pd.Series(forecasts, index=forecast_index)
        lower_series = pd.Series(forecasts - z * std_resid * np.sqrt(np.arange(1, horizon + 1)), index=forecast_index)
        upper_series = pd.Series(forecasts + z * std_resid * np.sqrt(np.arange(1, horizon + 1)), index=forecast_index)

        res_series = self.fitted.dropna().diff().dropna()
        metrics = NaiveDriftForecaster._compute_metrics(self.fitted.dropna(), res_series)

        return ForecastResult(
            model_name=f"moving_average_w{self.window}",
            fitted_values=self.fitted,
            forecast_values=forecast_series,
            forecast_index=forecast_index,
            confidence_lower=lower_series,
            confidence_upper=upper_series,
            residuals=res_series,
            metrics=metrics,
        )

    @staticmethod
    def _validate(series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("Input must be a pandas Series")
        if len(series) < 2:
            raise ValueError("Series must have at least 2 observations")
        return series


def walk_forward_validation(
    series: pd.Series,
    forecaster_class: type,
    horizon: int = 5,
    step: int = 5,
    **forecaster_kwargs,
) -> pd.DataFrame:
    """Perform walk-forward validation on a forecaster.

    Args:
        series: Full time series.
        forecaster_class: Class implementing fit() and forecast().
        horizon: Forecast horizon per fold.
        step: Step size between folds.
        **forecaster_kwargs: Constructor args for the forecaster.

    Returns:
        DataFrame with columns: fold, actual, predicted, error.
    """
    results = []
    min_train = max(forecaster_kwargs.get("window", 20), forecaster_kwargs.get("degree", 2)) + 10
    for start in range(min_train, len(series) - horizon, step):
        train = series.iloc[:start]
        actual = series.iloc[start : start + horizon]
        try:
            model = forecaster_class(**forecaster_kwargs)
            model.fit(train)
            fc = model.forecast(horizon=horizon)
            pred = fc.forecast_values.reindex(actual.index)
            for i, (idx, a) in enumerate(actual.items()):
                p = pred.iloc[i] if i < len(pred) and pd.notna(pred.iloc[i]) else np.nan
                results.append({
                    "fold": start,
                    "timestamp": idx,
                    "actual": a,
                    "predicted": p,
                    "error": a - p if pd.notna(p) else np.nan,
                })
        except Exception as e:
            logger.warning("Fold %s failed: %s", start, e)
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        mae = float(df["error"].abs().mean())
        rmse = float(math.sqrt((df["error"] ** 2).mean()))
        logger.info("Walk-forward validation: MAE=%.4f RMSE=%.4f over %s folds", mae, rmse, df["fold"].nunique())
    return df
