"""Technical indicators for financial time series analysis."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _validate_series(series: pd.Series, min_length: int = 1) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    if len(series) < min_length:
        raise ValueError(f"Series must have at least {min_length} observations")
    if series.isnull().all():
        raise ValueError("Series contains only NaN values")
    return series


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average.

    Args:
        series: Price series.
        window: Rolling window size.

    Returns:
        SMA series aligned with input index.
    """
    series = _validate_series(series, window)
    result = series.rolling(window=window, min_periods=window).mean()
    logger.debug("SMA computed with window=%s", window)
    return result


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average.

    Args:
        series: Price series.
        span: EMA span (decay center of mass).

    Returns:
        EMA series aligned with input index.
    """
    series = _validate_series(series, span)
    result = series.ewm(span=span, adjust=False, min_periods=span).mean()
    logger.debug("EMA computed with span=%s", span)
    return result


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    Args:
        series: Price series (typically closing prices).
        window: Lookback window for average gains/losses.

    Returns:
        RSI series ranging 0-100.
    """
    series = _validate_series(series, window + 1)
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Wilder smoothing continuation
    for i in range(window, len(avg_gain)):
        if pd.notna(avg_gain.iloc[i - 1]):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (window - 1) + gain.iloc[i]) / window
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (window - 1) + loss.iloc[i]) / window

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    logger.debug("RSI computed with window=%s", window)
    return rsi_series


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Moving Average Convergence Divergence (MACD).

    Args:
        series: Price series.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal line EMA span.

    Returns:
        Dictionary with keys: macd, signal, histogram.
    """
    if fast >= slow:
        raise ValueError("fast span must be smaller than slow span")
    series = _validate_series(series, slow)
    ema_fast = ema(series, span=fast)
    ema_slow = ema(series, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, span=signal)
    histogram = macd_line - signal_line
    logger.debug("MACD computed fast=%s slow=%s signal=%s", fast, slow, signal)
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> dict[str, pd.Series]:
    """Bollinger Bands.

    Args:
        series: Price series.
        window: SMA window.
        num_std: Number of standard deviations for band width.

    Returns:
        Dictionary with keys: middle (SMA), upper, lower, bandwidth, percent_b.
    """
    series = _validate_series(series, window)
    middle = sma(series, window=window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    logger.debug("Bollinger Bands computed window=%s num_std=%s", window, num_std)
    return {
        "middle": middle,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b,
    }


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (ATR).

    Args:
        df: DataFrame with columns high, low, close.
        window: Rolling window.

    Returns:
        ATR series.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")
    if len(df) < window + 1:
        raise ValueError(f"DataFrame must have at least {window + 1} rows")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = true_range.rolling(window=window, min_periods=window).mean()
    logger.debug("ATR computed window=%s", window)
    return atr_series


def stochastic(
    df: pd.DataFrame,
    window: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> dict[str, pd.Series]:
    """Stochastic Oscillator (%K and %D).

    Args:
        df: DataFrame with columns high, low, close.
        window: Lookback window for highs/lows.
        smooth_k: Smoothing window for %K.
        smooth_d: Smoothing window for %D.

    Returns:
        Dictionary with keys: k, d.
    """
    required = {"high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")
    if len(df) < window:
        raise ValueError(f"DataFrame must have at least {window} rows")

    lowest_low = df["low"].rolling(window=window, min_periods=window).min()
    highest_high = df["high"].rolling(window=window, min_periods=window).max()
    k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    k = k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    logger.debug("Stochastic computed window=%s", window)
    return {"k": k, "d": d}


class IndicatorPipeline:
    """Compose multiple indicators into a single annotated DataFrame."""

    def __init__(self, indicators: Optional[dict[str, dict]] = None):
        """Args:
            indicators: Mapping of indicator name to kwargs dict.
                        e.g., {"sma_20": {"func": "sma", "column": "close", "window": 20}}
        """
        self.indicators = indicators or {}

    def add(self, name: str, func_name: str, column: str, **kwargs) -> "IndicatorPipeline":
        self.indicators[name] = {"func": func_name, "column": column, **kwargs}
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured indicators to a DataFrame.

        Returns:
            New DataFrame with added indicator columns.
        """
        result = df.copy()
        for name, cfg in self.indicators.items():
            func_name = cfg.pop("func")
            column = cfg.pop("column")
            func = globals()[func_name]
            series_or_dict = func(result[column], **cfg) if func_name not in ("atr", "stochastic") else func(result, **cfg)
            if isinstance(series_or_dict, dict):
                for sub_name, sub_series in series_or_dict.items():
                    result[f"{name}_{sub_name}"] = sub_series
            else:
                result[name] = series_or_dict
        logger.info("IndicatorPipeline applied %s indicators", len(self.indicators))
        return result
