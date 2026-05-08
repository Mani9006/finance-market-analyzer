import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OHLCVGenerator:
    """Synthetic OHLCV (Open, High, Low, Close, Volume) market data generator.

    Generates realistic synthetic price series using geometric Brownian motion
    with configurable drift, volatility, and optional trend/mean-reversion
    components.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        start_price: float = 100.0,
        drift: float = 0.0002,
        volatility: float = 0.02,
        trend_strength: float = 0.0,
        mean_reversion_speed: float = 0.0,
        mean_reversion_level: float = 100.0,
    ):
        self.seed = seed
        self.start_price = float(start_price)
        self.drift = float(drift)
        self.volatility = float(volatility)
        self.trend_strength = float(trend_strength)
        self.mean_reversion_speed = float(mean_reversion_speed)
        self.mean_reversion_level = float(mean_reversion_level)
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        periods: int = 252,
        freq: str = "D",
        start_date: Optional[datetime] = None,
        symbol: str = "SYNTH",
    ) -> pd.DataFrame:
        """Generate a synthetic OHLCV DataFrame.

        Args:
            periods: Number of periods to generate.
            freq: Pandas frequency string (e.g., 'D', 'H', '15min').
            start_date: First timestamp. Defaults to 1 year ago.
            symbol: Asset symbol for metadata.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
            Index is a DatetimeIndex.

        Raises:
            ValueError: If periods <= 0 or parameters are invalid.
        """
        if periods <= 0:
            raise ValueError("periods must be > 0")
        if self.start_price <= 0:
            raise ValueError("start_price must be > 0")
        if self.volatility <= 0:
            raise ValueError("volatility must be > 0")

        if start_date is None:
            start_date = datetime.now() - timedelta(days=periods)

        date_index = pd.date_range(start=start_date, periods=periods, freq=freq)

        close_prices = self._generate_close_series(periods)

        # Derive realistic high/low from close with intraperiod noise
        intraday_noise = self.rng.normal(loc=0.0, scale=self.volatility * 0.5, size=periods)
        high_prices = np.maximum(close_prices, close_prices * (1 + np.abs(intraday_noise) * 0.5))
        low_prices = np.minimum(close_prices, close_prices * (1 - np.abs(intraday_noise) * 0.5))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = self.start_price

        # Volume with randomness and mild correlation to volatility
        log_volumes = self.rng.normal(loc=math.log(1_000_000), scale=0.5, size=periods)
        volumes = np.exp(log_volumes).astype(int)

        df = pd.DataFrame(
            {
                "open": np.round(open_prices, 4),
                "high": np.round(high_prices, 4),
                "low": np.round(low_prices, 4),
                "close": np.round(close_prices, 4),
                "volume": volumes,
            },
            index=date_index,
        )
        df.attrs["symbol"] = symbol
        df.attrs["generated_at"] = datetime.now().isoformat()
        logger.info("Generated %s synthetic OHLCV rows for %s", periods, symbol)
        return df

    def _generate_close_series(self, periods: int) -> np.ndarray:
        """Generate a close price series via discretized SDE."""
        dt = 1.0  # assume unit time step
        prices = np.zeros(periods)
        prices[0] = self.start_price

        for t in range(1, periods):
            z = self.rng.standard_normal()
            prev = prices[t - 1]

            # Mean reversion contribution
            mr_term = self.mean_reversion_speed * (self.mean_reversion_level - prev) * dt

            # Trend contribution
            trend_term = self.trend_strength * math.sin(2 * math.pi * t / 50) * prev * dt

            # GBM core
            gbmu = (self.drift - 0.5 * self.volatility**2) * dt
            gbm_diffusion = self.volatility * math.sqrt(dt) * z

            new_price = prev * math.exp(gbmu + gbm_diffusion) + mr_term + trend_term
            prices[t] = max(new_price, 0.01)  # floor to avoid zero/negative
        return prices

    @staticmethod
    def save_to_csv(df: pd.DataFrame, path: Path) -> None:
        """Save a DataFrame to CSV, creating parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        logger.info("Saved market data to %s", path)

    @staticmethod
    def save_to_parquet(df: pd.DataFrame, path: Path) -> None:
        """Save a DataFrame to Parquet, creating parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info("Saved market data to %s", path)

    @staticmethod
    def load_from_csv(path: Path) -> pd.DataFrame:
        """Load a DataFrame from CSV with a DatetimeIndex."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info("Loaded market data from %s", path)
        return df


class MultiAssetGenerator:
    """Generate correlated synthetic data for multiple assets."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_portfolio(
        self,
        symbols: list[str],
        periods: int = 252,
        correlations: Optional[np.ndarray] = None,
        base_price: float = 100.0,
        start_date: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        """Generate a dictionary of correlated OHLCV DataFrames.

        Args:
            symbols: List of asset symbols.
            periods: Number of periods.
            correlations: Optional symmetric correlation matrix (len(symbols) x len(symbols)).
            base_price: Starting price for all assets.
            start_date: First timestamp.

        Returns:
            Dictionary mapping symbol to OHLCV DataFrame.
        """
        n = len(symbols)
        if n == 0:
            raise ValueError("symbols must not be empty")

        if correlations is None:
            correlations = np.eye(n) * 0.3 + 0.7  # mild correlation

        # Ensure positive semi-definite
        min_eig = np.min(np.linalg.eigvalsh(correlations))
        if min_eig < 0:
            correlations += np.eye(n) * (-min_eig + 1e-6)

        L = np.linalg.cholesky(correlations)
        uncorrelated = self.rng.standard_normal((periods, n))
        correlated = uncorrelated @ L.T

        results: dict[str, pd.DataFrame] = {}
        if start_date is None:
            start_date = datetime.now() - timedelta(days=periods)

        date_index = pd.date_range(start=start_date, periods=periods, freq="D")

        for i, symbol in enumerate(symbols):
            z = correlated[:, i]
            returns = 0.0002 + 0.02 * z
            close = base_price * np.exp(np.cumsum(returns))
            close = np.maximum(close, 0.01)

            noise = np.abs(self.rng.normal(scale=0.01, size=periods))
            high = np.maximum(close, close * (1 + noise))
            low = np.minimum(close, close * (1 - noise))
            open_ = np.roll(close, 1)
            open_[0] = base_price
            volume = np.exp(self.rng.normal(loc=math.log(1_000_000), scale=0.5, size=periods)).astype(int)

            df = pd.DataFrame(
                {
                    "open": np.round(open_, 4),
                    "high": np.round(high, 4),
                    "low": np.round(low, 4),
                    "close": np.round(close, 4),
                    "volume": volume,
                },
                index=date_index,
            )
            df.attrs["symbol"] = symbol
            results[symbol] = df

        logger.info("Generated portfolio with %s assets", n)
        return results
