"""Tests for the market_data module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.market_data import OHLCVGenerator, MultiAssetGenerator


class TestOHLCVGenerator:
    """Tests for OHLCVGenerator."""

    def test_generate_default(self):
        gen = OHLCVGenerator(seed=42)
        df = gen.generate(periods=100, symbol="TEST")
        assert len(df) == 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.freq == "D"
        assert df.attrs["symbol"] == "TEST"

    def test_all_prices_positive(self):
        gen = OHLCVGenerator(seed=42, start_price=50.0)
        df = gen.generate(periods=200)
        assert (df["close"] > 0).all()
        assert (df["open"] > 0).all()
        assert (df["high"] > 0).all()
        assert (df["low"] > 0).all()

    def test_high_gte_low(self):
        gen = OHLCVGenerator(seed=42)
        df = gen.generate(periods=100)
        assert (df["high"] >= df["low"]).all()

    def test_high_gte_close_and_low_lte_close(self):
        gen = OHLCVGenerator(seed=42)
        df = gen.generate(periods=100)
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["close"]).all()

    def test_volume_positive(self):
        gen = OHLCVGenerator(seed=42)
        df = gen.generate(periods=100)
        assert (df["volume"] > 0).all()

    def test_invalid_periods_raises(self):
        gen = OHLCVGenerator()
        with pytest.raises(ValueError, match="periods must be > 0"):
            gen.generate(periods=0)
        with pytest.raises(ValueError, match="periods must be > 0"):
            gen.generate(periods=-1)

    def test_invalid_start_price_raises(self):
        gen = OHLCVGenerator(start_price=-10)
        with pytest.raises(ValueError, match="start_price must be > 0"):
            gen.generate(periods=10)

    def test_invalid_volatility_raises(self):
        gen = OHLCVGenerator(volatility=-0.01)
        with pytest.raises(ValueError, match="volatility must be > 0"):
            gen.generate(periods=10)

    def test_save_and_load_csv(self, tmp_path):
        gen = OHLCVGenerator(seed=42)
        df = gen.generate(periods=50)
        csv_path = tmp_path / "test_data.csv"
        gen.save_to_csv(df, csv_path)
        assert csv_path.exists()
        loaded = gen.load_from_csv(csv_path)
        assert len(loaded) == 50
        assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]

    def test_save_and_load_preserves_values(self, tmp_path):
        gen = OHLCVGenerator(seed=123)
        df = gen.generate(periods=30)
        csv_path = tmp_path / "test_data.csv"
        gen.save_to_csv(df, csv_path)
        loaded = gen.load_from_csv(csv_path)
        pd.testing.assert_frame_equal(
            df, loaded, check_freq=False, check_index_type=False, check_names=False
        )

    def test_load_missing_file_raises(self):
        gen = OHLCVGenerator()
        with pytest.raises(FileNotFoundError):
            gen.load_from_csv(Path("/nonexistent/file.csv"))

    def test_seed_reproducibility(self):
        gen1 = OHLCVGenerator(seed=42)
        gen2 = OHLCVGenerator(seed=42)
        start = pd.Timestamp("2024-01-01")
        df1 = gen1.generate(periods=100, start_date=start)
        df2 = gen2.generate(periods=100, start_date=start)
        pd.testing.assert_frame_equal(
            df1, df2, check_freq=False, check_index_type=False, check_names=False
        )

    def test_different_seeds_produce_different_data(self):
        gen1 = OHLCVGenerator(seed=1)
        gen2 = OHLCVGenerator(seed=2)
        df1 = gen1.generate(periods=100)
        df2 = gen2.generate(periods=100)
        assert not df1["close"].equals(df2["close"])

    def test_mean_reversion(self):
        gen = OHLCVGenerator(seed=42, mean_reversion_speed=0.1, mean_reversion_level=100.0)
        df = gen.generate(periods=500)
        # With strong mean reversion, mean should be closer to reversion level
        mean_close = df["close"].mean()
        assert 50 < mean_close < 200  # loose bound

    def test_drift_affects_trend(self):
        gen_pos = OHLCVGenerator(seed=42, drift=0.01, volatility=0.01)
        gen_neg = OHLCVGenerator(seed=42, drift=-0.01, volatility=0.01)
        df_pos = gen_pos.generate(periods=100)
        df_neg = gen_neg.generate(periods=100)
        assert df_pos["close"].iloc[-1] > df_pos["close"].iloc[0]
        assert df_neg["close"].iloc[-1] < df_neg["close"].iloc[0]


class TestMultiAssetGenerator:
    """Tests for MultiAssetGenerator."""

    def test_generate_portfolio(self):
        gen = MultiAssetGenerator(seed=42)
        symbols = ["A", "B", "C"]
        portfolio = gen.generate_portfolio(symbols, periods=100)
        assert set(portfolio.keys()) == set(symbols)
        for sym, df in portfolio.items():
            assert len(df) == 100
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_empty_symbols_raises(self):
        gen = MultiAssetGenerator(seed=42)
        with pytest.raises(ValueError, match="symbols must not be empty"):
            gen.generate_portfolio([])

    def test_correlation_structure(self):
        gen = MultiAssetGenerator(seed=42)
        corr = np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.3], [0.5, 0.3, 1.0]])
        portfolio = gen.generate_portfolio(["X", "Y", "Z"], periods=500, correlations=corr)
        returns = pd.DataFrame({s: portfolio[s]["close"].pct_change() for s in portfolio})
        corr_matrix = returns.corr()
        # Check that correlations are in the right ballpark
        assert corr_matrix.iloc[0, 1] > 0.3  # X-Y should be positively correlated
        assert corr_matrix.iloc[0, 2] > 0.1

    def test_all_assets_positive_prices(self):
        gen = MultiAssetGenerator(seed=42)
        portfolio = gen.generate_portfolio(["A", "B"], periods=100)
        for df in portfolio.values():
            assert (df[["open", "high", "low", "close"]] > 0).all().all()
