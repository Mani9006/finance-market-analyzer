"""Chart generation and visualization for financial market analysis."""

import logging
import os
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# Ensure output directories exist
CHARTS_DIR = Path(__file__).resolve().parent.parent / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def _save_or_show(fig: plt.Figure, save_path: Optional[Path] = None, dpi: int = 150) -> Optional[Path]:
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info("Chart saved to %s", save_path)
        plt.close(fig)
        return save_path
    plt.show()
    return None


def plot_ohlc(
    df: pd.DataFrame,
    title: str = "OHLC Chart",
    save_path: Optional[Path] = None,
    volume: bool = True,
    rolling_avg: bool = True,
    figsize: tuple[int, int] = (14, 8),
) -> Optional[Path]:
    """Plot OHLC price chart with optional volume and moving average."""
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    n_panels = 2 if volume else 1
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_panels, 1, height_ratios=[3, 1] if volume else [1])

    ax_price = fig.add_subplot(gs[0])
    ax_price.set_title(title, fontsize=14, fontweight="bold")

    # Plot close price with color-coded candle regions
    colors = ["green" if df["close"].iloc[i] >= df["open"].iloc[i] else "red" for i in range(len(df))]
    ax_price.plot(df.index, df["close"], color="#2196F3", linewidth=1.2, label="Close", zorder=3)
    ax_price.fill_between(df.index, df["low"], df["high"], alpha=0.15, color="gray", label="Range")

    if rolling_avg:
        ma20 = df["close"].rolling(window=20, min_periods=20).mean()
        ma50 = df["close"].rolling(window=50, min_periods=50).mean()
        ax_price.plot(df.index, ma20, color="orange", linewidth=1, label="MA20", alpha=0.8)
        ax_price.plot(df.index, ma50, color="purple", linewidth=1, label="MA50", alpha=0.8)

    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.3)

    if volume:
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
        vcolors = ["green" if df["close"].iloc[i] >= df["open"].iloc[i] else "red" for i in range(len(df))]
        ax_vol.bar(df.index, df["volume"], color=vcolors, alpha=0.6, width=0.8)
        ax_vol.set_ylabel("Volume")
        ax_vol.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_with_indicators(
    df: pd.DataFrame,
    indicators: dict[str, pd.Series],
    title: str = "Price with Indicators",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10),
) -> Optional[Path]:
    """Plot price chart overlaid with technical indicators.

    Args:
        df: OHLCV DataFrame.
        indicators: Dict of indicator name to pandas Series.
        title: Chart title.
        save_path: Optional path to save PNG.
        figsize: Figure dimensions.

    Returns:
        Path to saved file if save_path was provided.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    ax_main = fig.add_subplot(gs[0])
    ax_main.set_title(title, fontsize=14, fontweight="bold")
    ax_main.plot(df.index, df["close"], color="#2196F3", linewidth=1.2, label="Close", zorder=3)

    colors = plt.cm.tab10(np.linspace(0, 1, len(indicators)))
    for (name, series), color in zip(indicators.items(), colors):
        ax_main.plot(series.index, series.values, color=color, linewidth=1, label=name, alpha=0.8)

    ax_main.legend(loc="upper left", fontsize=9)
    ax_main.set_ylabel("Price")
    ax_main.grid(True, alpha=0.3)

    # Volume subplot
    ax_vol = fig.add_subplot(gs[1], sharex=ax_main)
    vcolors = ["green" if df["close"].iloc[i] >= df["open"].iloc[i] else "red" for i in range(len(df))]
    ax_vol.bar(df.index, df["volume"], color=vcolors, alpha=0.5)
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_macd(
    macd_result: dict[str, pd.Series],
    title: str = "MACD",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 5),
) -> Optional[Path]:
    """Plot MACD line, signal, and histogram.

    Args:
        macd_result: Dict from indicators.macd() with keys: macd, signal, histogram.
        title: Chart title.
        save_path: Optional path to save PNG.
        figsize: Figure dimensions.

    Returns:
        Path to saved file if save_path was provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold")

    idx = macd_result["macd"].index
    ax.plot(idx, macd_result["macd"].values, color="#2196F3", linewidth=1.2, label="MACD")
    ax.plot(idx, macd_result["signal"].values, color="orange", linewidth=1.2, label="Signal")

    hist = macd_result["histogram"].values
    pos_mask = hist >= 0
    ax.bar(idx[pos_mask], hist[pos_mask], color="green", alpha=0.6, width=0.8, label="Histogram +")
    ax.bar(idx[~pos_mask], hist[~pos_mask], color="red", alpha=0.6, width=0.8, label="Histogram -")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_rsi(
    rsi_series: pd.Series,
    title: str = "RSI",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 4),
) -> Optional[Path]:
    """Plot RSI with overbought/oversold zones.

    Args:
        rsi_series: RSI values (0-100).
        title: Chart title.
        save_path: Optional path to save PNG.
        figsize: Figure dimensions.

    Returns:
        Path to saved file if save_path was provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.plot(rsi_series.index, rsi_series.values, color="#673AB7", linewidth=1.2)
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8, label="Overbought (70)")
    ax.axhline(30, color="green", linestyle="--", linewidth=0.8, label="Oversold (30)")
    ax.fill_between(rsi_series.index, 70, 100, color="red", alpha=0.08)
    ax.fill_between(rsi_series.index, 0, 30, color="green", alpha=0.08)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_bollinger(
    df: pd.DataFrame,
    bb_result: dict[str, pd.Series],
    title: str = "Bollinger Bands",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 7),
) -> Optional[Path]:
    """Plot price with Bollinger Bands.

    Args:
        df: OHLCV DataFrame.
        bb_result: Dict from indicators.bollinger_bands().
        title: Chart title.
        save_path: Optional path to save PNG.
        figsize: Figure dimensions.

    Returns:
        Path to saved file if save_path was provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.plot(df.index, df["close"], color="#2196F3", linewidth=1.2, label="Close", zorder=3)
    ax.plot(bb_result["middle"].index, bb_result["middle"].values, color="orange", linewidth=1, label="SMA20")
    ax.plot(bb_result["upper"].index, bb_result["upper"].values, color="red", linewidth=0.8, linestyle="--", label="Upper")
    ax.plot(bb_result["lower"].index, bb_result["lower"].values, color="green", linewidth=0.8, linestyle="--", label="Lower")
    ax.fill_between(bb_result["upper"].index, bb_result["upper"].values, bb_result["lower"].values, alpha=0.1, color="gray")

    ax.legend(fontsize=9)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_forecast(
    historical: pd.Series,
    forecast_result,
    title: str = "Price Forecast",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 7),
) -> Optional[Path]:
    """Plot historical prices with forecast and confidence bands.

    Args:
        historical: Historical price series.
        forecast_result: ForecastResult dataclass from forecaster module.
        title: Chart title.
        save_path: Optional path to save PNG.
        figsize: Figure dimensions.

    Returns:
        Path to saved file if save_path was provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.plot(historical.index, historical.values, color="#2196F3", linewidth=1.2, label="Historical")

    fc = forecast_result.forecast_values
    ax.plot(fc.index, fc.values, color="orange", linewidth=1.5, linestyle="--", label="Forecast")

    if forecast_result.confidence_lower is not None and forecast_result.confidence_upper is not None:
        lower = forecast_result.confidence_lower
        upper = forecast_result.confidence_upper
        ax.fill_between(fc.index, lower.values, upper.values, alpha=0.2, color="orange", label="95% CI")

    ax.legend(fontsize=9)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_backtest_equity(
    equity_curve: pd.Series,
    trades_df: Optional[pd.DataFrame] = None,
    title: str = "Backtest Equity Curve",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 6),
) -> Optional[Path]:
    """Plot equity curve with trade markers.

    Args:
        equity_curve: Series of portfolio value over time.
        trades_df: Optional DataFrame of trades with columns entry_time, exit_time, pnl.
        title: Chart title.
        save_path: Optional path to save PNG.
        figsize: Figure dimensions.

    Returns:
        Path to saved file if save_path was provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.plot(equity_curve.index, equity_curve.values, color="#2196F3", linewidth=1.2, label="Equity")
    ax.axhline(equity_curve.iloc[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Initial Capital")

    if trades_df is not None and not trades_df.empty:
        for _, row in trades_df.iterrows():
            if pd.notna(row["pnl"]) and pd.notna(row["exit_time"]):
                color = "green" if row["pnl"] > 0 else "red"
                ax.axvline(pd.Timestamp(row["exit_time"]), color=color, alpha=0.15, linewidth=0.5)

    ax.legend(fontsize=9)
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown Chart",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 4),
) -> Optional[Path]:
    """Plot drawdown from peak over time.

    Args:
        equity_curve: Series of portfolio value.
        title: Chart title.
        save_path: Optional path to save PNG.
        figsize: Figure dimensions.

    Returns:
        Path to saved file if save_path was provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold")

    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak * 100
    ax.fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.5)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


def plot_correlation_heatmap(
    returns_df: pd.DataFrame,
    title: str = "Asset Correlation Heatmap",
    save_path: Optional[Path] = None,
    figsize: tuple[int, int] = (8, 7),
) -> Optional[Path]:
    """Plot a correlation heatmap for asset returns.

    Args:
        returns_df: DataFrame of asset returns (columns = assets).
        title: Chart title.
        save_path: Optional path to save PNG.
        figsize: Figure dimensions.

    Returns:
        Path to saved file if save_path was provided.
    """
    corr = returns_df.corr()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold")
    cax = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(cax, ax=ax, shrink=0.8)

    ticks = np.arange(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    return _save_or_show(fig, save_path)


def generate_all_charts(
    df: pd.DataFrame,
    output_dir: Path = CHARTS_DIR,
    prefix: str = "chart",
) -> list[Path]:
    """Generate a complete set of charts for a dataset.

    Args:
        df: OHLCV DataFrame.
        output_dir: Directory to save charts.
        prefix: Filename prefix.

    Returns:
        List of saved file paths.
    """
    from . import indicators as ind

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # OHLC
    p = plot_ohlc(df, title=f"{prefix} OHLC", save_path=output_dir / f"{prefix}_ohlc.png")
    if p:
        paths.append(p)

    # RSI
    rsi_vals = ind.rsi(df["close"], window=14)
    p = plot_rsi(rsi_vals, title=f"{prefix} RSI", save_path=output_dir / f"{prefix}_rsi.png")
    if p:
        paths.append(p)

    # MACD
    macd_vals = ind.macd(df["close"])
    p = plot_macd(macd_vals, title=f"{prefix} MACD", save_path=output_dir / f"{prefix}_macd.png")
    if p:
        paths.append(p)

    # Bollinger
    bb = ind.bollinger_bands(df["close"])
    p = plot_bollinger(df, bb, title=f"{prefix} Bollinger Bands", save_path=output_dir / f"{prefix}_bollinger.png")
    if p:
        paths.append(p)

    logger.info("Generated %s charts in %s", len(paths), output_dir)
    return paths
