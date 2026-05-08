"""Strategy backtesting engine for financial trading simulations."""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single executed trade."""

    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    position: str = "long"  # "long" or "short"
    size: float = 1.0
    pnl: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Container for backtest results."""

    strategy_name: str
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_trade_return: float
    profit_factor: float
    equity_curve: pd.Series = field(repr=False)
    trades: list[Trade] = field(default_factory=list, repr=False)
    trades_df: Optional[pd.DataFrame] = None


class BacktestEngine:
    """Event-driven backtesting engine.

    Simulates trading decisions on historical OHLCV data using a user-defined
    signal function. Supports position sizing, commission, and slippage.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 1.0,
    ):
        if initial_capital <= 0:
            raise ValueError("initial_capital must be > 0")
        if commission < 0 or slippage < 0:
            raise ValueError("commission and slippage must be >= 0")
        self.initial_capital = float(initial_capital)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.position_size = float(position_size)

    def run(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.DataFrame, int], int],
        strategy_name: str = "unnamed",
    ) -> BacktestResult:
        """Run a backtest.

        Args:
            df: OHLCV DataFrame with columns open, high, low, close, volume.
            signal_fn: Callable(df, index) -> signal where:
                1 = enter long, -1 = enter short, 0 = flat/exit.
            strategy_name: Human-readable strategy name.

        Returns:
            BacktestResult with full statistics.
        """
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")
        if len(df) < 2:
            raise ValueError("DataFrame must have at least 2 rows")

        capital = self.initial_capital
        equity = np.zeros(len(df))
        equity[0] = capital
        trades: list[Trade] = []
        current_trade: Optional[Trade] = None
        position = 0  # 0 = flat, 1 = long, -1 = short

        for i in range(1, len(df)):
            signal = signal_fn(df, i)
            price = float(df["close"].iloc[i])

            # Entry logic
            if position == 0 and signal != 0:
                position = 1 if signal > 0 else -1
                adjusted_price = price * (1 + self.slippage * np.sign(signal))
                current_trade = Trade(
                    entry_time=df.index[i],
                    entry_price=adjusted_price,
                    position="long" if signal > 0 else "short",
                    size=self.position_size,
                )

            # Exit logic
            elif position != 0 and signal == 0:
                adjusted_price = price * (1 - self.slippage * np.sign(position))
                if current_trade is not None:
                    current_trade.exit_time = df.index[i]
                    current_trade.exit_price = adjusted_price
                    if current_trade.position == "long":
                        gross_pnl = (adjusted_price - current_trade.entry_price) * current_trade.size
                    else:
                        gross_pnl = (current_trade.entry_price - adjusted_price) * current_trade.size
                    cost = (current_trade.entry_price + adjusted_price) * self.commission * current_trade.size
                    current_trade.pnl = gross_pnl - cost
                    current_trade.return_pct = (
                        current_trade.pnl / (current_trade.entry_price * current_trade.size)
                    ) * 100
                    current_trade.exit_reason = "signal"
                    capital += current_trade.pnl
                    trades.append(current_trade)
                position = 0
                current_trade = None

            # Update equity
            unrealized = 0.0
            if current_trade is not None:
                if current_trade.position == "long":
                    unrealized = (price - current_trade.entry_price) * current_trade.size
                else:
                    unrealized = (current_trade.entry_price - price) * current_trade.size
                # approximate unrealized commission
                unrealized -= price * self.commission * current_trade.size
            equity[i] = capital + unrealized

        # Force close any open position at the end
        if current_trade is not None:
            final_price = float(df["close"].iloc[-1])
            current_trade.exit_time = df.index[-1]
            current_trade.exit_price = final_price
            if current_trade.position == "long":
                gross_pnl = (final_price - current_trade.entry_price) * current_trade.size
            else:
                gross_pnl = (current_trade.entry_price - final_price) * current_trade.size
            cost = (current_trade.entry_price + final_price) * self.commission * current_trade.size
            current_trade.pnl = gross_pnl - cost
            current_trade.return_pct = (
                current_trade.pnl / (current_trade.entry_price * current_trade.size)
            ) * 100
            current_trade.exit_reason = "end_of_data"
            capital += current_trade.pnl
            trades.append(current_trade)
            equity[-1] = capital

        return self._build_result(strategy_name, equity, trades, df)

    def _build_result(
        self,
        strategy_name: str,
        equity: np.ndarray,
        trades: list[Trade],
        df: pd.DataFrame,
    ) -> BacktestResult:
        equity_series = pd.Series(equity, index=df.index)
        final_capital = equity[-1]
        total_return = final_capital - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        max_drawdown = float(np.min(drawdown))
        max_drawdown_pct = float(np.min(drawdown / np.where(peak > 0, peak, 1)) * 100)

        # Trade stats
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl < 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        avg_trade_return = float(np.mean([t.return_pct for t in trades])) if trades else 0.0

        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Returns and ratios
        returns = pd.Series(equity).pct_change().dropna().replace([np.inf, -np.inf], 0)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
            downside = returns[returns < 0]
            sortino = float(returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else None

        logger.info(
            "Backtest '%s': %s trades, %.2f%% return, max DD %.2f%%",
            strategy_name, total_trades, total_return_pct, max_drawdown_pct,
        )

        return BacktestResult(
            strategy_name=strategy_name,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            avg_trade_return=avg_trade_return,
            profit_factor=profit_factor,
            equity_curve=equity_series,
            trades=trades,
            trades_df=trades_df,
        )


def buy_and_hold_signal(_df: pd.DataFrame, idx: int) -> int:
    """Buy at start, hold forever."""
    return 1 if idx == 1 else 0


def sma_crossover_signal(df: pd.DataFrame, idx: int, fast: int = 10, slow: int = 30) -> int:
    """SMA crossover: long when fast > slow, short when fast < slow."""
    if idx < slow:
        return 0
    sma_fast = df["close"].iloc[idx - fast + 1 : idx + 1].mean()
    sma_slow = df["close"].iloc[idx - slow + 1 : idx + 1].mean()
    if sma_fast > sma_slow:
        return 1
    elif sma_fast < sma_slow:
        return -1
    return 0


def rsi_strategy_signal(df: pd.DataFrame, idx: int, period: int = 14, oversold: float = 30, overbought: float = 70) -> int:
    """RSI mean-reversion: long when oversold, short when overbought."""
    if idx < period:
        return 0
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).iloc[idx - period + 1 : idx + 1]
    loss = (-delta).where(delta < 0, 0).iloc[idx - period + 1 : idx + 1]
    avg_gain = gain.mean()
    avg_loss = loss.mean()
    if avg_loss == 0:
        return 0
    rsi_val = 100 - (100 / (1 + avg_gain / avg_loss))
    if rsi_val < oversold:
        return 1
    elif rsi_val > overbought:
        return -1
    return 0
