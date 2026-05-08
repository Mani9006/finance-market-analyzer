"""Command-line interface for the Financial Market Trend Analyzer."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import indicators as ind
from .backtester import BacktestEngine, sma_crossover_signal, rsi_strategy_signal, buy_and_hold_signal
from .forecaster import NaiveDriftForecaster, PolynomialTrendForecaster, MovingAverageForecaster
from .market_data import OHLCVGenerator
from .reporter import generate_summary_report, JSONReporter
from .visualizer import (
    generate_all_charts,
    plot_backtest_equity,
    plot_drawdown,
    plot_forecast,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate synthetic market data."""
    gen = OHLCVGenerator(
        seed=args.seed,
        start_price=args.start_price,
        drift=args.drift,
        volatility=args.volatility,
    )
    df = gen.generate(
        periods=args.periods,
        freq=args.freq,
        symbol=args.symbol,
    )
    out_path = _resolve_path(args.output)
    if args.format == "csv":
        gen.save_to_csv(df, out_path)
    else:
        gen.save_to_parquet(df, out_path)
    print(f"Generated {len(df)} rows -> {out_path}")
    return 0


def cmd_indicators(args: argparse.Namespace) -> int:
    """Compute technical indicators."""
    path = _resolve_path(args.input)
    df = OHLCVGenerator.load_from_csv(path)
    result = df.copy()
    result["sma_20"] = ind.sma(df["close"], window=20)
    result["ema_20"] = ind.ema(df["close"], span=20)
    result["rsi_14"] = ind.rsi(df["close"], window=14)
    bb = ind.bollinger_bands(df["close"], window=20, num_std=2.0)
    for k, v in bb.items():
        result[f"bb_{k}"] = v
    macd_vals = ind.macd(df["close"])
    for k, v in macd_vals.items():
        result[f"macd_{k}"] = v
    out_path = _resolve_path(args.output)
    result.to_csv(out_path)
    print(f"Indicators saved to {out_path} ({len(result.columns)} columns)")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    """Run time series forecasting."""
    path = _resolve_path(args.input)
    df = OHLCVGenerator.load_from_csv(path)
    series = df["close"]

    model_map = {
        "naive": NaiveDriftForecaster,
        "poly": PolynomialTrendForecaster,
        "ma": MovingAverageForecaster,
    }
    model_cls = model_map.get(args.model, NaiveDriftForecaster)
    kwargs = {}
    if args.model == "poly":
        kwargs["degree"] = args.degree
    elif args.model == "ma":
        kwargs["window"] = args.window

    model = model_cls(**kwargs)
    model.fit(series)
    fc = model.forecast(horizon=args.horizon, confidence=args.confidence)

    print(f"\nForecast ({fc.model_name}) - {args.horizon} periods:")
    print(fc.forecast_values.to_string())
    if fc.metrics:
        print("\nMetrics:")
        for k, v in fc.metrics.items():
            print(f"  {k}: {v:.4f}")

    if args.chart:
        chart_dir = _resolve_path(args.chart_dir)
        plot_forecast(series, fc, title=f"Forecast ({fc.model_name})", save_path=chart_dir / "forecast.png")
        print(f"Chart saved to {chart_dir}/forecast.png")

    if args.json:
        out = _resolve_path(args.json)
        JSONReporter.save(fc, out)
        print(f"Forecast JSON saved to {out}")

    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run strategy backtest."""
    path = _resolve_path(args.input)
    df = OHLCVGenerator.load_from_csv(path)

    signal_map = {
        "buy_hold": buy_and_hold_signal,
        "sma_cross": sma_crossover_signal,
        "rsi": rsi_strategy_signal,
    }
    signal_fn = signal_map.get(args.strategy, sma_crossover_signal)

    engine = BacktestEngine(
        initial_capital=args.capital,
        commission=args.commission,
        slippage=args.slippage,
    )
    result = engine.run(df, signal_fn, strategy_name=args.strategy)

    print(f"\n{'='*50}")
    print(f"Backtest Results: {result.strategy_name}")
    print(f"{'='*50}")
    print(f"Initial Capital:  ${result.initial_capital:,.2f}")
    print(f"Final Capital:    ${result.final_capital:,.2f}")
    print(f"Total Return:     {result.total_return_pct:.2f}%")
    print(f"Total Trades:     {result.total_trades}")
    print(f"Win Rate:         {result.win_rate:.1f}%")
    print(f"Max Drawdown:     {result.max_drawdown_pct:.2f}%")
    print(f"Sharpe Ratio:     {result.sharpe_ratio:.3f}")
    print(f"Sortino Ratio:    {result.sortino_ratio:.3f}")
    print(f"Profit Factor:    {result.profit_factor:.3f}")
    print(f"Avg Trade Return: {result.avg_trade_return:.3f}%")

    if args.chart:
        chart_dir = _resolve_path(args.chart_dir)
        plot_backtest_equity(
            result.equity_curve,
            result.trades_df,
            title=f"Equity Curve - {result.strategy_name}",
            save_path=chart_dir / "equity_curve.png",
        )
        plot_drawdown(
            result.equity_curve,
            title=f"Drawdown - {result.strategy_name}",
            save_path=chart_dir / "drawdown.png",
        )
        print(f"Charts saved to {chart_dir}/")

    if args.json:
        JSONReporter.save(result, _resolve_path(args.json))

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate a full analysis report."""
    path = _resolve_path(args.input)
    df = OHLCVGenerator.load_from_csv(path)

    # Compute indicators
    indicators_df = df.copy()
    indicators_df["sma_20"] = ind.sma(df["close"], 20)
    indicators_df["rsi_14"] = ind.rsi(df["close"], 14)
    bb = ind.bollinger_bands(df["close"])
    for k, v in bb.items():
        indicators_df[f"bb_{k}"] = v

    # Generate charts
    chart_dir = _resolve_path(args.chart_dir)
    chart_paths = generate_all_charts(df, output_dir=chart_dir, prefix=args.symbol)

    # Run quick forecast
    model = NaiveDriftForecaster()
    model.fit(df["close"])
    fc = model.forecast(horizon=30)

    # Run quick backtest
    engine = BacktestEngine()
    bt = engine.run(df, sma_crossover_signal, strategy_name="sma_cross")

    # Generate report
    report_path = _resolve_path(args.output)
    generate_summary_report(
        df=df,
        indicators_df=indicators_df,
        backtest_result=bt,
        forecast_result=fc,
        chart_paths=chart_paths,
        output_path=report_path,
    )
    print(f"Report saved to {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finanalyzer",
        description="Financial Market Trend Analyzer - CLI tool for technical analysis, forecasting, and backtesting.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic OHLCV data")
    gen_parser.add_argument("-p", "--periods", type=int, default=252, help="Number of periods (default: 252)")
    gen_parser.add_argument("--freq", default="D", help="Frequency (default: D)")
    gen_parser.add_argument("--start-price", type=float, default=100.0, help="Starting price (default: 100.0)")
    gen_parser.add_argument("--drift", type=float, default=0.0002, help="Daily drift (default: 0.0002)")
    gen_parser.add_argument("--volatility", type=float, default=0.02, help="Daily volatility (default: 0.02)")
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    gen_parser.add_argument("-s", "--symbol", default="SYNTH", help="Symbol name (default: SYNTH)")
    gen_parser.add_argument("-o", "--output", default="data/ohlcv.csv", help="Output file (default: data/ohlcv.csv)")
    gen_parser.add_argument("--format", choices=["csv", "parquet"], default="csv", help="Output format")

    # indicators
    ind_parser = subparsers.add_parser("indicators", help="Compute technical indicators")
    ind_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    ind_parser.add_argument("-o", "--output", default="data/indicators.csv", help="Output file")

    # forecast
    fc_parser = subparsers.add_parser("forecast", help="Run time series forecasting")
    fc_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    fc_parser.add_argument("--model", choices=["naive", "poly", "ma"], default="naive", help="Model type")
    fc_parser.add_argument("--horizon", type=int, default=30, help="Forecast horizon")
    fc_parser.add_argument("--degree", type=int, default=2, help="Polynomial degree (for poly model)")
    fc_parser.add_argument("--window", type=int, default=20, help="MA window (for ma model)")
    fc_parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")
    fc_parser.add_argument("--chart", action="store_true", help="Generate forecast chart")
    fc_parser.add_argument("--chart-dir", default="charts", help="Chart output directory")
    fc_parser.add_argument("--json", default=None, help="Save forecast as JSON")

    # backtest
    bt_parser = subparsers.add_parser("backtest", help="Run strategy backtest")
    bt_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    bt_parser.add_argument("--strategy", choices=["buy_hold", "sma_cross", "rsi"], default="sma_cross", help="Strategy")
    bt_parser.add_argument("--capital", type=float, default=100_000.0, help="Initial capital")
    bt_parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    bt_parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage rate")
    bt_parser.add_argument("--chart", action="store_true", help="Generate charts")
    bt_parser.add_argument("--chart-dir", default="charts", help="Chart output directory")
    bt_parser.add_argument("--json", default=None, help="Save results as JSON")

    # report
    rep_parser = subparsers.add_parser("report", help="Generate full analysis report")
    rep_parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    rep_parser.add_argument("-o", "--output", default="reports/report.html", help="Output HTML report")
    rep_parser.add_argument("-s", "--symbol", default="SYNTH", help="Symbol name")
    rep_parser.add_argument("--chart-dir", default="charts", help="Chart output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    command_map = {
        "generate": cmd_generate,
        "indicators": cmd_indicators,
        "forecast": cmd_forecast,
        "backtest": cmd_backtest,
        "report": cmd_report,
    }

    handler = command_map.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return handler(args)
    except Exception as e:
        logger.error("Command '%s' failed: %s", args.command, e)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
