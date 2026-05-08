"""Report generation for financial market analysis results."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class HTMLReporter:
    """Generate HTML reports with charts and analysis summaries."""

    def __init__(self, title: str = "Financial Market Analysis Report"):
        self.title = title
        self.sections: list[dict[str, Any]] = []

    def add_section(self, heading: str, content: str, chart_paths: Optional[list[Path]] = None) -> "HTMLReporter":
        """Add a section to the report.

        Args:
            heading: Section heading.
            content: HTML content string.
            chart_paths: Optional list of chart image paths to embed.
        """
        self.sections.append({"heading": heading, "content": content, "charts": chart_paths or []})
        return self

    def add_dataframe(self, heading: str, df: pd.DataFrame, max_rows: int = 20) -> "HTMLReporter":
        """Add a DataFrame as an HTML table section."""
        display_df = df.head(max_rows) if len(df) > max_rows else df
        html_table = display_df.to_html(border=0, classes="dataframe-table", index=True)
        content = f'<div class="table-wrap">{html_table}</div>'
        if len(df) > max_rows:
            content += f'<p class="note">Showing first {max_rows} of {len(df)} rows.</p>'
        self.add_section(heading, content)
        return self

    def add_metrics(self, heading: str, metrics: dict[str, Any]) -> "HTMLReporter":
        """Add a metrics/key-value section."""
        rows = "\n".join(
            f"<tr><td class='metric-key'>{k}</td><td class='metric-val'>{self._fmt(v)}</td></tr>"
            for k, v in metrics.items()
        )
        content = f'<table class="metrics-table">{rows}</table>'
        self.add_section(heading, content)
        return self

    @staticmethod
    def _fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:,.4f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

    def render(self) -> str:
        """Render the full HTML report string."""
        sections_html = ""
        for i, sec in enumerate(self.sections):
            charts_html = ""
            for chart_path in sec["charts"]:
                rel_path = str(chart_path)  # embed as relative
                charts_html += f'<div class="chart"><img src="{rel_path}" alt="chart" /></div>\n'
            sections_html += f"""
            <section id="sec-{i}">
                <h2>{sec['heading']}</h2>
                {sec['content']}
                {charts_html}
            </section>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        :root {{
            --bg: #f8f9fa;
            --fg: #212529;
            --accent: #2196F3;
            --muted: #6c757d;
            --border: #dee2e6;
            --card: #ffffff;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--fg);
            max-width: 1100px;
            margin: 0 auto;
            padding: 2rem 1rem;
            line-height: 1.6;
        }}
        header {{
            border-bottom: 2px solid var(--accent);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }}
        header h1 {{
            margin: 0;
            font-size: 1.8rem;
            color: var(--accent);
        }}
        header .meta {{
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.3rem;
        }}
        section {{
            background: var(--card);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        section h2 {{
            margin-top: 0;
            font-size: 1.3rem;
            color: var(--fg);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
        }}
        .dataframe-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9rem;
        }}
        .dataframe-table th, .dataframe-table td {{
            padding: 6px 10px;
            text-align: right;
            border-bottom: 1px solid var(--border);
        }}
        .dataframe-table th {{
            background: #f1f3f5;
            font-weight: 600;
        }}
        .metrics-table {{
            width: 100%;
            max-width: 500px;
            border-collapse: collapse;
        }}
        .metrics-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
        }}
        .metric-key {{
            font-weight: 600;
            color: var(--muted);
            width: 50%;
        }}
        .metric-val {{
            text-align: right;
            font-family: "SF Mono", Monaco, monospace;
        }}
        .chart img {{
            max-width: 100%;
            border-radius: 6px;
            margin-top: 1rem;
        }}
        .note {{
            color: var(--muted);
            font-size: 0.85rem;
        }}
        footer {{
            text-align: center;
            color: var(--muted);
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <header>
        <h1>{self.title}</h1>
        <div class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </header>
    {sections_html}
    <footer>
        Financial Market Trend Analyzer &mdash; Generated Report
    </footer>
</body>
</html>"""

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the HTML report to a file.

        Args:
            path: Output file path. Defaults to reports_dir/report_TIMESTAMP.html.

        Returns:
            Path to saved file.
        """
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = REPORTS_DIR / f"report_{ts}.html"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        logger.info("HTML report saved to %s", path)
        return path


class JSONReporter:
    """Generate JSON reports from analysis results."""

    @staticmethod
    def serialize(result: Any) -> dict[str, Any]:
        """Serialize a result object to a JSON-compatible dict."""
        if hasattr(result, "__dataclass_fields__"):
            d = asdict(result)
            # Convert non-serializable items
            for key, value in list(d.items()):
                if isinstance(value, pd.Series):
                    d[key] = {"index": value.index.astype(str).tolist(), "values": value.values.tolist()}
                elif isinstance(value, pd.DataFrame):
                    d[key] = value.head(100).to_dict(orient="list")
                elif isinstance(value, list) and value and hasattr(value[0], "__dataclass_fields__"):
                    d[key] = [asdict(v) for v in value]
            return d
        if isinstance(result, pd.DataFrame):
            return result.head(1000).to_dict(orient="list")
        if isinstance(result, pd.Series):
            return result.dropna().head(1000).to_dict()
        return {"result": str(result)}

    @classmethod
    def save(cls, result: Any, path: Optional[Path] = None) -> Path:
        """Serialize and save a result to JSON.

        Args:
            result: Analysis result object.
            path: Output file path. Defaults to reports_dir/result_TIMESTAMP.json.

        Returns:
            Path to saved file.
        """
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = REPORTS_DIR / f"result_{ts}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = cls.serialize(result)
        data["generated_at"] = datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("JSON report saved to %s", path)
        return path


def generate_summary_report(
    df: pd.DataFrame,
    indicators_df: Optional[pd.DataFrame] = None,
    backtest_result: Optional[Any] = None,
    forecast_result: Optional[Any] = None,
    chart_paths: Optional[list[Path]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate a comprehensive HTML summary report.

    Args:
        df: Primary OHLCV DataFrame.
        indicators_df: DataFrame with indicator columns.
        backtest_result: BacktestResult object.
        forecast_result: ForecastResult object.
        chart_paths: List of chart image paths to include.
        output_path: Target HTML file path.

    Returns:
        Path to saved HTML report.
    """
    reporter = HTMLReporter(title="Financial Market Analysis Summary")

    # Data overview
    reporter.add_section(
        "Data Overview",
        f"""
        <p><strong>Symbol:</strong> {df.attrs.get('symbol', 'N/A')}</p>
        <p><strong>Periods:</strong> {len(df):,}</p>
        <p><strong>Date Range:</strong> {df.index[0].date()} to {df.index[-1].date()}</p>
        <p><strong>Latest Close:</strong> {df['close'].iloc[-1]:,.4f}</p>
        """,
    )

    # Price statistics
    stats = {
        "Mean Close": df["close"].mean(),
        "Std Dev": df["close"].std(),
        "Min Close": df["close"].min(),
        "Max Close": df["close"].max(),
        "Mean Volume": df["volume"].mean(),
        "Total Return (%)": (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100,
    }
    reporter.add_metrics("Price Statistics", stats)

    if indicators_df is not None:
        reporter.add_dataframe("Indicators Sample", indicators_df)

    if backtest_result is not None:
        bt_metrics = {
            "Strategy": backtest_result.strategy_name,
            "Initial Capital": backtest_result.initial_capital,
            "Final Capital": backtest_result.final_capital,
            "Total Return (%)": backtest_result.total_return_pct,
            "Total Trades": backtest_result.total_trades,
            "Win Rate (%)": backtest_result.win_rate,
            "Max Drawdown (%)": backtest_result.max_drawdown_pct,
            "Sharpe Ratio": backtest_result.sharpe_ratio,
            "Sortino Ratio": backtest_result.sortino_ratio,
            "Profit Factor": backtest_result.profit_factor,
        }
        reporter.add_metrics("Backtest Results", bt_metrics)

    if forecast_result is not None and forecast_result.metrics is not None:
        reporter.add_metrics("Forecast Metrics", forecast_result.metrics)

    if chart_paths:
        reporter.add_section("Charts", "", chart_paths=chart_paths)

    return reporter.save(output_path)
