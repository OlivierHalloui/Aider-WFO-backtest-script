#!/usr/bin/env python3
"""
Analyze and compare exit-signal performance from a TradingView trade list CSV,
and generate a PDF report with charts.

Example:
  python apps/exit_signals/analyze_exit_signals.py \
    --input /home/olivier/Downloads/ATDMF_strat_long_BTCUSDC_05S-MEXC_V6_2_BINANCE_BTCUSDT_2026-02-01_e3c73.csv \
    --exclude-signal Open \
    --pdf-report /home/olivier/ATDMF_backtest_V1/reports/exit_signal_report.pdf
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple


REQUIRED_COLUMNS = [
    "Type",
    "Date and time",
    "Signal",
    "Net P&L USDT",
    "Net P&L %",
]


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _signal_stats(group: pd.DataFrame) -> pd.Series:
    pnl = group["Net P&L USDT"].astype(float)
    pnl_pct = group["Net P&L %"].astype(float)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    avg_win = _safe_mean(wins)
    avg_loss = _safe_mean(losses)
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss

    profit_factor = np.inf
    if losses.sum() < 0:
        profit_factor = wins.sum() / abs(losses.sum())

    std_pnl_pct = float(pnl_pct.std(ddof=0))
    downside_dev_pct = float(np.sqrt(np.mean(np.minimum(pnl_pct, 0) ** 2))) if len(pnl_pct) else 0.0

    if "Date and time" in group.columns:
        group_sorted = group.sort_values("Date and time")
    else:
        group_sorted = group
    cum_pnl = group_sorted["Net P&L USDT"].astype(float).cumsum()
    drawdown = cum_pnl - cum_pnl.cummax()
    max_drawdown_usdt = float(drawdown.min()) if len(drawdown) else 0.0

    payoff_ratio = np.inf
    if avg_loss < 0:
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss else np.inf

    out = {
        "exits": int(len(group)),
        "total_pnl_usdt": float(pnl.sum()),
        "avg_pnl_usdt": float(pnl.mean()),
        "median_pnl_usdt": float(pnl.median()),
        "std_pnl_usdt": float(pnl.std(ddof=0)),
        "total_pnl_pct": float(pnl_pct.sum()),
        "avg_pnl_pct": float(pnl_pct.mean()),
        "std_pnl_pct": std_pnl_pct,
        "downside_dev_pct": downside_dev_pct,
        "win_rate": win_rate,
        "avg_win_usdt": float(avg_win),
        "avg_loss_usdt": float(avg_loss),
        "expectancy_usdt": float(expectancy),
        "profit_factor": float(profit_factor),
        "payoff_ratio": float(payoff_ratio),
        "max_drawdown_usdt": max_drawdown_usdt,
    }

    for col in ["Favorable excursion USDT", "Adverse excursion USDT"]:
        if col in group.columns:
            out[f"avg_{col.lower().replace(' ', '_')}"] = float(group[col].mean())

    return pd.Series(out)


def _ensure_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _make_outdir(base_outdir: Optional[str]) -> str:
    if base_outdir:
        outdir = base_outdir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = os.path.join("reports", f"exit_signals_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _load_plotting() -> Tuple[bool, Optional[object], Optional[object], Optional[object], Optional[str]]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
        from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        return False, None, None, None, str(exc)
    return True, plt, sns, PdfPages, None


def _save_fig(plt_module, path: str) -> None:
    plt_module.tight_layout()
    plt_module.savefig(path, dpi=160)
    plt_module.close()


def _rank01(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ranked = series.rank(pct=True, method="average")
    if not higher_is_better:
        ranked = 1.0 - ranked
    return ranked.fillna(0.0)


def _compute_weights(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    summary = summary.copy()

    # Derived metrics
    summary["sharpe_like_pct"] = summary.apply(
        lambda r: r["avg_pnl_pct"] / r["std_pnl_pct"] if r["std_pnl_pct"] > 0 else 0.0,
        axis=1,
    )
    summary["stability_score"] = summary.apply(
        lambda r: (1.0 / r["downside_dev_pct"]) if r["downside_dev_pct"] > 0 else 0.0,
        axis=1,
    )

    # Handle inf profit factor for ranking
    finite_pf = summary["profit_factor"].replace([np.inf, -np.inf], np.nan)
    max_pf = finite_pf.max() if finite_pf.notna().any() else 1.0
    summary["profit_factor_rank"] = summary["profit_factor"].replace(np.inf, max_pf * 1.1)

    # Ranks (0..1)
    r_expectancy = _rank01(summary["expectancy_usdt"], higher_is_better=True)
    r_winrate = _rank01(summary["win_rate"], higher_is_better=True)
    r_profit = _rank01(summary["profit_factor_rank"], higher_is_better=True)
    r_sharpe = _rank01(summary["sharpe_like_pct"], higher_is_better=True)
    r_drawdown = _rank01(summary["max_drawdown_usdt"].abs(), higher_is_better=False)
    r_total_pnl = _rank01(summary["total_pnl_usdt"], higher_is_better=True)
    r_stability = _rank01(summary["downside_dev_pct"], higher_is_better=False)

    weight_sum = (
        args.w_expectancy
        + args.w_winrate
        + args.w_profit_factor
        + args.w_sharpe
        + args.w_drawdown
        + args.w_total_pnl
        + args.w_stability
    )
    if weight_sum <= 0:
        raise SystemExit("Sum of weight components must be > 0.")

    composite = (
        args.w_expectancy * r_expectancy
        + args.w_winrate * r_winrate
        + args.w_profit_factor * r_profit
        + args.w_sharpe * r_sharpe
        + args.w_drawdown * r_drawdown
        + args.w_total_pnl * r_total_pnl
        + args.w_stability * r_stability
    ) / weight_sum

    # Confidence factor based on sample size
    confidence = 1.0 - np.exp(-summary["exits"] / max(args.confidence_k, 1e-9))
    raw_score = composite * confidence

    # Hard filters for negative signals / insufficient data
    negative_mask = (
        (summary["total_pnl_usdt"] <= args.negative_pnl_threshold)
        | (summary["expectancy_usdt"] <= args.negative_expectancy_threshold)
        | (summary["exits"] < args.min_exits_weight)
    )
    raw_score = raw_score.mask(negative_mask, 0.0)

    eligible_mask = ~negative_mask
    eligible_count = int(eligible_mask.sum())
    if eligible_count == 0:
        weight_frac = raw_score * 0.0
    else:
        sum_raw = raw_score[eligible_mask].sum()
        if sum_raw > 0:
            base_frac = raw_score / sum_raw
        else:
            base_frac = raw_score * 0.0
            base_frac[eligible_mask] = 1.0 / eligible_count

        floor = max(args.min_positive_weight_pct, 0.0) / 100.0
        if floor * eligible_count > 1.0:
            floor = 1.0 / eligible_count

        remaining = 1.0 - floor * eligible_count
        if remaining < 0:
            remaining = 0.0

        weight_frac = raw_score * 0.0
        weight_frac[eligible_mask] = floor
        if remaining > 0:
            if sum_raw > 0:
                prop = base_frac[eligible_mask]
            else:
                prop = pd.Series(1.0 / eligible_count, index=summary.index)[eligible_mask]
            weight_frac[eligible_mask] += prop * remaining

    summary["score"] = raw_score
    summary["confidence"] = confidence
    summary["weight_frac"] = weight_frac
    summary["weight_pct"] = summary["weight_frac"] * 100.0
    summary["recommended_exit_pct"] = summary["weight_frac"] * args.base_exit_pct

    return summary


def _build_pdf_report(
    pdf_path: str,
    summary: pd.DataFrame,
    chart_paths: list[str],
    args: argparse.Namespace,
    input_path: str,
    exit_df: pd.DataFrame,
    plt_module,
    pdf_pages_cls,
) -> None:
    total_exits = len(exit_df)
    total_pnl = float(exit_df["Net P&L USDT"].sum())
    total_pnl_pct = float(exit_df["Net P&L %"].sum())

    with pdf_pages_cls(pdf_path) as pdf:
        # Cover / metadata page (A4 portrait)
        fig = plt_module.figure(figsize=(8.27, 11.69))
        fig.suptitle("Exit Signal Performance Report", fontsize=16, y=0.98)
        meta_lines = [
            f"Input file: {input_path}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Type filter: {args.type_filter}",
            f"Excluded signals: {', '.join(args.exclude_signal) if args.exclude_signal else 'None'}",
            f"Min exits for charts: {args.min_exits}",
            f"Min exits for weights: {args.min_exits_weight}",
            f"Confidence k: {args.confidence_k}",
            f"Negative P&L threshold: {args.negative_pnl_threshold}",
            f"Negative expectancy threshold: {args.negative_expectancy_threshold}",
            f"Min positive weight %: {args.min_positive_weight_pct}",
            f"Base exit %: {args.base_exit_pct}",
            f"Total exits analyzed: {total_exits}",
            f"Total P&L (USDT): {total_pnl:.2f}",
            f"Total P&L (%): {total_pnl_pct:.2f}",
            f"Signals: {summary['Signal'].nunique()}",
        ]
        fig.text(0.05, 0.90, "\n".join(meta_lines), va="top", fontsize=10)
        pdf.savefig(fig)
        plt_module.close(fig)

        # Summary table page
        table_cols = [
            "Signal",
            "exits",
            "total_pnl_usdt",
            "avg_pnl_usdt",
            "expectancy_usdt",
            "win_rate",
            "profit_factor",
        ]
        if "weight_pct" in summary.columns:
            table_cols.append("weight_pct")
        if "recommended_exit_pct" in summary.columns:
            table_cols.append("recommended_exit_pct")
        display_df = summary[table_cols].copy()
        display_df["total_pnl_usdt"] = display_df["total_pnl_usdt"].map(lambda x: f"{x:.2f}")
        display_df["avg_pnl_usdt"] = display_df["avg_pnl_usdt"].map(lambda x: f"{x:.2f}")
        display_df["expectancy_usdt"] = display_df["expectancy_usdt"].map(lambda x: f"{x:.2f}")
        display_df["win_rate"] = display_df["win_rate"].map(lambda x: f"{x:.2%}")
        display_df["profit_factor"] = display_df["profit_factor"].map(
            lambda x: "inf" if np.isinf(x) else f"{x:.2f}"
        )
        if "weight_pct" in display_df.columns:
            display_df["weight_pct"] = display_df["weight_pct"].map(lambda x: f"{x:.1f}%")
        if "recommended_exit_pct" in display_df.columns:
            display_df["recommended_exit_pct"] = display_df["recommended_exit_pct"].map(
                lambda x: f"{x:.2f}%"
            )

        fig, ax = plt_module.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        ax.set_title("Summary (sorted by total P&L USDT)", fontsize=12, pad=12)
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.2)
        pdf.savefig(fig)
        plt_module.close(fig)

        # Chart pages
        for chart_path in chart_paths:
            if not os.path.exists(chart_path):
                continue
            fig, ax = plt_module.subplots(figsize=(11.69, 8.27))
            img = plt_module.imread(chart_path)
            ax.imshow(img)
            ax.axis("off")
            title = os.path.basename(chart_path).replace("_", " ").replace(".png", "").title()
            fig.suptitle(title, fontsize=14)
            pdf.savefig(fig)
            plt_module.close(fig)


def run_exit_signal_analysis(args: argparse.Namespace, verbose: bool = True) -> dict:
    df = pd.read_csv(args.input)
    _ensure_columns(df)

    df["Date and time"] = pd.to_datetime(df["Date and time"], errors="coerce")

    exit_df = df[df["Type"].str.contains(args.type_filter, case=False, na=False)].copy()
    exit_df = exit_df[exit_df["Signal"].notna()]
    exit_df["Signal"] = exit_df["Signal"].astype(str).str.strip()
    exit_df = exit_df[exit_df["Signal"] != ""]

    if args.exclude_signal:
        exit_df = exit_df[~exit_df["Signal"].isin(args.exclude_signal)]

    if exit_df.empty:
        raise SystemExit("No exit rows found after filtering.")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="DataFrameGroupBy.apply operated on the grouping columns",
        )
        try:
            summary = (
                exit_df.groupby("Signal", dropna=False)
                .apply(_signal_stats, include_groups=False)
                .reset_index()
                .sort_values("total_pnl_usdt", ascending=False)
            )
        except TypeError:
            summary = (
                exit_df.groupby("Signal", dropna=False)
                .apply(_signal_stats)
                .reset_index()
                .sort_values("total_pnl_usdt", ascending=False)
            )

    summary = _compute_weights(summary, args)

    outdir = _make_outdir(args.outdir)

    summary_path = os.path.join(outdir, "exit_signal_summary.csv")
    summary.to_csv(summary_path, index=False)

    weights_path = os.path.join(outdir, "exit_signal_weights.csv")
    summary.sort_values("weight_pct", ascending=False).to_csv(weights_path, index=False)

    if verbose:
        print("Exit signal summary (top 10 by total P&L USDT):")
        print(summary.head(10).to_string(index=False))
        print(f"\nSaved summary to: {summary_path}")
        print(f"Saved weights to: {weights_path}")

        print("\nExit signal weights (top 10 by weight %):")
        print(
            summary.sort_values("weight_pct", ascending=False)[
                [
                    "Signal",
                    "exits",
                    "total_pnl_usdt",
                    "expectancy_usdt",
                    "weight_pct",
                    "recommended_exit_pct",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    # Filter for charts with a minimum number of exits
    chart_summary = summary[summary["exits"] >= args.min_exits].copy()
    if chart_summary.empty:
        raise SystemExit("No signals meet min-exits threshold for charts.")

    # Add trade counts to labels so charts show sample size
    chart_summary["Signal_n"] = chart_summary.apply(
        lambda r: f"{r['Signal']} (n={int(r['exits'])})",
        axis=1,
    )
    label_map = dict(zip(chart_summary["Signal"], chart_summary["Signal_n"]))

    charts_ok, plt, sns, pdf_pages_cls, err = _load_plotting()
    if not charts_ok:
        if verbose:
            print("\nCharts and PDF report skipped: plotting dependencies are missing.")
            print("Install dependencies with: pip install -r apps/exit_signals/requirements.txt")
            print(f"Details: {err}")
        return {
            "summary": summary,
            "exit_df": exit_df,
            "outdir": outdir,
            "summary_path": summary_path,
            "weights_path": weights_path,
            "chart_paths": [],
            "pdf_path": None,
            "charts_ok": False,
            "plot_error": err,
        }

    sns.set_style("whitegrid")
    chart_paths: list[str] = []

    # 1) Total P&L by signal
    plt.figure(figsize=(10, 6))
    order = chart_summary.sort_values("total_pnl_usdt", ascending=True)["Signal_n"]
    sns.barplot(data=chart_summary, y="Signal_n", x="total_pnl_usdt", order=order, palette="viridis")
    plt.title("Total P&L (USDT) by Exit Signal")
    plt.xlabel("Total P&L (USDT)")
    plt.ylabel("Signal (n=exits)")
    chart_path = os.path.join(outdir, "total_pnl_by_signal.png")
    _save_fig(plt, chart_path)
    chart_paths.append(chart_path)

    # 2) Average P&L per exit by signal
    plt.figure(figsize=(10, 6))
    order = chart_summary.sort_values("avg_pnl_usdt", ascending=True)["Signal_n"]
    sns.barplot(data=chart_summary, y="Signal_n", x="avg_pnl_usdt", order=order, palette="mako")
    plt.title("Average P&L per Exit (USDT) by Signal")
    plt.xlabel("Average P&L per Exit (USDT)")
    plt.ylabel("Signal (n=exits)")
    chart_path = os.path.join(outdir, "avg_pnl_by_signal.png")
    _save_fig(plt, chart_path)
    chart_paths.append(chart_path)

    # 3) Win rate by signal
    plt.figure(figsize=(10, 6))
    order = chart_summary.sort_values("win_rate", ascending=True)["Signal_n"]
    sns.barplot(data=chart_summary, y="Signal_n", x="win_rate", order=order, palette="crest")
    plt.title("Win Rate by Exit Signal")
    plt.xlabel("Win Rate")
    plt.ylabel("Signal (n=exits)")
    plt.xlim(0, 1)
    chart_path = os.path.join(outdir, "win_rate_by_signal.png")
    _save_fig(plt, chart_path)
    chart_paths.append(chart_path)

    # 4) Weight by signal
    plt.figure(figsize=(10, 6))
    order = chart_summary.sort_values("weight_pct", ascending=True)["Signal_n"]
    sns.barplot(data=chart_summary, y="Signal_n", x="weight_pct", order=order, palette="flare")
    plt.title("Weight % by Exit Signal")
    plt.xlabel("Weight %")
    plt.ylabel("Signal (n=exits)")
    chart_path = os.path.join(outdir, "weight_pct_by_signal.png")
    _save_fig(plt, chart_path)
    chart_paths.append(chart_path)

    # 5) Distribution of P&L % by signal
    plt.figure(figsize=(11, 6))
    plot_box = exit_df[exit_df["Signal"].isin(chart_summary["Signal"])].copy()
    plot_box["Signal_n"] = plot_box["Signal"].map(label_map)
    sns.boxplot(
        data=plot_box,
        x="Net P&L %",
        y="Signal_n",
        order=chart_summary.sort_values("avg_pnl_pct", ascending=True)["Signal_n"],
        showfliers=False,
        palette="vlag",
    )
    plt.title("Distribution of Net P&L % by Exit Signal")
    plt.xlabel("Net P&L %")
    plt.ylabel("Signal (n=exits)")
    chart_path = os.path.join(outdir, "pnl_pct_distribution_by_signal.png")
    _save_fig(plt, chart_path)
    chart_paths.append(chart_path)

    # 6) Cumulative P&L by signal over time
    plt.figure(figsize=(12, 6))
    plot_df = exit_df.sort_values("Date and time")
    for signal in chart_summary["Signal"]:
        s = plot_df[plot_df["Signal"] == signal]
        if s.empty:
            continue
        s = s.copy()
        s["cum_pnl"] = s["Net P&L USDT"].cumsum()
        plt.plot(s["Date and time"], s["cum_pnl"], label=label_map.get(signal, signal))
    plt.title("Cumulative P&L (USDT) by Exit Signal")
    plt.xlabel("Date and time")
    plt.ylabel("Cumulative P&L (USDT)")
    plt.legend(loc="best", fontsize=8)
    chart_path = os.path.join(outdir, "cumulative_pnl_by_signal.png")
    _save_fig(plt, chart_path)
    chart_paths.append(chart_path)

    if verbose:
        print(f"\nCharts saved in: {outdir}")

    pdf_path = args.pdf_report or os.path.join(outdir, "exit_signal_report.pdf")
    _build_pdf_report(
        pdf_path=pdf_path,
        summary=summary,
        chart_paths=chart_paths,
        args=args,
        input_path=args.input,
        exit_df=exit_df,
        plt_module=plt,
        pdf_pages_cls=pdf_pages_cls,
    )
    if verbose:
        print(f"PDF report saved to: {pdf_path}")

    return {
        "summary": summary,
        "exit_df": exit_df,
        "outdir": outdir,
        "summary_path": summary_path,
        "weights_path": weights_path,
        "chart_paths": chart_paths,
        "pdf_path": pdf_path,
        "charts_ok": True,
        "plot_error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare performance of exit signals.")
    parser.add_argument("--input", required=True, help="Path to TradingView CSV")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for reports (default: reports/exit_signals_TIMESTAMP)",
    )
    parser.add_argument(
        "--type-filter",
        default="Exit",
        help="Filter rows where Type contains this text (default: Exit)",
    )
    parser.add_argument(
        "--exclude-signal",
        action="append",
        default=[],
        help="Exclude signal name (repeatable)",
    )
    parser.add_argument(
        "--min-exits",
        type=int,
        default=1,
        help="Minimum exits per signal to be included in charts (default: 1)",
    )
    parser.add_argument(
        "--pdf-report",
        default=None,
        help="Path to PDF report (default: <outdir>/exit_signal_report.pdf)",
    )
    parser.add_argument(
        "--min-exits-weight",
        type=int,
        default=20,
        help="Minimum exits per signal to be assigned a weight (default: 20)",
    )
    parser.add_argument(
        "--min-positive-weight-pct",
        type=float,
        default=1.0,
        help="Minimum weight % for non-negative signals (default: 1.0)",
    )
    parser.add_argument(
        "--base-exit-pct",
        type=float,
        default=10.0,
        help="Base exit percent of position to distribute across signals (default: 10)",
    )
    parser.add_argument(
        "--confidence-k",
        type=float,
        default=50.0,
        help="Confidence scaling factor for sample size (default: 50)",
    )
    parser.add_argument(
        "--negative-pnl-threshold",
        type=float,
        default=0.0,
        help="Signals with total P&L <= this get weight 0 (default: 0)",
    )
    parser.add_argument(
        "--negative-expectancy-threshold",
        type=float,
        default=0.0,
        help="Signals with expectancy <= this get weight 0 (default: 0)",
    )
    parser.add_argument(
        "--w-expectancy",
        type=float,
        default=0.3,
        help="Weight for expectancy in scoring (default: 0.30)",
    )
    parser.add_argument(
        "--w-winrate",
        type=float,
        default=0.15,
        help="Weight for win rate in scoring (default: 0.15)",
    )
    parser.add_argument(
        "--w-profit-factor",
        type=float,
        default=0.1,
        help="Weight for profit factor in scoring (default: 0.10)",
    )
    parser.add_argument(
        "--w-sharpe",
        type=float,
        default=0.15,
        help="Weight for sharpe-like metric in scoring (default: 0.15)",
    )
    parser.add_argument(
        "--w-drawdown",
        type=float,
        default=0.1,
        help="Weight for max drawdown (lower is better) (default: 0.10)",
    )
    parser.add_argument(
        "--w-total-pnl",
        type=float,
        default=0.1,
        help="Weight for total P&L (default: 0.10)",
    )
    parser.add_argument(
        "--w-stability",
        type=float,
        default=0.1,
        help="Weight for stability (lower downside deviation is better) (default: 0.10)",
    )
    args = parser.parse_args()

    run_exit_signal_analysis(args, verbose=True)


if __name__ == "__main__":
    main()
