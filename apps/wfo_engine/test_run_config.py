"""Headless test: WFO + Final Backtest avec un fichier de config JSON."""
import json
import sys
import traceback
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_test.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)

config.setdefault("from_file", True)

print("=" * 60)
print("CONFIG")
print(f"  timeframe={config['timeframe']} | method={config['optimization_method']}")
print(f"  windows={config['n_windows']} | trials={config['max_trials']}")
print(f"  metric={config['metric1_name']} | cross_window={config['cross_window_method']}")
print(f"  data={config['file_path']}")
print("=" * 60)

# ── 1. CHARGEMENT DONNÉES ──────────────────────────────────────────────────
try:
    from data_loading import load_data
    print("\n[1/3] Chargement données...")
    df = load_data(
        start_date=config["start_date"],
        end_date=config["end_date"],
        timeframe=config.get("timeframe", "5s"),
        from_file=config.get("from_file", True),
        file_path=config.get("file_path"),
    )
    print(f"  OK — {len(df)} bars | {df.index[0]} → {df.index[-1]}")
except Exception:
    print("ERREUR chargement données:")
    traceback.print_exc()
    sys.exit(1)

# ── 2. WFO ────────────────────────────────────────────────────────────────
try:
    from services.run_service import run_optimization_job
    print("\n[2/3] WFO en cours...")
    ts = time.strftime("%Y%m%d_%H%M%S")
    results, df_out, elapsed = run_optimization_job(
        config=config,
        control=None,
        job_state=None,
        df=df,
        run_ts=ts,
        log_dir="reports/error_logs",
    )
    print(f"  OK — {len(results.get('out_of_sample_performance', []))} fenêtres | {elapsed:.1f}s")
    oos_perf = results.get("out_of_sample_performance", [])
    if oos_perf:
        import numpy as np
        returns = [w.get("return", 0) for w in oos_perf]
        sharpes = [w.get("sharpe", 0) for w in oos_perf]
        avg_pl  = [w.get("avg_pl_per_trade", float("nan")) for w in oos_perf]
        pqs     = [w.get("pqs", float("nan")) for w in oos_perf]
        print(f"  OOS — Avg Return: {np.mean(returns):+.2f}% | Avg Sharpe: {np.mean(sharpes):.3f}"
              f" | Avg P&L: {np.nanmean(avg_pl):+.4f}% | Avg PQS: {np.nanmean(pqs):.4f}")
except Exception:
    print("ERREUR WFO:")
    traceback.print_exc()
    sys.exit(1)

# ── 3. FINAL BACKTEST ─────────────────────────────────────────────────────
try:
    import numpy as np
    from ui.final_backtest_panel import _select_final_params_from_results
    from strategy_adapters import resolve_strategy_adapter
    from metrics import calc_avg_pl, calc_pqs

    print(f"\n[3/3] Final Backtest (méthode L2: {config.get('cross_window_method', 'best_is_oos')})...")

    chosen_params, best_score, best_window, is_metrics, oos_metrics, final_source, robust_summary = \
        _select_final_params_from_results(results, config)

    print(f"  Source paramètres : {final_source}"
          + (f" (fenêtre {best_window})" if best_window is not None else ""))
    print(f"  Paramètres sélectionnés :")
    for k, v in chosen_params.items():
        print(f"    {k} = {v}")

    # Charger données pour la période complète (même plage ici)
    df_final = load_data(
        start_date=config["start_date"],
        end_date=config["end_date"],
        timeframe=config.get("timeframe", "5s"),
        from_file=config.get("from_file", True),
        file_path=config.get("file_path"),
    )

    adapter = resolve_strategy_adapter(config=config)
    params_full = {**chosen_params, **{
        "order_sizing_mode": config.get("order_sizing_mode", "fixed_cash"),
        "order_fixed_cash":  config.get("order_fixed_cash", 10000.0),
        "fees_pct":          config.get("fees_pct", 0.0),
        "timeframe":         config.get("timeframe", "5s"),
        "strategy_direction": config.get("strategy_direction", "long_only"),
    }}

    pf = adapter.run_backtest(df_final, params_full, timeframe=config.get("timeframe", "5s"),
                               return_portfolio=True)

    avg_pl_val = calc_avg_pl(pf)
    pqs_val    = calc_pqs(pf, n_ref=int(config.get("pqs_n_ref", 50)))
    n_trades   = len(pf.trades)

    print("\n  ── Résultats Final Backtest ──")
    print(f"  Total Return   : {pf.total_return * 100:+.2f}%")
    print(f"  Sharpe Ratio   : {pf.sharpe_ratio:.3f}")
    print(f"  Mean P&L %     : {avg_pl_val:+.4f}%")
    print(f"  Max Drawdown   : {pf.max_drawdown * 100:.2f}%")
    print(f"  Win Rate       : {pf.trades.win_rate * 100:.1f}%")
    print(f"  PQS            : {pqs_val:.4f}")
    print(f"  Trades         : {n_trades}")

except Exception:
    print("ERREUR Final Backtest:")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("TEST TERMINÉ SANS ERREUR")
print("=" * 60)
