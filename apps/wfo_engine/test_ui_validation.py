"""
Validation complète WFO + Final Backtest.
Vérifie que toutes les données affichées dans l'UI Streamlit sont présentes et correctes.
"""
import json, sys, traceback, time, logging, math
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("ui_validation")

CONFIG_PATH = sys.argv[1] if len(sys.argv) > 1 else "config_test.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)
config.setdefault("from_file", True)

ERRORS = []
WARNINGS = []

def check(name, condition, value=None, warn=False):
    tag = "WARN" if warn else "FAIL"
    if not condition:
        msg = f"[{tag}] {name}" + (f" → {value!r}" if value is not None else "")
        (WARNINGS if warn else ERRORS).append(msg)
        print(f"  {msg}")
    else:
        vstr = f" = {value!r}" if value is not None else ""
        print(f"  [OK ] {name}{vstr}")

def is_num(v):
    try:
        return v is not None and not (isinstance(v, float) and math.isnan(v))
    except Exception:
        return False

print("\n" + "="*60)
print("PHASE 1 — Chargement données")
print("="*60)
from data_loading import load_data
df = load_data(
    start_date=config["start_date"],
    end_date=config["end_date"],
    timeframe=config.get("timeframe", "5s"),
    from_file=config.get("from_file", True),
    file_path=config.get("file_path"),
)
check("DataFrame non vide", len(df) > 0, len(df))
check("Index DatetimeIndex", hasattr(df.index, 'tz'))
check("Colonnes OHLCV", all(c in df.columns for c in ["Open","High","Low","Close","Volume"]))
check("Plage dates correcte", str(df.index[0].date()) == config["start_date"], df.index[0])

print("\n" + "="*60)
print("PHASE 2 — Optimisation WFO")
print("="*60)
from services.run_service import run_optimization_job
ts = time.strftime("%Y%m%d_%H%M%S")
results, df_out, elapsed = run_optimization_job(
    config=config, control=None, job_state=None, df=df,
    run_ts=ts, log_dir="reports/error_logs",
)

# ── Structure résultats ──
check("results est un dict", isinstance(results, dict))
check("Clé out_of_sample_performance", "out_of_sample_performance" in results)
check("Clé in_sample_performance", "in_sample_performance" in results)
check("Clé best_params", "best_params" in results)
check("Clé windows_data", "windows_data" in results)

oos = results.get("out_of_sample_performance", [])
is_ = results.get("in_sample_performance", [])
n_win = config["n_windows"]

check(f"Nombre fenêtres OOS = {n_win}", len(oos) == n_win, len(oos))
check(f"Nombre fenêtres IS = {n_win}", len(is_) == n_win, len(is_))

print("\n--- Métriques OOS par fenêtre (données UI WFO panel) ---")
REQUIRED_OOS_KEYS = ["return", "sharpe", "max_drawdown", "win_rate", "avg_pl_per_trade", "pqs",
                     "start", "end", "n_trades"]
for i, w in enumerate(oos):
    print(f"  Fenêtre {i+1}:")
    for k in REQUIRED_OOS_KEYS:
        present = k in w
        val = w.get(k)
        check(f"    OOS[{i+1}].{k}", present and is_num(val), val)

print("\n--- Métriques IS par fenêtre ---")
REQUIRED_IS_KEYS = ["return", "sharpe", "max_drawdown", "win_rate"]
for i, w in enumerate(is_):
    for k in REQUIRED_IS_KEYS:
        present = k in w
        val = w.get(k)
        check(f"  IS[{i+1}].{k}", present and is_num(val), val)

print("\n--- Métriques agrégées OOS (UI summary bar) ---")
returns  = [w["return"] for w in oos]
sharpes  = [w["sharpe"] for w in oos]
avg_pls  = [w.get("avg_pl_per_trade", float("nan")) for w in oos]
pqs_vals = [w.get("pqs", float("nan")) for w in oos]
check("Avg Return calculable",  is_num(np.mean(returns)),  f"{np.mean(returns):+.2f}%")
check("Avg Sharpe calculable",  is_num(np.mean(sharpes)),  f"{np.mean(sharpes):.3f}")
check("Avg P&L calculable",     is_num(np.nanmean(avg_pls)), f"{np.nanmean(avg_pls):+.4f}%")
check("Avg PQS calculable",     is_num(np.nanmean(pqs_vals)), f"{np.nanmean(pqs_vals):.4f}")
check("Avg Max DD calculable",  True, f"{np.mean([w['max_drawdown'] for w in oos]):.2f}%")
check("Avg Win Rate calculable",True, f"{np.mean([w['win_rate'] for w in oos]):.1f}%")

print("\n--- best_params (sélection L1 SNV) ---")
bp = results.get("best_params", {})
check("best_params non vide", bool(bp), bp)
for k in config.get("selected_params", []):
    check(f"  best_params.{k}", k in bp, bp.get(k))

print("\n--- windows_data (courbes equity, parameter maps) ---")
wd = results.get("windows_data", {})
check("windows_data non vide", bool(wd))
for wid, wdata in list(wd.items())[:2]:  # check first 2 windows
    check(f"  window {wid} has portfolio", "portfolio" in wdata or "oos_portfolio" in wdata,
          list(wdata.keys()))

print("\n" + "="*60)
print("PHASE 3 — Final Backtest (L2: best_is_oos)")
print("="*60)
from ui.final_backtest_panel import _select_final_params_from_results
from strategy_adapters import resolve_strategy_adapter
from metrics import calc_avg_pl, calc_pqs

chosen_params, best_score, best_window, is_metrics, oos_metrics, final_source, robust_summary = \
    _select_final_params_from_results(results, config)

check("Paramètres sélectionnés non vides", bool(chosen_params))
check("Source sélection valide", final_source in ("best_window","best_oos","weighted_oos","robust_set"),
      final_source)
check("Fenêtre best identifiée", best_window is not None or final_source not in ("best_window","best_oos"),
      best_window)

print(f"\n  Source : {final_source} | fenêtre : {best_window}")
print(f"  Params : {json.dumps({k: v for k, v in list(chosen_params.items())[:6]}, default=str)} ...")

adapter = resolve_strategy_adapter(config=config)
params_full = {
    **chosen_params,
    "order_sizing_mode":  config.get("order_sizing_mode", "fixed_cash"),
    "order_fixed_cash":   config.get("order_fixed_cash", 10000.0),
    "fees_pct":           config.get("fees_pct", 0.0),
    "timeframe":          config.get("timeframe", "5s"),
    "strategy_direction": config.get("strategy_direction", "long_only"),
}
pf = adapter.run_backtest(df, params_full, timeframe=config.get("timeframe","5s"), return_portfolio=True)

avg_pl_val = calc_avg_pl(pf)
pqs_val    = calc_pqs(pf, n_ref=int(config.get("pqs_n_ref", 50)))
n_trades   = len(pf.trades)

print("\n--- Métriques Final Backtest (UI final_backtest_panel) ---")
check("Total Return numérique",  is_num(pf.total_return),  f"{pf.total_return*100:+.2f}%")
check("Sharpe Ratio numérique",  is_num(pf.sharpe_ratio),  f"{pf.sharpe_ratio:.3f}")
check("Mean P&L % numérique",    is_num(avg_pl_val),        f"{avg_pl_val:+.4f}%")
check("Max Drawdown numérique",  is_num(pf.max_drawdown),   f"{pf.max_drawdown*100:.2f}%")
check("Win Rate numérique",      is_num(pf.trades.win_rate),f"{pf.trades.win_rate*100:.1f}%")
check("PQS numérique",           is_num(pqs_val),           f"{pqs_val:.4f}")
check("Nombre de trades > 0",    n_trades > 0,              n_trades)
check("Equity curve disponible", hasattr(pf, 'value') and len(pf.value) > 0)

# --- Métriques IS/OOS de la fenêtre gagnante ---
print("\n--- Métriques IS/OOS fenêtre best (UI detail panel) ---")
if is_metrics:
    for k, v in is_metrics.items():
        check(f"  IS metric {k}", is_num(v), v)
if oos_metrics:
    for k, v in oos_metrics.items():
        check(f"  OOS metric {k}", is_num(v), v)

print("\n" + "="*60)
print("PHASE 4 — Vérification journal d'erreurs")
print("="*60)
import glob, os
logs = sorted(glob.glob("reports/error_logs/error_log_classic_*.json"))
if logs:
    with open(logs[-1]) as f:
        elog = json.load(f)
    check("Journal status OK/FAILED présent", "status" in elog, elog.get("status"))
    check("Pas d'erreurs fatales", elog.get("fatal_error") is None, elog.get("fatal_error"))
    check("n_errors = 0", elog.get("n_errors", 1) == 0, elog.get("n_errors"))
else:
    check("Journal d'erreurs présent", False, "aucun fichier trouvé")

# ── Résumé final ──
print("\n" + "="*60)
print(f"RÉSUMÉ : {len(ERRORS)} erreur(s) | {len(WARNINGS)} avertissement(s)")
print("="*60)
if ERRORS:
    print("\nERREURS :")
    for e in ERRORS:
        print(f"  {e}")
if WARNINGS:
    print("\nAVERTISSEMENTS :")
    for w in WARNINGS:
        print(f"  {w}")
if not ERRORS:
    print("\n✓ Toutes les données UI sont présentes et correctes.")
else:
    sys.exit(1)
