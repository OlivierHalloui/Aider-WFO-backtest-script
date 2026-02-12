"""Adaptive continuous optimization engine.

This module keeps a rolling memory of parameter-value performance and
progressively narrows/refreshes the active grid without fixed WFO windows.
"""

import time
from datetime import timedelta

import numpy as np
import pandas as pd

from config import WFOSettings
from strategy_adapters import resolve_strategy_adapter


def _value_token(value):
    if isinstance(value, (bool, np.bool_)):
        return ("bool", int(bool(value)))
    if isinstance(value, (np.integer, int)):
        return ("int", int(value))
    if isinstance(value, (np.floating, float)):
        return ("float", round(float(value), 12))
    return ("str", str(value))


def _to_scalar_score(score):
    if score is None:
        return np.nan
    if np.isscalar(score):
        try:
            out = float(score)
            if np.isnan(out) or np.isinf(out):
                return np.nan
            return out
        except Exception:
            return np.nan
    try:
        if hasattr(score, "values"):
            arr = np.asarray(score.values, dtype=float)
        else:
            arr = np.asarray(score, dtype=float)
        if arr.size == 0:
            return np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.mean(arr))
    except Exception:
        return np.nan


def _python_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _param_grid_combinations(param_grid):
    if not param_grid:
        return 0
    lengths = [len(v) for v in param_grid.values()]
    if any(length <= 0 for length in lengths):
        return 0
    return int(np.prod(lengths))


def _random_candidates(param_grid, n_candidates, rng):
    if not param_grid or n_candidates <= 0:
        return []
    for values in param_grid.values():
        if not values:
            return []

    param_names = list(param_grid.keys())
    unique = set()
    candidates = []
    max_attempts = max(n_candidates * 8, 200)
    attempts = 0
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        candidate = {}
        key_parts = []
        for name in param_names:
            values = param_grid[name]
            idx = int(rng.integers(0, len(values)))
            value = values[idx]
            candidate[name] = value
            key_parts.append((name, _value_token(value)))
        key = tuple(key_parts)
        if key in unique:
            continue
        unique.add(key)
        candidates.append(candidate)
    return candidates


def _trade_stat(trades, attr_name, default=0.0):
    value = getattr(trades, attr_name, None)
    if value is not None:
        return value
    try:
        stats = trades.stats()
    except Exception:
        return default
    keys = [
        attr_name,
        attr_name.replace("_", " "),
        attr_name.replace("_", " ").title(),
        attr_name.replace("_", " ").capitalize(),
    ]
    for key in keys:
        try:
            value = stats.get(key) if hasattr(stats, "get") else stats[key]
        except Exception:
            value = None
        if value is not None:
            return value
    return default


def _calc_avg_pl(port):
    """Return average P/L per trade for scalar or vectorized portfolios."""
    try:
        total_ret = port.total_return * 100
        n_trades = port.trades.count()
        if n_trades is None:
            return 0.0

        if hasattr(n_trades, "replace"):
            safe_trades = n_trades.replace(0, np.nan)
            avg_pl = total_ret / safe_trades
            avg_pl = avg_pl.replace([np.inf, -np.inf], 0).fillna(0)
            return avg_pl

        if float(n_trades) == 0.0:
            return 0.0
        avg_pl = total_ret / n_trades
        if hasattr(avg_pl, "replace"):
            avg_pl = avg_pl.replace([np.inf, -np.inf], 0).fillna(0)
        else:
            if np.isinf(avg_pl) or np.isnan(avg_pl):
                avg_pl = 0.0
        return avg_pl
    except Exception:
        return 0.0


def _portfolio_metrics(portfolio, cycle_id):
    try:
        n_trades = len(portfolio.trades)
    except Exception:
        n_trades = 0
    return {
        "window": cycle_id,
        "return": float(_to_scalar_score(getattr(portfolio, "total_return", 0.0) * 100)),
        "sharpe": float(_to_scalar_score(getattr(portfolio, "sharpe_ratio", 0.0))),
        "max_drawdown": float(_to_scalar_score(getattr(portfolio, "max_drawdown", 0.0) * 100)),
        "win_rate": float(_to_scalar_score(getattr(getattr(portfolio, "trades", object()), "win_rate", 0.0))),
        "avg_gain_per_trade": float(_to_scalar_score(_trade_stat(portfolio.trades, "avg_winning_trade"))),
        "avg_loss_per_trade": float(_to_scalar_score(_trade_stat(portfolio.trades, "avg_losing_trade"))),
        "avg_pl_per_trade": float(_to_scalar_score(_calc_avg_pl(portfolio))),
        "calmar_ratio": float(_to_scalar_score(getattr(portfolio, "calmar_ratio", 0.0))),
        "sortino_ratio": float(_to_scalar_score(getattr(portfolio, "sortino_ratio", 0.0))),
        "n_trades": int(n_trades),
    }


class AdaptiveValueModel:
    """Tracks per-parameter value statistics and builds the next active grid."""

    def __init__(self, base_param_grid, settings, rng):
        self.base_param_grid = {k: list(v) for k, v in base_param_grid.items()}
        self.param_names = list(self.base_param_grid.keys())
        self.rng = rng
        self.decay = float(getattr(settings, "adaptive_decay", 0.98))
        self.keep_ratio = float(getattr(settings, "adaptive_keep_ratio", 0.40))
        self.exploration_ratio = float(getattr(settings, "adaptive_exploration_ratio", 0.20))
        self.min_values = int(getattr(settings, "adaptive_min_values_per_param", 2))
        self.ucb_beta = float(getattr(settings, "adaptive_ucb_beta", 0.75))
        self.warmup_trials = int(getattr(settings, "adaptive_warmup_trials", 300))

        self.sum_w = {}
        self.sum_score = {}
        self.sum_sq = {}
        self.token_to_index = {}
        self.total_trials = 0.0

        for name, values in self.base_param_grid.items():
            self.sum_w[name] = np.zeros(len(values), dtype=float)
            self.sum_score[name] = np.zeros(len(values), dtype=float)
            self.sum_sq[name] = np.zeros(len(values), dtype=float)
            mapping = {}
            for idx, value in enumerate(values):
                mapping[_value_token(value)] = idx
            self.token_to_index[name] = mapping

    def _stats_arrays(self, name):
        w = self.sum_w[name]
        score = self.sum_score[name]
        sq = self.sum_sq[name]
        mean = np.zeros_like(w)
        var = np.ones_like(w)
        mask = w > 1e-12
        mean[mask] = score[mask] / w[mask]
        var[mask] = np.maximum(sq[mask] / w[mask] - mean[mask] ** 2, 1e-6)
        return mean, var, w

    def apply_decay(self):
        """Apply exponential forgetting so recent trials matter more."""
        decay = min(max(self.decay, 0.0), 1.0)
        if decay >= 0.999999:
            return
        for name in self.param_names:
            self.sum_w[name] *= decay
            self.sum_score[name] *= decay
            self.sum_sq[name] *= decay
        self.total_trials *= decay

    def update_single(self, params, score, weight=1.0):
        """Update statistics for one evaluated parameter combination."""
        score_f = _to_scalar_score(score)
        if np.isnan(score_f) or np.isinf(score_f):
            return
        w = float(max(weight, 0.0))
        if w <= 0:
            return
        for name in self.param_names:
            if name not in params:
                continue
            token = _value_token(params[name])
            idx = self.token_to_index[name].get(token)
            if idx is None:
                continue
            self.sum_w[name][idx] += w
            self.sum_score[name][idx] += w * score_f
            self.sum_sq[name][idx] += w * score_f * score_f
        self.total_trials += w

    def update_from_trials(self, trials_df, score_col="combined_score"):
        if trials_df is None or trials_df.empty or score_col not in trials_df.columns:
            return
        for _, row in trials_df.iterrows():
            self.update_single(row.to_dict(), row.get(score_col), weight=1.0)

    def score_candidate_thompson(self, candidate):
        total = 0.0
        used = 0
        for name in self.param_names:
            token = _value_token(candidate.get(name))
            idx = self.token_to_index[name].get(token)
            if idx is None:
                continue
            mean, var, count = self._stats_arrays(name)
            c = count[idx]
            m = mean[idx] if c > 0 else 0.0
            v = var[idx] if c > 0 else 1.0
            sigma = np.sqrt(max(v, 1e-6) / (c + 1.0))
            total += float(self.rng.normal(m, sigma))
            used += 1
        if used == 0:
            return 0.0
        return total / used

    def parameter_weights(self):
        raw = {}
        for name in self.param_names:
            mean, _, count = self._stats_arrays(name)
            mask = count > 0
            if not np.any(mask):
                raw[name] = 0.0
                continue
            spread = float(np.nanmax(mean[mask]) - np.nanmin(mean[mask]))
            raw[name] = max(spread, 0.0)
        total = float(sum(raw.values()))
        if total <= 0:
            if not raw:
                return {}
            uniform = 1.0 / len(raw)
            return {k: uniform for k in raw.keys()}
        return {k: float(v / total) for k, v in raw.items()}

    def build_active_grid(self, last_best_params=None):
        """Build next-cycle grid by balancing exploitation and exploration."""
        baseline_combos = _param_grid_combinations(self.base_param_grid)
        if self.total_trials < self.warmup_trials:
            return self.base_param_grid, {
                "enabled": False,
                "reason": "warmup",
                "historical_trials": float(self.total_trials),
                "baseline_combinations": baseline_combos,
                "guided_combinations": baseline_combos,
                "parameter_weights": self.parameter_weights(),
            }

        active_grid = {}
        for name, values in self.base_param_grid.items():
            if len(values) <= 1:
                active_grid[name] = list(values)
                continue

            mean, var, count = self._stats_arrays(name)
            uncertainty = np.sqrt(np.maximum(var, 1e-6) / (count + 1.0))
            rank_score = mean + self.ucb_beta * uncertainty

            keep_n = max(self.min_values, int(np.ceil(len(values) * self.keep_ratio)))
            keep_n = min(len(values), keep_n)
            top_idx = np.argsort(-rank_score)[:keep_n]
            selected_idx = set(int(i) for i in top_idx.tolist())

            if isinstance(last_best_params, dict) and name in last_best_params:
                best_token = _value_token(last_best_params[name])
                best_idx = self.token_to_index[name].get(best_token)
                if best_idx is not None:
                    selected_idx.add(int(best_idx))

            # Keep some random values to avoid premature collapse of the search space.
            explore_n = max(1, int(np.ceil(len(values) * self.exploration_ratio)))
            remaining = [i for i in range(len(values)) if i not in selected_idx]
            if remaining:
                self.rng.shuffle(remaining)
                for idx in remaining[:explore_n]:
                    selected_idx.add(int(idx))

            active_values = [values[i] for i in range(len(values)) if i in selected_idx]
            active_grid[name] = active_values if active_values else list(values)

        guided_combos = _param_grid_combinations(active_grid)
        return active_grid, {
            "enabled": True,
            "historical_trials": float(self.total_trials),
            "baseline_combinations": baseline_combos,
            "guided_combinations": guided_combos,
            "reduction_ratio": 1.0 - (guided_combos / baseline_combos) if baseline_combos > 0 else 0.0,
            "parameter_weights": self.parameter_weights(),
        }

    def summarize_top_values(self, top_n=5):
        summary = {}
        top_n = max(1, int(top_n))
        for name, values in self.base_param_grid.items():
            mean, var, count = self._stats_arrays(name)
            rows = []
            for idx, value in enumerate(values):
                rows.append({
                    "value": _python_scalar(value),
                    "mean_score": float(mean[idx]),
                    "std_score": float(np.sqrt(max(var[idx], 1e-6))),
                    "effective_trials": float(count[idx]),
                })
            rows.sort(key=lambda x: x["mean_score"], reverse=True)
            summary[name] = rows[:top_n]
        return summary


def adaptive_continuous_optimization(
    df,
    param_grid=None,
    metrics_info=None,
    timeframe="5s",
    settings=None,
    status_callback=None,
    control=None,
    strategy_adapter=None,
):
    """Run adaptive cycles on a rolling train/OOS split until data is exhausted."""
    if settings is None:
        settings = WFOSettings()
    if strategy_adapter is None:
        strategy_adapter = resolve_strategy_adapter(strategy_mode='native_atdmf')
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("df must be a non-empty pandas DataFrame.")
    if param_grid is None or not isinstance(param_grid, dict) or not param_grid:
        raise ValueError("param_grid must be a non-empty dict for adaptive mode.")
    if metrics_info is None:
        metrics_info = {
            "metric1_name": settings.optimization_metric,
            "metric2_name": settings.secondary_metric,
            "weight_metric1": settings.metric_weights[0],
            "weight_metric2": settings.metric_weights[1],
        }

    train_bars = int(getattr(settings, "adaptive_train_bars", 5000))
    cycle_bars = int(getattr(settings, "adaptive_cycle_bars", 1000))
    trials_per_cycle = int(getattr(settings, "adaptive_trials_per_cycle", 150))
    candidate_pool_size = int(getattr(settings, "adaptive_candidate_pool_size", 3000))
    exploration_ratio = float(getattr(settings, "adaptive_exploration_ratio", 0.20))
    max_cycles = int(getattr(settings, "adaptive_max_cycles", 0))
    oos_weight = float(getattr(settings, "adaptive_oos_weight", 2.0))

    total_rows = len(df)
    train_bars = max(50, min(train_bars, max(50, total_rows - 2)))
    cycle_bars = max(1, cycle_bars)
    trials_per_cycle = max(5, trials_per_cycle)
    candidate_pool_size = max(trials_per_cycle * 3, candidate_pool_size)
    exploration_ratio = min(max(exploration_ratio, 0.0), 0.9)

    baseline_combinations = _param_grid_combinations(param_grid)
    rng = np.random.default_rng(int(getattr(settings, "random_state", 42)))
    model = AdaptiveValueModel(param_grid, settings, rng)

    start_time = time.time()
    cycle_times = []
    trial_counts = []
    last_best_params = None

    results = {
        "mode": "adaptive_continuous",
        "window_results": [],
        "cycle_results": [],
        "in_sample_performance": [],
        "out_of_sample_performance": [],
        "best_params": [],
        "nn_guidance": [],
        "prev_best_guidance": [],
        "adaptive_guidance": [],
        "settings": {
            "n_windows": 0,
            "optimization_method": settings.optimization_method,
            "optimization_regime": "adaptive_continuous",
            "adaptive_train_bars": train_bars,
            "adaptive_cycle_bars": cycle_bars,
            "adaptive_trials_per_cycle": trials_per_cycle,
            "adaptive_candidate_pool_size": candidate_pool_size,
            "adaptive_keep_ratio": float(getattr(settings, "adaptive_keep_ratio", 0.40)),
            "adaptive_exploration_ratio": exploration_ratio,
            "adaptive_min_values_per_param": int(getattr(settings, "adaptive_min_values_per_param", 2)),
            "adaptive_decay": float(getattr(settings, "adaptive_decay", 0.98)),
            "adaptive_ucb_beta": float(getattr(settings, "adaptive_ucb_beta", 0.75)),
            "adaptive_warmup_trials": int(getattr(settings, "adaptive_warmup_trials", 300)),
            "adaptive_max_cycles": max_cycles,
            "adaptive_oos_weight": oos_weight,
            "parallel_backend": settings.parallel_backend,
            "use_numba": settings.use_numba,
            "baseline_param_combinations": baseline_combinations,
        },
    }

    def log(message):
        print(message)
        if status_callback:
            status_callback(message)

    def report(payload):
        if status_callback:
            status_callback(payload)

    log("Starting Adaptive Continuous Optimization")
    log(f"Baseline parameter combinations: {baseline_combinations}")
    log(f"Train bars: {train_bars} | Cycle bars: {cycle_bars} | Trials/cycle: {trials_per_cycle}")

    pointer = train_bars
    cycle_id = 1
    while pointer < total_rows - 1:
        if max_cycles > 0 and cycle_id > max_cycles:
            break
        if control and control.should_stop():
            log("Stop requested. Exiting adaptive optimization loop.")
            break

        cycle_start = time.time()
        train_start = max(0, pointer - train_bars)
        train_end = pointer
        oos_end = min(total_rows, pointer + cycle_bars)

        train_df = df.iloc[train_start:train_end].copy()
        oos_df = df.iloc[pointer:oos_end].copy()
        if train_df.empty or oos_df.empty:
            break

        # Step 1: age historical memory, then derive the active grid for this cycle.
        model.apply_decay()
        active_grid, guidance_info = model.build_active_grid(last_best_params=last_best_params)
        active_combinations = _param_grid_combinations(active_grid)

        candidates = _random_candidates(active_grid, candidate_pool_size, rng)
        if not candidates:
            break

        # Step 2: rank candidates using stochastic value estimates (Thompson-style).
        ranked = [(model.score_candidate_thompson(c), c) for c in candidates]
        ranked.sort(key=lambda x: x[0], reverse=True)

        explore_trials = int(np.ceil(trials_per_cycle * exploration_ratio))
        exploit_trials = max(1, trials_per_cycle - explore_trials)

        selected = []
        selected_keys = set()
        for _, candidate in ranked:
            key = tuple((_value_token(candidate[name])) for name in param_grid.keys())
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(candidate)
            if len(selected) >= exploit_trials:
                break

        if explore_trials > 0:
            explore_candidates = _random_candidates(param_grid, explore_trials * 6, rng)
            for candidate in explore_candidates:
                key = tuple((_value_token(candidate[name])) for name in param_grid.keys())
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                selected.append(candidate)
                if len(selected) >= trials_per_cycle:
                    break

        if len(selected) < trials_per_cycle:
            for _, candidate in ranked:
                key = tuple((_value_token(candidate[name])) for name in param_grid.keys())
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                selected.append(candidate)
                if len(selected) >= trials_per_cycle:
                    break

        # Step 3: evaluate a mixed batch (top-ranked + exploration).
        selected = selected[:trials_per_cycle]
        trials_rows = []
        for candidate in selected:
            if control and control.should_stop():
                break
            params_eval = candidate.copy()
            params_eval.update(metrics_info)
            score = strategy_adapter.run_backtest(
                train_df,
                params_eval,
                timeframe=timeframe,
                return_portfolio=False,
            )
            score_f = _to_scalar_score(score)
            if np.isnan(score_f) or np.isinf(score_f):
                continue
            row = {k: _python_scalar(v) for k, v in candidate.items()}
            row["combined_score"] = float(score_f)
            trials_rows.append(row)

        if not trials_rows:
            pointer = oos_end
            cycle_id += 1
            continue

        trials_df = pd.DataFrame(trials_rows).sort_values("combined_score", ascending=False).reset_index(drop=True)
        best_row = trials_df.iloc[0]
        best_params = {k: _python_scalar(best_row[k]) for k in param_grid.keys() if k in best_row}

        in_sample_portfolio = strategy_adapter.run_backtest(
            train_df, best_params, timeframe=timeframe, return_portfolio=True
        )
        in_sample_metrics = _portfolio_metrics(in_sample_portfolio, cycle_id)
        out_sample_portfolio = strategy_adapter.run_backtest(
            oos_df, best_params, timeframe=timeframe, return_portfolio=True
        )
        out_sample_metrics = _portfolio_metrics(out_sample_portfolio, cycle_id)

        results["in_sample_performance"].append(in_sample_metrics)
        results["out_of_sample_performance"].append(out_sample_metrics)
        results["best_params"].append(best_params)

        # Step 4: update memory with in-sample trials, then reinforce with OOS result.
        model.update_from_trials(trials_df, score_col="combined_score")
        oos_eval_params = best_params.copy()
        oos_eval_params.update(metrics_info)
        oos_score = strategy_adapter.run_backtest(
            oos_df,
            oos_eval_params,
            timeframe=timeframe,
            return_portfolio=False,
        )
        oos_score_f = _to_scalar_score(oos_score)
        if np.isfinite(oos_score_f):
            model.update_single(best_params, oos_score_f, weight=max(0.0, oos_weight))

        window_dates = {
            "window": cycle_id,
            "start_date": train_df.index[0],
            "end_date": oos_df.index[-1],
            "in_sample_start": train_df.index[0],
            "in_sample_end": train_df.index[-1],
            "out_sample_start": oos_df.index[0],
            "out_sample_end": oos_df.index[-1],
        }
        cycle_info = {
            "cycle": cycle_id,
            "train_rows": int(len(train_df)),
            "out_rows": int(len(oos_df)),
            "trials_tested": int(len(trials_df)),
            "baseline_combinations": int(baseline_combinations),
            "active_combinations": int(active_combinations),
        }
        window_result = {
            "window_info": window_dates,
            "optimization_results": trials_df.head(5).to_dict("records"),
            # Keep all cycle trial rows for downstream statistical analyses.
            "optimization_trials": trials_df.to_dict("records"),
            "optimization_trials_count": int(len(trials_df)),
            "best_params": best_params,
            "cycle_info": cycle_info,
        }
        results["window_results"].append(window_result)
        results["cycle_results"].append(window_result)

        adaptive_entry = {
            "window": cycle_id,
            **guidance_info,
            "baseline_combinations": int(baseline_combinations),
            "active_combinations": int(active_combinations),
            "trials_tested": int(len(trials_df)),
        }
        results["adaptive_guidance"].append(adaptive_entry)

        last_best_params = best_params.copy()

        cycle_elapsed = time.time() - cycle_start
        cycle_times.append(cycle_elapsed)
        trial_counts.append(int(len(trials_df)))

        log(
            f"Cycle {cycle_id}: best train score={best_row['combined_score']:.4f}, "
            f"active combos={active_combinations}, trials={len(trials_df)}"
        )
        report({
            "type": "stats",
            "window": cycle_id,
            "cycle": cycle_id,
            "progress": float(oos_end / max(1, total_rows)),
            "evaluations": int(len(trials_df)),
            "message": f"Cycle {cycle_id} | rows {oos_end}/{total_rows}",
            "window_metrics": {
                "in_sample": in_sample_metrics,
                "out_sample": out_sample_metrics,
            },
        })

        pointer = oos_end
        cycle_id += 1

    if not results["window_results"]:
        raise ValueError("Adaptive optimization produced no cycle results.")

    total_time = time.time() - start_time
    results["settings"]["n_windows"] = len(results["window_results"])
    results["timing"] = {
        "total_time": total_time,
        "window_times": cycle_times,
        "optimization_times": cycle_times,
        "avg_window_time": float(np.mean(cycle_times)) if cycle_times else 0.0,
        "avg_optimization_time": float(np.mean(cycle_times)) if cycle_times else 0.0,
        "backend": settings.parallel_backend,
        "use_numba": settings.use_numba,
        "param_combinations": baseline_combinations,
        "total_trials": int(sum(trial_counts)),
    }
    results["adaptive_summary"] = {
        "cycles_completed": int(len(results["window_results"])),
        "total_trials": int(sum(trial_counts)),
        "parameter_weights": model.parameter_weights(),
        "top_values_by_parameter": model.summarize_top_values(top_n=5),
    }

    if results["in_sample_performance"]:
        is_df = pd.DataFrame(results["in_sample_performance"])
        log("\n=== Adaptive In-Sample Summary ===")
        log(f"Average Return: {is_df['return'].mean():.2f}%")
        log(f"Average Sharpe Ratio: {is_df['sharpe'].mean():.2f}")
        log(f"Average Max Drawdown: {is_df['max_drawdown'].mean():.2f}%")
        log(f"Average Win Rate: {is_df['win_rate'].mean():.2f}%")

    if results["out_of_sample_performance"]:
        oos_df = pd.DataFrame(results["out_of_sample_performance"])
        log("\n=== Adaptive Out-of-Sample Summary ===")
        log(f"Average Return: {oos_df['return'].mean():.2f}%")
        log(f"Average Sharpe Ratio: {oos_df['sharpe'].mean():.2f}")
        log(f"Average Max Drawdown: {oos_df['max_drawdown'].mean():.2f}%")
        log(f"Average Win Rate: {oos_df['win_rate'].mean():.2f}%")

    log("\n=== Adaptive Timing Summary ===")
    log(f"Total cycles: {len(results['window_results'])}")
    log(f"Total trials: {sum(trial_counts)}")
    log(f"Total processing time: {timedelta(seconds=int(total_time))}")
    if cycle_times:
        log(f"Average cycle time: {timedelta(seconds=int(np.mean(cycle_times)))}")

    return results
