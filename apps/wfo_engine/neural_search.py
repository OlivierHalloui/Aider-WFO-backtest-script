"""Neural-network-guided search space reduction for WFO parameter optimization."""

import numpy as np


def _safe_float(value, default=0.0):
    try:
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if value else 0.0
        if isinstance(value, (np.integer, int, np.floating, float)):
            value = float(value)
            if np.isnan(value) or np.isinf(value):
                return float(default)
            return value
        return float(default)
    except Exception:
        return float(default)


class NeuralSearchGuide:
    """Lightweight numpy MLP used to guide WFO search space from past windows."""

    def __init__(self, base_param_grid, settings):
        self.param_keys = list(base_param_grid.keys())
        self.category_maps = {key: {} for key in self.param_keys}
        self.records = []
        self.targets = []
        self.max_records = int(getattr(settings, 'nn_max_records', 200000))
        self.min_samples = int(getattr(settings, 'nn_min_samples', 500))
        self.candidate_pool_size = int(getattr(settings, 'nn_candidate_pool_size', 3000))
        self.top_k = int(getattr(settings, 'nn_top_k', 250))
        self.exploration_ratio = float(getattr(settings, 'nn_exploration_ratio', 0.15))
        self.hidden_size = int(getattr(settings, 'nn_hidden_size', 32))
        self.epochs = int(getattr(settings, 'nn_epochs', 60))
        self.learning_rate = float(getattr(settings, 'nn_learning_rate', 0.01))
        self.l2 = float(getattr(settings, 'nn_l2', 1e-4))
        self.random_state = int(getattr(settings, 'random_state', 42))
        self.rng = np.random.default_rng(self.random_state)
        self.last_best_params = None

        self.x_mean = None
        self.x_std = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.trained = False

    def _encode_value(self, key, value):
        if isinstance(value, (bool, np.bool_)):
            return 1.0 if value else 0.0
        if isinstance(value, (np.integer, int, np.floating, float)):
            return _safe_float(value, default=0.0)
        map_key = str(value)
        cat_map = self.category_maps.setdefault(key, {})
        if map_key not in cat_map:
            cat_map[map_key] = len(cat_map)
        return float(cat_map[map_key])

    def _encode_rows(self, rows):
        encoded = []
        for row in rows:
            vector = [self._encode_value(key, row.get(key)) for key in self.param_keys]
            encoded.append(vector)
        return np.asarray(encoded, dtype=float)

    def _init_model(self, in_dim):
        scale1 = np.sqrt(2.0 / max(1, in_dim))
        scale2 = np.sqrt(2.0 / max(1, self.hidden_size))
        self.W1 = self.rng.normal(0.0, scale1, size=(in_dim, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size), dtype=float)
        self.W2 = self.rng.normal(0.0, scale2, size=(self.hidden_size, 1))
        self.b2 = np.zeros((1, 1), dtype=float)

    def _fit(self):
        if len(self.targets) < self.min_samples:
            self.trained = False
            return

        X = self._encode_rows(self.records)
        y = np.asarray(self.targets, dtype=float).reshape(-1, 1)
        if X.ndim != 2 or X.shape[0] < self.min_samples:
            self.trained = False
            return

        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True)
        self.x_std = np.where(self.x_std < 1e-9, 1.0, self.x_std)
        Xn = (X - self.x_mean) / self.x_std

        self.y_mean = float(np.nanmean(y))
        y_std = float(np.nanstd(y))
        self.y_std = 1.0 if y_std < 1e-9 else y_std
        yn = (y - self.y_mean) / self.y_std

        in_dim = Xn.shape[1]
        if self.W1 is None or self.W1.shape[0] != in_dim:
            self._init_model(in_dim)

        n = Xn.shape[0]
        lr = max(1e-5, self.learning_rate)
        l2 = max(0.0, self.l2)

        for _ in range(max(1, self.epochs)):
            h = np.tanh(Xn @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            err = pred - yn

            d_pred = (2.0 / n) * err
            grad_W2 = h.T @ d_pred + l2 * self.W2
            grad_b2 = np.sum(d_pred, axis=0, keepdims=True)

            d_h = d_pred @ self.W2.T
            d_z = d_h * (1.0 - h ** 2)
            grad_W1 = Xn.T @ d_z + l2 * self.W1
            grad_b1 = np.sum(d_z, axis=0, keepdims=True)

            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1

        self.trained = True

    def predict_scores(self, rows):
        if not self.trained or self.W1 is None:
            return None
        X = self._encode_rows(rows)
        Xn = (X - self.x_mean) / self.x_std
        h = np.tanh(Xn @ self.W1 + self.b1)
        pred = h @ self.W2 + self.b2
        return pred.reshape(-1) * self.y_std + self.y_mean

    def update(self, optimization_results, param_grid):
        if optimization_results is None or optimization_results.empty:
            return
        if 'combined_score' not in optimization_results.columns:
            return

        valid_rows = optimization_results.replace([np.inf, -np.inf], np.nan).dropna(subset=['combined_score'])
        if valid_rows.empty:
            return

        param_cols = [key for key in self.param_keys if key in valid_rows.columns]
        if not param_cols:
            return

        for record in valid_rows[param_cols + ['combined_score']].to_dict('records'):
            score = _safe_float(record.get('combined_score'), default=np.nan)
            if np.isnan(score) or np.isinf(score):
                continue
            self.records.append({key: record[key] for key in param_cols})
            self.targets.append(score)

        if len(self.targets) > self.max_records:
            self.records = self.records[-self.max_records:]
            self.targets = self.targets[-self.max_records:]

        self._fit()

    def get_parameter_weights(self):
        if not self.trained or self.W1 is None or self.W2 is None:
            return {}
        # Input saliency proxy from absolute path weights through first hidden layer.
        raw = np.abs(self.W1) @ np.abs(self.W2).reshape(-1, 1)
        raw = raw.reshape(-1)
        total = float(np.sum(raw))
        if total <= 0:
            return {key: 0.0 for key in self.param_keys}
        return {key: float(raw[idx] / total) for idx, key in enumerate(self.param_keys)}

    def build_guided_grid(self, base_param_grid):
        """Create a reduced parameter grid guided by NN scores plus exploration safeguards."""
        baseline_combos = int(np.prod([len(v) for v in base_param_grid.values()]))
        if not self.trained:
            return base_param_grid, {
                'enabled': False,
                'trained': False,
                'reason': 'insufficient_samples',
                'baseline_combinations': baseline_combos,
                'guided_combinations': baseline_combos
            }

        sample_count = max(self.top_k * 2, self.candidate_pool_size)
        sample_count = max(500, sample_count)
        sampled_rows = []
        for _ in range(sample_count):
            candidate = {}
            for key, values in base_param_grid.items():
                if not values:
                    continue
                idx = int(self.rng.integers(0, len(values)))
                candidate[key] = values[idx]
            sampled_rows.append(candidate)

        predicted = self.predict_scores(sampled_rows)
        if predicted is None or len(predicted) == 0:
            return base_param_grid, {
                'enabled': False,
                'trained': True,
                'reason': 'prediction_failed',
                'baseline_combinations': baseline_combos,
                'guided_combinations': baseline_combos
            }

        top_k = max(1, min(self.top_k, len(sampled_rows)))
        top_idx = np.argpartition(predicted, -top_k)[-top_k:]
        top_rows = [sampled_rows[i] for i in top_idx]

        guided_grid = {}
        for key, base_values in base_param_grid.items():
            base_values = list(base_values)
            if len(base_values) <= 1:
                guided_grid[key] = base_values
                continue

            selected_values = []
            seen = set()
            for row in top_rows:
                value = row.get(key)
                if value in seen:
                    continue
                seen.add(value)
                selected_values.append(value)

            if self.last_best_params and key in self.last_best_params and self.last_best_params[key] in base_values:
                best_value = self.last_best_params[key]
                if best_value not in seen:
                    seen.add(best_value)
                    selected_values.append(best_value)

            min_keep = max(2, int(np.ceil(len(base_values) * 0.2)))
            while len(selected_values) < min_keep:
                for value in base_values:
                    if value not in seen:
                        seen.add(value)
                        selected_values.append(value)
                        break
                else:
                    break

            exploration_count = max(1, int(np.ceil(len(base_values) * self.exploration_ratio)))
            remaining = [value for value in base_values if value not in seen]
            if remaining:
                self.rng.shuffle(remaining)
                for value in remaining[:exploration_count]:
                    seen.add(value)
                    selected_values.append(value)

            # Preserve original order for reproducibility.
            guided_values = [value for value in base_values if value in seen]
            guided_grid[key] = guided_values if guided_values else base_values

        guided_combos = int(np.prod([len(v) for v in guided_grid.values()]))
        weights = self.get_parameter_weights()
        return guided_grid, {
            'enabled': True,
            'trained': True,
            'baseline_combinations': baseline_combos,
            'guided_combinations': guided_combos,
            'reduction_ratio': 1.0 - (guided_combos / baseline_combos) if baseline_combos > 0 else 0.0,
            'parameter_weights': weights
        }


def _match_prev_value_to_candidates(prev_value, candidates):
    if candidates is None:
        return None
    for candidate in candidates:
        if candidate == prev_value:
            return candidate
    # Fallbacks for bool/num casting mismatches
    if isinstance(prev_value, (bool, np.bool_)):
        prev_bool = bool(prev_value)
        for candidate in candidates:
            if isinstance(candidate, (bool, np.bool_)) and bool(candidate) == prev_bool:
                return candidate
        return None

    if isinstance(prev_value, (int, float, np.integer, np.floating)):
        prev_num = float(prev_value)
        for candidate in candidates:
            if isinstance(candidate, (int, float, np.integer, np.floating)) and np.isclose(float(candidate), prev_num):
                return candidate
    return None


def _build_prev_best_grid(base_param_grid, prev_best_params):
    """Build a grid centered on previous-window best values when they still exist in domains."""
    guided_grid = {}
    fixed_count = 0
    for key, values in base_param_grid.items():
        base_values = list(values)
        prev_value = prev_best_params.get(key) if isinstance(prev_best_params, dict) else None
        matched = _match_prev_value_to_candidates(prev_value, base_values)
        if matched is not None:
            guided_grid[key] = [matched]
            fixed_count += 1
        else:
            guided_grid[key] = base_values

    baseline = int(np.prod([len(v) for v in base_param_grid.values()])) if base_param_grid else 0
    guided = int(np.prod([len(v) for v in guided_grid.values()])) if guided_grid else 0
    return guided_grid, {
        'enabled': True,
        'trained': True,
        'fixed_params': fixed_count,
        'baseline_combinations': baseline,
        'guided_combinations': guided,
        'reduction_ratio': 1.0 - (guided / baseline) if baseline > 0 else 0.0
    }
