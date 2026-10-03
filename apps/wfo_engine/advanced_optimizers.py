"""
Advanced black-box optimizers for the WFO Engine: TuRBO and BADS.

Both target a costly, noisy, non-convex objective (a full backtest score) over a
DISCRETE parameter grid (each parameter exposes a finite list of candidate
values: int / float / bool / str). They use a Gaussian Process surrogate
(scikit-learn) to spend the evaluation budget wisely, for low-to-moderate
dimension.

- TuRBO (Trust-Region Bayesian Optimization), Eriksson et al., NeurIPS 2019:
  maintains one or more local trust regions (adaptive boxes) and optimises the GP
  acquisition (Expected Improvement) inside them.  Regions expand on success and
  shrink on failure; a region that collapses triggers a restart near the best
  point seen so far.  Suited to ~5-20 active numeric dimensions.

- BADS (Bayesian Adaptive Direct Search), Acerbi & Ma, NeurIPS 2017: couples a
  Mesh Adaptive Direct Search (MADS) poll step with a GP that ranks candidates,
  so each poll evaluates the most promising neighbours first.  Excellent for
  rugged / step-like objectives (as a discrete backtest score).

Both are used as ``optimization_method = 'turbo' | 'bads'`` in ``wfo.py``.
They maximise ``evaluate_params(params)`` (higher = better) and return the list
of evaluated configurations so the caller can build a results DataFrame.

This module has no heavyweight dependency beyond scikit-learn (already pulled in
by scikit-optimize), keeping the pinned numpy/numba stack intact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# The GP surrogate is fit many times in a loop; scikit-learn's length-scale
# convergence warnings are benign here and would flood the WFO run log.
try:
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:  # pragma: no cover - sklearn always present in this stack
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter encoding: discrete grid <-> continuous [0, 1]^d box
# ---------------------------------------------------------------------------


@dataclass
class GridEncoder:
    """Encode a discrete parameter grid into a continuous unit hypercube.

    Each dimension is one parameter; its candidate values are indexed from 0 to
    ``n - 1`` and normalised to [0, 1].  Decoding snaps a continuous coordinate
    to the nearest candidate, which keeps optimiser proposals legal regardless
    of the underlying value type (int / float / bool / str).
    """

    names: List[str]
    values: Dict[str, Sequence]
    n_values: Dict[str, int]

    @classmethod
    def from_grid(cls, tunable_grid: Dict[str, Sequence]) -> "GridEncoder":
        names = list(tunable_grid.keys())
        values = {k: list(v) for k, v in tunable_grid.items()}
        n_values = {k: len(v) for k, v in values.items()}
        if any(n < 1 for n in n_values.values()):
            raise ValueError("Every tunable parameter needs at least one candidate value.")
        return cls(names=names, values=values, n_values=n_values)

    @property
    def dim(self) -> int:
        return len(self.names)

    def encode(self, params: Dict) -> np.ndarray:
        """Map a discrete configuration to its unit-cube coordinate."""
        x = np.empty(self.dim, dtype=float)
        for i, name in enumerate(self.names):
            vals = self.values[name]
            n = self.n_values[name]
            try:
                idx = vals.index(params[name])
            except ValueError:
                # Fall back to the closest numeric candidate if the exact value
                # is absent (e.g. a float produced by the optimiser).
                idx = min(
                    range(n),
                    key=lambda j: abs(float(vals[j]) - float(params[name])),
                )
            x[i] = 0.0 if n == 1 else idx / (n - 1)
        return x

    def decode(self, x: np.ndarray) -> Dict:
        """Snap a unit-cube coordinate to the nearest candidate configuration."""
        params = {}
        for i, name in enumerate(self.names):
            vals = self.values[name]
            n = self.n_values[name]
            idx = 0 if n == 1 else int(round(float(np.clip(x[i], 0.0, 1.0)) * (n - 1)))
            idx = max(0, min(n - 1, idx))
            params[name] = vals[idx]
        return params

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# GP surrogate
# ---------------------------------------------------------------------------


def _fit_gp(X: np.ndarray, y: np.ndarray, seed: int = 0):
    """Fit a Matern-5/2 GP; fall back to a constant model on numerical failure."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, alpha=1e-6,
            random_state=seed,
        )
        gp.fit(X, y)
        return gp
    except Exception as exc:  # pragma: no cover - numerical safety net
        logger.warning("GP fit failed (%s); using constant surrogate.", exc)
        return None


def _gp_predict(gp, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if gp is None:
        # Constant surrogate: predict the data mean with unit variance.
        mu = np.zeros(len(X))
        std = np.ones(len(X))
        return mu, std
    mu, std = gp.predict(X, return_std=True)
    std = np.maximum(std, 1e-9)
    return mu, std


def _expected_improvement(mu: np.ndarray, std: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """EI for maximisation (higher score is better)."""
    imp = mu - best - xi
    z = imp / std
    from math import erf, sqrt  # local import keeps module import light

    # Standard normal CDF / PDF without requiring scipy.
    cdf = 0.5 * (1.0 + np.vectorize(lambda t: erf(t / sqrt(2.0)))(z))
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2.0 * math.pi)
    ei = imp * cdf + std * pdf
    ei = np.where(std > 1e-9, ei, 0.0)
    return ei


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------


def _snap_key(encoder: "GridEncoder", x: np.ndarray) -> Tuple:
    """Canonical hashable key of the discrete config ``x`` snaps to (for dedup)."""
    p = encoder.decode(x)
    return tuple(p[name] for name in encoder.names)


@dataclass
class _EvalLog:
    """Records every evaluated configuration for the results DataFrame."""

    rows: List[Dict] = field(default_factory=list)
    X: List[np.ndarray] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    seen: set = field(default_factory=set)

    def add(self, params: Dict, x: np.ndarray, score: float) -> None:
        row = dict(params)
        row["combined_score"] = float(score)
        self.rows.append(row)
        self.X.append(np.asarray(x, dtype=float))
        self.y.append(float(score))


def _evaluate_once(
    x: np.ndarray,
    encoder: GridEncoder,
    evaluate_params: Callable[[Dict], float],
    fixed_params: Dict,
    metrics_info: Dict,
    log: _EvalLog,
) -> float:
    params = encoder.decode(x)
    full = dict(params)
    full.update(fixed_params)
    full.update(metrics_info)
    score = evaluate_params(full)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = float("nan")
    if not math.isfinite(score):
        # Map non-finite scores (NaN from a no-trade backtest, +/-inf) to a finite
        # "very bad" value: a -inf/NaN in y makes EVERY GP fit fail and silently
        # degrades the rest of the run to near-random search.
        finite = [v for v in log.y if math.isfinite(v)]
        score = (min(finite) if finite else 0.0) - 1.0
    log.seen.add(_snap_key(encoder, encoder.clip(x)))
    log.add({**params, **fixed_params, **metrics_info}, encoder.clip(x), score)
    return score


def _uniform_sample(encoder: GridEncoder, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random unit-cube sample biased toward grid-interior indices."""
    return rng.random((n, encoder.dim))


def _latin_sample(encoder: GridEncoder, n: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Latin-hypercube sample in the unit cube for a space-filling start."""
    d = encoder.dim
    cut = np.linspace(0.0, 1.0, n + 1)
    u = rng.random((n, d))
    a = cut[:n]
    b = cut[1 : n + 1]
    pts = u * (b - a)[:, None] + a[:, None]
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


# ---------------------------------------------------------------------------
# TuRBO — Trust-Region Bayesian Optimization
# ---------------------------------------------------------------------------


def turbo_optimize(
    evaluate_params: Callable[[Dict], float],
    tunable_grid: Dict[str, Sequence],
    fixed_params: Dict,
    metrics_info: Dict,
    n_evals: int,
    n_init: int = 12,
    n_candidates: int = 512,
    n_trust_regions: int = 1,
    init_length: float = 0.8,
    min_length: float = 0.02,
    max_length: float = 1.6,
    grow_factor: float = 1.5,
    shrink_factor: float = 0.5,
    success_tol: int = 3,
    failure_tol: int = 5,
    rng: np.random.Generator | None = None,
) -> List[Dict]:
    """TuRBO over a discrete grid; returns evaluated configs (best first by caller)."""
    rng = rng or np.random.default_rng(42)
    encoder = GridEncoder.from_grid(tunable_grid)
    d = encoder.dim
    log = _EvalLog()

    n_init = max(2, min(n_init, n_evals))
    start_pts = _latin_sample(encoder, n_init, rng)
    for x in start_pts:
        if _snap_key(encoder, x) in log.seen:
            continue  # dedup initial design (the discrete snap can collapse points)
        _evaluate_once(x, encoder, evaluate_params, fixed_params, metrics_info, log)
        if len(log.y) >= n_evals:
            return log.rows

    # Trust-region state (per region).
    centres = [encoder.clip(log.X[int(np.argmax(log.y))]).copy() for _ in range(max(1, n_trust_regions))]
    lengths = [init_length] * len(centres)
    success = [0] * len(centres)
    failure = [0] * len(centres)

    n_iter_max = 20 * n_evals  # safety against a fully-covered neighbourhood
    iteration = 0
    while len(log.y) < n_evals and iteration < n_iter_max:
        iteration += 1
        X = np.asarray(log.X)
        y = np.asarray(log.y)
        gp = _fit_gp(X, y, seed=int(rng.integers(0, 2**31 - 1)))
        best = float(np.max(y))

        for r in range(len(centres)):
            if len(log.y) >= n_evals:
                break
            half = lengths[r] / 2.0
            lo = np.clip(centres[r] - half, 0.0, 1.0)
            hi = np.clip(centres[r] + half, 0.0, 1.0)

            # Sample candidates in the region and pick the best UNSEEN by EI
            # (dedup: the discrete snap can collapse distinct points to one config).
            cand = rng.uniform(lo, hi, size=(n_candidates, d))
            cand = np.clip(cand, 0.0, 1.0)
            mu, std = _gp_predict(gp, cand)
            ei = _expected_improvement(mu, std, best)
            order = np.argsort(-ei)
            x_next = None
            for idx in order:
                if _snap_key(encoder, cand[idx]) not in log.seen:
                    x_next = cand[idx]
                    break
            if x_next is None:
                # Whole neighbourhood already evaluated: shrink and retry without
                # spending the budget.
                lengths[r] *= shrink_factor
                success[r] = failure[r] = 0
                if lengths[r] < min_length:
                    centres[r] = encoder.clip(log.X[int(np.argmax(log.y))]).copy()
                    lengths[r] = init_length
                continue

            score = _evaluate_once(x_next, encoder, evaluate_params, fixed_params, metrics_info, log)

            if score > best + 1e-12:
                best = score
                centres[r] = encoder.clip(x_next).copy()
                success[r] += 1
                failure[r] = 0
                if success[r] >= success_tol:
                    lengths[r] = min(lengths[r] * grow_factor, max_length)
                    success[r] = 0
            else:
                failure[r] += 1
                success[r] = 0
                if failure[r] >= failure_tol:
                    lengths[r] *= shrink_factor
                    failure[r] = 0

            # Restart a collapsed region around the global incumbent.
            if lengths[r] < min_length:
                centres[r] = encoder.clip(log.X[int(np.argmax(log.y))]).copy()
                lengths[r] = init_length
                success[r] = failure[r] = 0

    return log.rows


# ---------------------------------------------------------------------------
# BADS — Bayesian Adaptive Direct Search (mesh adaptive + GP-ranked poll)
# ---------------------------------------------------------------------------


def bads_optimize(
    evaluate_params: Callable[[Dict], float],
    tunable_grid: Dict[str, Sequence],
    fixed_params: Dict,
    metrics_info: Dict,
    n_evals: int,
    n_init: int = 8,
    init_mesh: float = 0.3,
    min_mesh: float = 0.005,
    max_mesh: float = 0.6,
    n_poll: int = 2,
    rng: np.random.Generator | None = None,
) -> List[Dict]:
    """BADS (MADS poll + GP ranking) over a discrete grid; returns evaluated configs."""
    rng = rng or np.random.default_rng(42)
    encoder = GridEncoder.from_grid(tunable_grid)
    d = encoder.dim
    log = _EvalLog()

    n_init = max(2, min(n_init, n_evals))
    for x in _latin_sample(encoder, n_init, rng):
        if _snap_key(encoder, x) in log.seen:
            continue  # dedup initial design (the discrete snap can collapse points)
        _evaluate_once(x, encoder, evaluate_params, fixed_params, metrics_info, log)
        if len(log.y) >= n_evals:
            return log.rows

    # Start from the best init point.
    x_c = encoder.clip(log.X[int(np.argmax(log.y))]).copy()
    y_c = float(np.max(log.y))
    mesh = init_mesh

    n_iter_max = 20 * n_evals
    iteration = 0
    while len(log.y) < n_evals and iteration < n_iter_max:
        iteration += 1
        X = np.asarray(log.X)
        y = np.asarray(log.y)
        gp = _fit_gp(X, y, seed=int(rng.integers(0, 2**31 - 1)))

        # Build a poll stencil: axis-aligned neighbours at the current mesh size
        # (plus a coarse diagonal), then rank them with the GP.
        offsets = []
        for i in range(d):
            for s in (-1.0, 1.0):
                off = np.zeros(d)
                off[i] = s * mesh
                offsets.append(off)
        for _ in range(n_poll):
            offsets.append(rng.normal(0.0, mesh, size=d))
        cand = np.clip(np.array([x_c + off for off in offsets]), 0.0, 1.0)

        mu, std = _gp_predict(gp, cand)
        order = np.argsort(-(mu + 0.5 * std))  # optimistic ranking

        improved = False
        for idx in order:
            if len(log.y) >= n_evals:
                break
            if _snap_key(encoder, cand[idx]) in log.seen:
                continue  # already evaluated (dedup: keep the budget for new configs)
            score = _evaluate_once(cand[idx], encoder, evaluate_params, fixed_params, metrics_info, log)
            if score > y_c + 1e-12:
                x_c = cand[idx].copy()
                y_c = score
                improved = True
                break  # successful poll: move and re-base the stencil

        if improved:
            mesh = min(mesh * 1.5, max_mesh)  # opportunistic expansion
        else:
            mesh = mesh / 2.0 if mesh > min_mesh else init_mesh
            if mesh <= min_mesh:
                # Restart near the global incumbent to escape a stale basin.
                x_c = encoder.clip(log.X[int(np.argmax(log.y))]).copy()
                y_c = float(np.max(log.y))
                mesh = init_mesh

    return log.rows
