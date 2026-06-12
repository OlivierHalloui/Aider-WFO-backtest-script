"""Serialization and scalar-conversion helpers shared across the app."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd


def json_safe(obj: Any) -> Any:
    """Convert common numpy/pandas scalar objects into JSON-safe values."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, (pd.Timedelta, datetime.timedelta)):
        return str(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


def sanitize_for_json(value: Any) -> Any:
    """Recursively normalize values so they can be serialized to valid JSON (RFC 7159).

    float('nan') and float('inf') are converted to None (JSON null) because the
    JSON spec does not permit NaN/Infinity literals. Callers should pass
    allow_nan=False to json.dumps to catch any residual cases.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, np.floating):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(
        value,
        (
            np.integer,
            np.ndarray,
            pd.Timestamp,
            datetime.datetime,
            datetime.date,
            pd.Timedelta,
            datetime.timedelta,
            np.bool_,
        ),
    ):
        return json_safe(value)
    return value


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_json(value: Any) -> str | None:
    """Hash sanitized JSON representation for traceability."""
    try:
        normalized = sanitize_for_json(value)
        serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except Exception:
        return None


def to_jsonable(value: Any) -> Any:
    """Convert a scalar-like value to a JSON-friendly primitive when possible."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta, datetime.timedelta)):
        return str(value)
    return value


def safe_float_scalar(value: Any) -> float:
    """Return a finite float scalar, using mean reduction for array-like values."""
    try:
        if np.isscalar(value):
            out = float(value)
            return out if np.isfinite(out) else np.nan
        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return np.nan
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.mean(arr))
    except Exception:
        return np.nan
