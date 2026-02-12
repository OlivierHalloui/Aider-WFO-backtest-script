"""Parity report helpers for Pine V3 (P1.4)."""

from __future__ import annotations

from typing import Any
import datetime
import math


DEFAULT_PARITY_THRESHOLDS = {
    "trade_count_rel_pct": 2.0,
    "entry_count_rel_pct": 2.0,
    "exit_count_rel_pct": 2.0,
    "total_return_abs_pct": 3.0,
    "max_drawdown_abs_pct": 3.0,
}
DEFAULT_PARITY_DETAIL_THRESHOLDS = {
    "entry_event_count_rel_pct": 5.0,
    "exit_event_count_rel_pct": 5.0,
    "entry_event_match_min_ratio": 0.85,
    "exit_event_match_min_ratio": 0.85,
    "trade_match_min_ratio": 0.80,
    "event_time_tolerance_sec": 5.0,
    "trade_time_tolerance_sec": 5.0,
}

PARITY_REFERENCE_SCHEMA_VERSION = "pine_parity_reference.v1"
PARITY_REFERENCE_VALIDATION_SCHEMA_VERSION = "pine_parity_reference_validation.v1"
PARITY_REQUIRED_METRICS = ("trade_count", "total_return_pct", "max_drawdown_pct")


_METRIC_ALIASES = {
    "trade_count": {"trade_count", "trades", "n_trades", "number_of_trades"},
    "entry_count": {"entry_count", "entries", "n_entries", "entry_trades"},
    "exit_count": {"exit_count", "exits", "n_exits", "exit_trades"},
    "total_return_pct": {"total_return_pct", "total_return", "return_pct", "return_percent"},
    "max_drawdown_pct": {"max_drawdown_pct", "max_drawdown", "drawdown_pct", "mdd_pct"},
}
_EVENT_ALIASES = {
    "entries": {"entries", "entry_events", "entry_times", "entry_timestamps"},
    "exits": {"exits", "exit_events", "exit_times", "exit_timestamps"},
}


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_metric_name(name: str) -> str:
    token = str(name or "").strip().lower()
    for canonical, aliases in _METRIC_ALIASES.items():
        if token in aliases:
            return canonical
    return token


def normalize_metrics(raw: dict[str, Any] | None) -> dict[str, float]:
    """Normalize metric names and keep numeric scalar values only."""
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        canonical = _normalize_metric_name(str(key))
        try:
            out[canonical] = float(value)
        except Exception:
            continue
    return out


def _safe_rel_diff_pct(reference: float, current: float) -> float:
    denom = max(abs(float(reference)), 1.0)
    return abs(float(current) - float(reference)) / denom * 100.0


def _normalize_event_name(name: str) -> str:
    token = str(name or "").strip().lower()
    for canonical, aliases in _EVENT_ALIASES.items():
        if token in aliases:
            return canonical
    return token


def _to_epoch_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            ts = float(value)
            if math.isfinite(ts):
                return ts
        except Exception:
            return None
        return None
    token = str(value).strip()
    if not token:
        return None
    try:
        as_num = float(token)
        if math.isfinite(as_num):
            return as_num
    except Exception:
        pass
    try:
        if token.endswith("Z"):
            token = token[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(token)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _epoch_to_iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(float(ts), tz=datetime.timezone.utc).isoformat()


def normalize_event_timestamps(raw: Any) -> list[float]:
    """Normalize event timestamps to sorted epoch-second list."""
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        candidate = item
        if isinstance(item, dict):
            for key in ("time", "timestamp", "datetime", "ts", "entry_time", "exit_time"):
                if key in item:
                    candidate = item.get(key)
                    break
        ts = _to_epoch_seconds(candidate)
        if ts is None:
            continue
        out.append(float(ts))
    out.sort()
    return out


def normalize_trade_pairs(raw: Any) -> list[dict[str, float]]:
    """Normalize trade rows to sorted epoch-second entry/exit pairs."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry_val = None
        exit_val = None
        for key in ("entry_time", "entry_timestamp", "entry_ts", "entry"):
            if key in item:
                entry_val = item.get(key)
                break
        for key in ("exit_time", "exit_timestamp", "exit_ts", "exit"):
            if key in item:
                exit_val = item.get(key)
                break
        entry_ts = _to_epoch_seconds(entry_val)
        exit_ts = _to_epoch_seconds(exit_val)
        if entry_ts is None or exit_ts is None:
            continue
        if float(exit_ts) < float(entry_ts):
            continue
        row: dict[str, float] = {
            "entry_ts": float(entry_ts),
            "exit_ts": float(exit_ts),
        }
        try:
            if "pnl" in item and item.get("pnl") is not None:
                row["pnl"] = float(item.get("pnl"))
        except Exception:
            pass
        out.append(row)
    out.sort(key=lambda x: (x.get("entry_ts", 0.0), x.get("exit_ts", 0.0)))
    return out


def normalize_events(raw: dict[str, Any] | None) -> dict[str, list[float]]:
    """Normalize event dictionary with `entries` and `exits` keys."""
    if not isinstance(raw, dict):
        return {"entries": [], "exits": []}
    normalized: dict[str, list[float]] = {"entries": [], "exits": []}
    for key, value in raw.items():
        canonical = _normalize_event_name(str(key))
        if canonical in ("entries", "exits"):
            normalized[canonical] = normalize_event_timestamps(value)
    return normalized


def _count_time_matches(reference: list[float], current: list[float], tolerance_sec: float) -> int:
    """Greedy matching count for timestamp lists with tolerance."""
    ref = sorted([float(x) for x in reference])
    cur = sorted([float(x) for x in current])
    tol = abs(float(tolerance_sec))
    i = 0
    j = 0
    matched = 0
    while i < len(ref) and j < len(cur):
        rv = ref[i]
        cv = cur[j]
        if cv < rv - tol:
            j += 1
        elif cv > rv + tol:
            i += 1
        else:
            matched += 1
            i += 1
            j += 1
    return matched


def _count_trade_matches(
    reference: list[dict[str, float]],
    current: list[dict[str, float]],
    tolerance_sec: float,
) -> int:
    """Greedy matching count for trade entry/exit timestamp pairs."""
    ref = sorted(reference, key=lambda x: (x.get("entry_ts", 0.0), x.get("exit_ts", 0.0)))
    cur = sorted(current, key=lambda x: (x.get("entry_ts", 0.0), x.get("exit_ts", 0.0)))
    tol = abs(float(tolerance_sec))
    i = 0
    j = 0
    matched = 0
    while i < len(ref) and j < len(cur):
        re = float(ref[i].get("entry_ts", 0.0))
        ce = float(cur[j].get("entry_ts", 0.0))
        if ce < re - tol:
            j += 1
            continue
        if ce > re + tol:
            i += 1
            continue
        rx = float(ref[i].get("exit_ts", 0.0))
        cx = float(cur[j].get("exit_ts", 0.0))
        if abs(cx - rx) <= tol:
            matched += 1
            i += 1
            j += 1
        elif cx < rx - tol:
            j += 1
        else:
            i += 1
    return matched


def build_parity_reference_payload(
    reference_metrics: dict[str, Any] | None,
    reference_events: dict[str, Any] | None = None,
    reference_trades: list[dict[str, Any]] | None = None,
    source: dict[str, Any] | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build canonical parity reference payload in `pine_parity_reference.v1` format."""
    metrics = normalize_metrics(reference_metrics)
    events_raw = normalize_events(reference_events)
    trades_raw = normalize_trade_pairs(reference_trades)
    src = source if isinstance(source, dict) else {}
    return {
        "schema_version": PARITY_REFERENCE_SCHEMA_VERSION,
        "generated_at_utc": str(generated_at_utc or _utc_now_iso()),
        "source": {
            "provider": str(src.get("provider") or ""),
            "strategy_id": str(src.get("strategy_id") or ""),
            "symbol": str(src.get("symbol") or ""),
            "timeframe": str(src.get("timeframe") or ""),
            "start_date": str(src.get("start_date") or ""),
            "end_date": str(src.get("end_date") or ""),
            "notes": str(src.get("notes") or ""),
        },
        "reference_metrics": metrics,
        "reference_events": {
            "entries": [_epoch_to_iso(ts) for ts in events_raw.get("entries", [])],
            "exits": [_epoch_to_iso(ts) for ts in events_raw.get("exits", [])],
        },
        "reference_trades": [
            {
                "entry_time": _epoch_to_iso(float(row.get("entry_ts", 0.0))),
                "exit_time": _epoch_to_iso(float(row.get("exit_ts", 0.0))),
                **({"pnl": float(row.get("pnl"))} if "pnl" in row else {}),
            }
            for row in trades_raw
        ],
    }


def validate_parity_reference_payload(
    payload: dict[str, Any] | None,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """
    Validate and normalize parity reference payload.

    Canonical format:
    - schema_version: pine_parity_reference.v1
    - source: dict (optional metadata)
    - reference_metrics: dict
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(payload, dict):
        return {
            "schema_version": PARITY_REFERENCE_VALIDATION_SCHEMA_VERSION,
            "valid": False,
            "errors": ["Payload de référence invalide: objet JSON attendu."],
            "warnings": [],
            "normalized_payload": {},
        }

    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != PARITY_REFERENCE_SCHEMA_VERSION:
        if schema_version:
            errors.append(
                f"schema_version invalide: `{schema_version}` (attendu: `{PARITY_REFERENCE_SCHEMA_VERSION}`)."
            )
        elif not allow_legacy:
            errors.append(f"schema_version manquante (attendu: `{PARITY_REFERENCE_SCHEMA_VERSION}`).")
        else:
            warnings.append(
                "Format legacy détecté (sans schema_version). Migration automatique vers "
                f"`{PARITY_REFERENCE_SCHEMA_VERSION}`."
            )

    raw_metrics: dict[str, Any] = {}
    if isinstance(payload.get("reference_metrics"), dict):
        raw_metrics = payload.get("reference_metrics") or {}
    elif allow_legacy:
        raw_metrics = dict(payload)
    else:
        errors.append("Champ `reference_metrics` manquant ou invalide.")

    normalized_metrics = normalize_metrics(raw_metrics)
    if not normalized_metrics:
        errors.append("Aucune métrique numérique exploitable trouvée dans `reference_metrics`.")

    for metric_name in PARITY_REQUIRED_METRICS:
        if metric_name not in normalized_metrics:
            errors.append(f"Métrique requise manquante: `{metric_name}`.")

    for metric_name, value in normalized_metrics.items():
        if not math.isfinite(float(value)):
            errors.append(f"Métrique non finie: `{metric_name}` = {value}.")

    for count_name in ("trade_count", "entry_count", "exit_count"):
        if count_name in normalized_metrics:
            value = float(normalized_metrics[count_name])
            if value < 0:
                errors.append(f"Métrique `{count_name}` doit être >= 0.")
            if not float(value).is_integer():
                warnings.append(f"Métrique `{count_name}` est non entière ({value}).")

    if "max_drawdown_pct" in normalized_metrics and float(normalized_metrics["max_drawdown_pct"]) > 0:
        warnings.append(
            "Métrique `max_drawdown_pct` positive détectée: la convention habituelle est négative."
        )

    if "entry_count" in normalized_metrics and "exit_count" in normalized_metrics:
        entry_count = float(normalized_metrics["entry_count"])
        exit_count = float(normalized_metrics["exit_count"])
        mismatch = abs(entry_count - exit_count)
        mismatch_rel = _safe_rel_diff_pct(entry_count, exit_count)
        if mismatch > 0 and mismatch_rel > 2.0:
            warnings.append(
                f"Déséquilibre entrées/sorties: entry_count={entry_count}, exit_count={exit_count}."
            )

    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    raw_events = payload.get("reference_events")
    raw_trades = payload.get("reference_trades")
    if allow_legacy:
        if not isinstance(raw_events, dict):
            legacy_events = {}
            if isinstance(payload.get("entry_events"), list):
                legacy_events["entries"] = payload.get("entry_events")
            if isinstance(payload.get("exit_events"), list):
                legacy_events["exits"] = payload.get("exit_events")
            if legacy_events:
                raw_events = legacy_events
                warnings.append("Événements legacy migrés vers `reference_events`.")
        if not isinstance(raw_trades, list) and isinstance(payload.get("trades"), list):
            raw_trades = payload.get("trades")
            warnings.append("Trades legacy migrés vers `reference_trades`.")
    normalized_events = normalize_events(raw_events if isinstance(raw_events, dict) else {})
    normalized_trades = normalize_trade_pairs(raw_trades if isinstance(raw_trades, list) else [])

    if isinstance(raw_events, dict):
        for k in ("entries", "exits"):
            if isinstance(raw_events.get(k), list) and len(raw_events.get(k) or []) > 0:
                if len(normalized_events.get(k, [])) == 0:
                    warnings.append(f"Aucun timestamp valide dans `reference_events.{k}`.")
    if isinstance(raw_trades, list) and len(raw_trades) > 0 and len(normalized_trades) == 0:
        warnings.append("Aucune ligne valide dans `reference_trades`.")

    normalized_payload = build_parity_reference_payload(
        reference_metrics=normalized_metrics,
        reference_events=normalized_events,
        reference_trades=normalized_trades,
        source=source,
        generated_at_utc=payload.get("generated_at_utc"),
    )

    # Optional coherence checks with detailed references.
    entries_ref = normalized_payload.get("reference_events", {}).get("entries", [])
    exits_ref = normalized_payload.get("reference_events", {}).get("exits", [])
    trades_ref = normalized_payload.get("reference_trades", [])
    if entries_ref and "entry_count" in normalized_metrics:
        diff = _safe_rel_diff_pct(float(normalized_metrics["entry_count"]), float(len(entries_ref)))
        if diff > 5.0:
            warnings.append(
                f"entry_count ({normalized_metrics['entry_count']}) diffère des events entries ({len(entries_ref)})."
            )
    if exits_ref and "exit_count" in normalized_metrics:
        diff = _safe_rel_diff_pct(float(normalized_metrics["exit_count"]), float(len(exits_ref)))
        if diff > 5.0:
            warnings.append(
                f"exit_count ({normalized_metrics['exit_count']}) diffère des events exits ({len(exits_ref)})."
            )
    if trades_ref and "trade_count" in normalized_metrics:
        diff = _safe_rel_diff_pct(float(normalized_metrics["trade_count"]), float(len(trades_ref)))
        if diff > 5.0:
            warnings.append(
                f"trade_count ({normalized_metrics['trade_count']}) diffère des trades détaillés ({len(trades_ref)})."
            )

    valid = len(errors) == 0
    return {
        "schema_version": PARITY_REFERENCE_VALIDATION_SCHEMA_VERSION,
        "valid": bool(valid),
        "errors": errors,
        "warnings": warnings,
        "normalized_payload": normalized_payload,
    }


def build_parity_report(
    reference_metrics: dict[str, Any] | None,
    current_metrics: dict[str, Any] | None,
    thresholds: dict[str, Any] | None = None,
    reference_events: dict[str, Any] | None = None,
    current_events: dict[str, Any] | None = None,
    reference_trades: list[dict[str, Any]] | None = None,
    current_trades: list[dict[str, Any]] | None = None,
    detail_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build parity report with deterministic pass/fail checks.

    Rules:
    - Counts (`trade_count`, `entry_count`, `exit_count`) use relative diff %.
    - Percent metrics (`total_return_pct`, `max_drawdown_pct`) use absolute diff.
    """
    ref = normalize_metrics(reference_metrics)
    cur = normalize_metrics(current_metrics)
    th = dict(DEFAULT_PARITY_THRESHOLDS)
    if isinstance(thresholds, dict):
        for k, v in thresholds.items():
            try:
                th[str(k)] = float(v)
            except Exception:
                continue
    detail_th = dict(DEFAULT_PARITY_DETAIL_THRESHOLDS)
    if isinstance(detail_thresholds, dict):
        for k, v in detail_thresholds.items():
            try:
                detail_th[str(k)] = float(v)
            except Exception:
                continue

    checks = []
    for metric_name, threshold_key in (
        ("trade_count", "trade_count_rel_pct"),
        ("entry_count", "entry_count_rel_pct"),
        ("exit_count", "exit_count_rel_pct"),
        ("total_return_pct", "total_return_abs_pct"),
        ("max_drawdown_pct", "max_drawdown_abs_pct"),
    ):
        ref_present = metric_name in ref
        cur_present = metric_name in cur
        if not ref_present:
            checks.append(
                {
                    "metric": metric_name,
                    "status": "missing_reference",
                    "reference": None,
                    "current": cur.get(metric_name),
                    "diff": None,
                    "threshold": th.get(threshold_key),
                    "passed": None,
                }
            )
            continue
        if not cur_present:
            checks.append(
                {
                    "metric": metric_name,
                    "status": "missing_current",
                    "reference": ref.get(metric_name),
                    "current": None,
                    "diff": None,
                    "threshold": th.get(threshold_key),
                    "passed": False,
                }
            )
            continue

        reference_value = float(ref[metric_name])
        current_value = float(cur[metric_name])
        threshold = float(th.get(threshold_key, 0.0))
        if metric_name.endswith("_count"):
            diff = _safe_rel_diff_pct(reference_value, current_value)
            passed = diff <= threshold
            diff_unit = "rel_pct"
        else:
            diff = abs(current_value - reference_value)
            passed = diff <= threshold
            diff_unit = "abs_pct"
        checks.append(
            {
                "metric": metric_name,
                "status": "ok",
                "reference": reference_value,
                "current": current_value,
                "diff": float(diff),
                "diff_unit": diff_unit,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    required_metrics = ["trade_count", "total_return_pct", "max_drawdown_pct"]
    required_checks = [c for c in checks if c.get("metric") in required_metrics]
    # Required checks are evaluable only when status is ok/missing_current.
    has_missing_reference = any(c.get("status") == "missing_reference" for c in required_checks)
    if has_missing_reference:
        status = "insufficient_reference"
        parity_pass = None
    else:
        failed = any(c.get("passed") is False for c in required_checks)
        parity_pass = not failed
        status = "passed" if parity_pass else "failed"

    ref_events = normalize_events(reference_events)
    cur_events = normalize_events(current_events)
    ref_trades = normalize_trade_pairs(reference_trades)
    cur_trades = normalize_trade_pairs(current_trades)
    detail_checks = []
    detail_required = []

    def _append_detail_check(
        name: str,
        status_value: str,
        passed_value: bool | None,
        threshold_value: float | None,
        detail: dict[str, Any] | None = None,
    ):
        entry = {
            "name": name,
            "status": status_value,
            "passed": passed_value,
            "threshold": threshold_value,
        }
        if isinstance(detail, dict):
            entry.update(detail)
        detail_checks.append(entry)
        if status_value not in ("not_applicable", "no_reference"):
            detail_required.append(entry)

    # Entry/exit events detail parity (if reference events are provided).
    for event_name, count_key, match_key in (
        ("entries", "entry_event_count_rel_pct", "entry_event_match_min_ratio"),
        ("exits", "exit_event_count_rel_pct", "exit_event_match_min_ratio"),
    ):
        ref_list = ref_events.get(event_name, [])
        cur_list = cur_events.get(event_name, [])
        if len(ref_list) == 0:
            _append_detail_check(
                name=f"{event_name}_events",
                status_value="no_reference",
                passed_value=None,
                threshold_value=None,
                detail={"reference_count": 0, "current_count": len(cur_list)},
            )
            continue
        count_diff = _safe_rel_diff_pct(float(len(ref_list)), float(len(cur_list)))
        count_thr = float(detail_th.get(count_key, 0.0))
        count_pass = count_diff <= count_thr
        matched = _count_time_matches(
            reference=ref_list,
            current=cur_list,
            tolerance_sec=float(detail_th.get("event_time_tolerance_sec", 0.0)),
        )
        match_ratio = float(matched) / max(1, len(ref_list))
        ratio_thr = float(detail_th.get(match_key, 0.0))
        ratio_pass = match_ratio >= ratio_thr
        _append_detail_check(
            name=f"{event_name}_events",
            status_value="ok",
            passed_value=bool(count_pass and ratio_pass),
            threshold_value=ratio_thr,
            detail={
                "reference_count": len(ref_list),
                "current_count": len(cur_list),
                "count_diff_rel_pct": float(count_diff),
                "count_diff_threshold": count_thr,
                "matched_count": int(matched),
                "match_ratio": float(match_ratio),
                "match_ratio_threshold": ratio_thr,
                "tolerance_sec": float(detail_th.get("event_time_tolerance_sec", 0.0)),
            },
        )

    # Trade detail parity (if reference trades are provided).
    if len(ref_trades) == 0:
        _append_detail_check(
            name="trades_detailed",
            status_value="no_reference",
            passed_value=None,
            threshold_value=None,
            detail={"reference_count": 0, "current_count": len(cur_trades)},
        )
    else:
        trade_count_diff = _safe_rel_diff_pct(float(len(ref_trades)), float(len(cur_trades)))
        trade_count_thr = float(th.get("trade_count_rel_pct", 0.0))
        trade_count_pass = trade_count_diff <= trade_count_thr
        trade_matches = _count_trade_matches(
            reference=ref_trades,
            current=cur_trades,
            tolerance_sec=float(detail_th.get("trade_time_tolerance_sec", 0.0)),
        )
        trade_ratio = float(trade_matches) / max(1, len(ref_trades))
        trade_ratio_thr = float(detail_th.get("trade_match_min_ratio", 0.0))
        trade_ratio_pass = trade_ratio >= trade_ratio_thr
        _append_detail_check(
            name="trades_detailed",
            status_value="ok",
            passed_value=bool(trade_count_pass and trade_ratio_pass),
            threshold_value=trade_ratio_thr,
            detail={
                "reference_count": len(ref_trades),
                "current_count": len(cur_trades),
                "count_diff_rel_pct": float(trade_count_diff),
                "count_diff_threshold": trade_count_thr,
                "matched_count": int(trade_matches),
                "match_ratio": float(trade_ratio),
                "match_ratio_threshold": trade_ratio_thr,
                "tolerance_sec": float(detail_th.get("trade_time_tolerance_sec", 0.0)),
            },
        )

    detail_available = bool(
        len(ref_events.get("entries", [])) > 0
        or len(ref_events.get("exits", [])) > 0
        or len(ref_trades) > 0
    )
    detail_failed = any(c.get("passed") is False for c in detail_required)
    detail_pass = None
    if detail_available:
        detail_pass = not detail_failed

    if parity_pass is True and detail_pass is False:
        parity_pass = False
        status = "failed"
    elif parity_pass is None and detail_pass is False:
        status = "failed"

    return {
        "schema_version": "pine_parity_report.v1",
        "generated_at_utc": _utc_now_iso(),
        "status": status,
        "parity_pass": parity_pass,
        "checks": checks,
        "detail_checks": detail_checks,
        "detail_available": detail_available,
        "detail_pass": detail_pass,
        "required_metrics": required_metrics,
        "thresholds": th,
        "detail_thresholds": detail_th,
        "reference_metrics": ref,
        "current_metrics": cur,
        "reference_events": {
            "entries_count": len(ref_events.get("entries", [])),
            "exits_count": len(ref_events.get("exits", [])),
        },
        "current_events": {
            "entries_count": len(cur_events.get("entries", [])),
            "exits_count": len(cur_events.get("exits", [])),
        },
        "reference_trades_count": len(ref_trades),
        "current_trades_count": len(cur_trades),
    }
