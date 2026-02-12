"""Runtime/time estimation helpers for long optimization jobs."""

from __future__ import annotations

import datetime

import numpy as np


def timeframe_to_seconds(tf: str) -> int | None:
    if not isinstance(tf, str) or len(tf) < 2:
        return None
    unit = tf[-1].lower()
    try:
        value = int(tf[:-1])
    except Exception:
        return None
    if value <= 0:
        return None
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    return None


def parse_iso_date(value: str) -> datetime.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        if len(text) <= 10:
            return datetime.datetime.combine(datetime.date.fromisoformat(text[:10]), datetime.time.min)
        return datetime.datetime.fromisoformat(text)
    except Exception:
        return None


def humanize_seconds(seconds: float | int | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "n/a"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    mins, sec = divmod(seconds, 60)
    if mins < 60:
        return f"{mins}m {sec}s"
    hours, mins = divmod(mins, 60)
    if hours < 24:
        return f"{hours}h {mins}m"
    days, hours = divmod(hours, 24)
    return f"{days}j {hours}h"


def parse_utc_iso(value: str) -> datetime.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except Exception:
        return None


def compute_running_elapsed_seconds(job_state: dict) -> float | None:
    if not isinstance(job_state, dict):
        return None
    started = parse_utc_iso(job_state.get("started_at_utc"))
    if started is None:
        return None
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if started.tzinfo is None:
        started = started.replace(tzinfo=datetime.timezone.utc)
    elapsed = (now_utc - started).total_seconds()
    if elapsed < 0:
        return None
    return float(elapsed)
