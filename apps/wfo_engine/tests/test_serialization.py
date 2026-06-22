"""Tests for domain/serialization.py — no VBT dependency."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from domain.serialization import sanitize_for_json, sha256_json, to_jsonable


class TestSanitizeForJsonNanInf:
    def test_float_nan_becomes_none(self):
        assert sanitize_for_json(float("nan")) is None

    def test_float_pos_inf_becomes_none(self):
        assert sanitize_for_json(float("inf")) is None

    def test_float_neg_inf_becomes_none(self):
        assert sanitize_for_json(float("-inf")) is None

    def test_float_finite_unchanged(self):
        assert sanitize_for_json(3.14) == pytest.approx(3.14)

    def test_float_zero_unchanged(self):
        assert sanitize_for_json(0.0) == 0.0

    def test_np_floating_nan_becomes_none(self):
        assert sanitize_for_json(np.float64("nan")) is None

    def test_np_floating_inf_becomes_none(self):
        assert sanitize_for_json(np.float64("inf")) is None

    def test_np_floating_finite_survives(self):
        result = sanitize_for_json(np.float64(1.5))
        assert result == pytest.approx(1.5)
        assert isinstance(result, float)

    def test_nested_dict_nan(self):
        d = {"a": float("nan"), "b": {"c": float("inf")}}
        result = sanitize_for_json(d)
        assert result == {"a": None, "b": {"c": None}}

    def test_nested_list_nan(self):
        result = sanitize_for_json([float("nan"), 1.0, float("-inf")])
        assert result == [None, 1.0, None]

    def test_tuple_converted_to_list(self):
        result = sanitize_for_json((float("nan"), 2))
        assert result == [None, 2]

    def test_result_is_valid_json(self):
        payload = {
            "score": float("nan"),
            "pnl": float("inf"),
            "trades": 5,
            "metrics": {"sharpe": float("-inf"), "win_rate": 0.6},
        }
        sanitized = sanitize_for_json(payload)
        # Must not raise with allow_nan=False
        serialized = json.dumps(sanitized, allow_nan=False)
        parsed = json.loads(serialized)
        assert parsed["score"] is None
        assert parsed["pnl"] is None
        assert parsed["metrics"]["sharpe"] is None
        assert parsed["trades"] == 5


class TestSanitizeForJsonTypes:
    def test_np_integer(self):
        result = sanitize_for_json(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_bool_(self):
        assert sanitize_for_json(np.bool_(True)) is True
        assert sanitize_for_json(np.bool_(False)) is False

    def test_string_passthrough(self):
        assert sanitize_for_json("hello") == "hello"

    def test_int_passthrough(self):
        assert sanitize_for_json(7) == 7

    def test_none_passthrough(self):
        assert sanitize_for_json(None) is None


class TestSha256Json:
    def test_stable_hash(self):
        d = {"a": 1, "b": 2.5}
        assert sha256_json(d) == sha256_json(d)

    def test_nan_payload_does_not_crash(self):
        # sha256_json calls sanitize_for_json internally — NaN should not raise
        result = sha256_json({"score": float("nan")})
        assert result is not None
        assert len(result) == 64
