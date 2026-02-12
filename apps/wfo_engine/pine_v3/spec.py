"""Build and validate `strategy_spec.v1` artifacts from Pine scripts."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def _utc_now_iso() -> str:
    # Local helper keeps this module independent from Streamlit imports.
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_identifier(value: str, default: str = "pine_strategy") -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def _extract_call_block(text: str, call_name: str) -> str:
    """Extract a function call block including nested parentheses."""
    pattern = re.compile(rf"\b{re.escape(call_name)}\s*\(", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    start = match.start()
    idx = match.end() - 1  # points to '('
    depth = 0
    in_string = False
    quote = ""
    i = idx
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == quote and (i == 0 or text[i - 1] != "\\"):
                in_string = False
        else:
            if ch in ("'", '"'):
                in_string = True
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    return text[start:]  # unbalanced fallback


def _extract_strategy_name(strategy_call_block: str) -> str:
    if not strategy_call_block:
        return ""
    title_match = re.search(
        r"""title\s*=\s*(['"])(.*?)\1""",
        strategy_call_block,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if title_match:
        return title_match.group(2).strip()
    first_string = re.search(r"""(['"])(.*?)\1""", strategy_call_block, flags=re.DOTALL)
    if first_string:
        return first_string.group(2).strip()
    return ""


def _extract_imports(text: str) -> list[str]:
    lines = re.findall(r"^\s*import\s+.+$", text, flags=re.IGNORECASE | re.MULTILINE)
    return [ln.strip() for ln in lines]


def _extract_imports_detail(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _extract_imports(text):
        line = str(raw or "").strip()
        match = re.match(
            r"^\s*import\s+([A-Za-z0-9_./-]+)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            rows.append({"raw": line, "module_ref": None, "alias": None, "parse_ok": False})
            continue
        rows.append(
            {
                "raw": line,
                "module_ref": str(match.group(1) or "").strip(),
                "alias": str(match.group(2) or "").strip(),
                "parse_ok": True,
            }
        )
    return rows


def _extract_inputs(text: str) -> list[dict[str, Any]]:
    """
    Extract simple `var = input.xxx(...)` declarations.

    This parser is intentionally conservative for V3 lot 0:
    it collects high-value metadata without claiming full Pine parsing.
    """
    inputs: list[dict[str, Any]] = []
    input_re = re.compile(
        r"""^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*input(?:\.(int|float|bool|string|source|timeframe))?\s*\((.*)\)\s*$""",
        flags=re.IGNORECASE,
    )
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        m = input_re.match(line)
        if not m:
            continue
        name = m.group(1)
        input_type = (m.group(2) or "generic").lower()
        args = m.group(3) or ""
        defval_match = re.search(r"""defval\s*=\s*([^,)\n]+)""", args, flags=re.IGNORECASE)
        title_match = re.search(r"""title\s*=\s*(['"])(.*?)\1""", args, flags=re.IGNORECASE)
        group_match = re.search(r"""group\s*=\s*(['"])(.*?)\1""", args, flags=re.IGNORECASE)
        item: dict[str, Any] = {
            "name": name,
            "type": input_type,
            "line": line_no,
            "default_raw": defval_match.group(1).strip() if defval_match else None,
            "title": title_match.group(2).strip() if title_match else None,
            "group": group_match.group(2).strip() if group_match else None,
        }
        inputs.append(item)
    return inputs


def build_strategy_spec_v1_from_pine_text(
    pine_text: str,
    source_name: str = "",
    strategy_id: str = "",
    precheck_report: dict[str, Any] | None = None,
    compatibility_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a first-pass `strategy_spec.v1` object from Pine source text."""
    text = str(pine_text or "")
    source_sha1 = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    line_count = text.count("\n") + (1 if text else 0)
    char_count = len(text)
    version_match = re.search(r"//@version\s*=\s*(\d+)", text, flags=re.IGNORECASE)
    pine_version = int(version_match.group(1)) if version_match else None

    strategy_call = _extract_call_block(text, "strategy")
    strategy_name = _extract_strategy_name(strategy_call) or "Imported Pine Strategy"
    normalized_name = _normalize_identifier(strategy_name, default="imported_pine_strategy")
    final_strategy_id = str(strategy_id or f"pine_{normalized_name}")

    imports = _extract_imports(text)
    imports_detail = _extract_imports_detail(text)
    inputs = _extract_inputs(text)

    detected_precheck = precheck_report.get("detected_features", {}) if isinstance(precheck_report, dict) else {}
    capabilities = {
        "uses_request_security": bool(
            detected_precheck.get("uses_request_security")
            if isinstance(detected_precheck, dict) and "uses_request_security" in detected_precheck
            else re.search(r"\brequest\.security\s*\(", text, flags=re.IGNORECASE)
        ),
        "uses_request_security_lower_tf": bool(
            detected_precheck.get("uses_request_security_lower_tf")
            if isinstance(detected_precheck, dict) and "uses_request_security_lower_tf" in detected_precheck
            else re.search(r"\brequest\.security_lower_tf\s*\(", text, flags=re.IGNORECASE)
        ),
        "uses_strategy_entry": bool(
            detected_precheck.get("uses_strategy_entry")
            if isinstance(detected_precheck, dict) and "uses_strategy_entry" in detected_precheck
            else re.search(r"\bstrategy\.entry\s*\(", text, flags=re.IGNORECASE)
        ),
        "uses_strategy_exit": bool(
            detected_precheck.get("uses_strategy_exit")
            if isinstance(detected_precheck, dict) and "uses_strategy_exit" in detected_precheck
            else re.search(r"\bstrategy\.exit\s*\(", text, flags=re.IGNORECASE)
        ),
        "uses_strategy_close": bool(
            detected_precheck.get("uses_strategy_close")
            if isinstance(detected_precheck, dict) and "uses_strategy_close" in detected_precheck
            else re.search(r"\bstrategy\.close\s*\(", text, flags=re.IGNORECASE)
        ),
        "uses_strategy_cancel": bool(
            detected_precheck.get("uses_strategy_cancel")
            if isinstance(detected_precheck, dict) and "uses_strategy_cancel" in detected_precheck
            else re.search(r"\bstrategy\.cancel\s*\(", text, flags=re.IGNORECASE)
        ),
    }

    warnings: list[str] = []
    if imports:
        warnings.append("Imports Pine externes détectés: mapping Python local requis.")
    if capabilities["uses_request_security_lower_tf"]:
        warnings.append("request.security_lower_tf détecté: support partiel/non garanti.")

    return {
        "schema_version": "strategy_spec.v1",
        "generated_at_utc": _utc_now_iso(),
        "strategy": {
            "id": final_strategy_id,
            "name": strategy_name,
            "kind": "pine_imported",
        },
        "source": {
            "file_name": str(source_name or ""),
            "pine_version": pine_version,
            "source_sha1": source_sha1,
            "line_count": line_count,
            "char_count": char_count,
        },
        "imports": imports,
        "imports_detail": imports_detail,
        "import_resolution": (
            compatibility_report.get("import_resolution", [])
            if isinstance(compatibility_report, dict)
            else []
        ),
        "inputs": inputs,
        "capabilities": capabilities,
        "compatibility_summary": compatibility_report if isinstance(compatibility_report, dict) else None,
        "warnings": warnings,
    }


def validate_strategy_spec_v1(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate strict minimum contract for `strategy_spec.v1`."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(spec, dict):
        errors.append("Spec must be a JSON object.")
        return {
            "schema_version": "strategy_spec_validation.v1",
            "valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    if spec.get("schema_version") != "strategy_spec.v1":
        errors.append("schema_version must be 'strategy_spec.v1'.")

    for key in ("generated_at_utc", "strategy", "source", "imports", "inputs", "capabilities", "warnings"):
        if key not in spec:
            errors.append(f"Missing required key: {key}")

    strategy = spec.get("strategy")
    if not isinstance(strategy, dict):
        errors.append("strategy must be an object.")
    else:
        for key in ("id", "name", "kind"):
            if key not in strategy:
                errors.append(f"strategy.{key} is required.")
            elif not isinstance(strategy.get(key), str) or not str(strategy.get(key)).strip():
                errors.append(f"strategy.{key} must be a non-empty string.")
        if strategy.get("kind") != "pine_imported":
            errors.append("strategy.kind must be 'pine_imported'.")

    source = spec.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object.")
    else:
        required_source_types = {
            "file_name": str,
            "source_sha1": str,
            "line_count": int,
            "char_count": int,
        }
        for key, expected_type in required_source_types.items():
            if key not in source:
                errors.append(f"source.{key} is required.")
            else:
                val = source.get(key)
                if expected_type is int:
                    if not isinstance(val, int):
                        errors.append(f"source.{key} must be an integer.")
                elif not isinstance(val, expected_type):
                    errors.append(f"source.{key} must be of type {expected_type.__name__}.")
        pine_version = source.get("pine_version")
        if pine_version is not None and not isinstance(pine_version, int):
            errors.append("source.pine_version must be an integer or null.")

    imports = spec.get("imports")
    if not isinstance(imports, list):
        errors.append("imports must be an array.")
    elif any(not isinstance(x, str) for x in imports):
        errors.append("imports must contain only strings.")

    imports_detail = spec.get("imports_detail")
    if imports_detail is not None:
        if not isinstance(imports_detail, list):
            errors.append("imports_detail must be an array when provided.")
        elif any(not isinstance(x, dict) for x in imports_detail):
            errors.append("imports_detail must contain only objects.")

    import_resolution = spec.get("import_resolution")
    if import_resolution is not None:
        if not isinstance(import_resolution, list):
            errors.append("import_resolution must be an array when provided.")
        elif any(not isinstance(x, dict) for x in import_resolution):
            errors.append("import_resolution must contain only objects.")

    inputs = spec.get("inputs")
    if not isinstance(inputs, list):
        errors.append("inputs must be an array.")
    else:
        for idx, item in enumerate(inputs):
            if not isinstance(item, dict):
                errors.append(f"inputs[{idx}] must be an object.")
                continue
            for key in ("name", "type", "line"):
                if key not in item:
                    errors.append(f"inputs[{idx}].{key} is required.")
            if "name" in item and not isinstance(item.get("name"), str):
                errors.append(f"inputs[{idx}].name must be a string.")
            if "type" in item and not isinstance(item.get("type"), str):
                errors.append(f"inputs[{idx}].type must be a string.")
            if "line" in item and not isinstance(item.get("line"), int):
                errors.append(f"inputs[{idx}].line must be an integer.")

    capabilities = spec.get("capabilities")
    cap_keys = (
        "uses_request_security",
        "uses_request_security_lower_tf",
        "uses_strategy_entry",
        "uses_strategy_exit",
        "uses_strategy_close",
        "uses_strategy_cancel",
    )
    if not isinstance(capabilities, dict):
        errors.append("capabilities must be an object.")
    else:
        for key in cap_keys:
            if key not in capabilities:
                errors.append(f"capabilities.{key} is required.")
            elif not isinstance(capabilities.get(key), bool):
                errors.append(f"capabilities.{key} must be boolean.")

    spec_warnings = spec.get("warnings")
    if not isinstance(spec_warnings, list):
        errors.append("warnings must be an array.")
    elif any(not isinstance(x, str) for x in spec_warnings):
        errors.append("warnings must contain only strings.")

    if isinstance(imports, list) and len(imports) > 0:
        warnings.append("External imports are present and require explicit mapping.")
    if isinstance(source, dict) and source.get("pine_version") not in (None, 6):
        warnings.append("Pine version differs from v6 target.")

    return {
        "schema_version": "strategy_spec_validation.v1",
        "validated_at_utc": _utc_now_iso(),
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
