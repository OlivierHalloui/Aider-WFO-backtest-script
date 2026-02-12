"""Build and validate `strategy_spec.v1` artifacts from Pine scripts."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import re
from typing import Any


def _utc_now_iso() -> str:
    # Local helper keeps this module independent from Streamlit imports.
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_with_pynescript(pine_text: str) -> dict[str, Any]:
    """
    Try parsing Pine text with pynescript (optional dependency).

    This integration is intentionally conservative:
    - if pynescript is missing or parsing fails, caller can fallback to regex parser;
    - AST extraction remains incremental in V3, but this validates parser readiness.
    """
    text = str(pine_text or "")
    info: dict[str, Any] = {
        "available": False,
        "version": None,
        "parse_ok": False,
        "entrypoint": None,
        "error": None,
    }
    try:
        pynescript = importlib.import_module("pynescript")
    except Exception as e:
        info["error"] = f"pynescript import failed: {e}"
        return info

    info["available"] = True
    try:
        info["version"] = importlib.metadata.version("pynescript")
    except Exception:
        info["version"] = None

    candidates: list[tuple[str, str]] = [
        ("pynescript", "parse"),
        ("pynescript", "loads"),
        ("pynescript.parser", "parse"),
        ("pynescript.ast", "parse"),
    ]
    attempted_errors: list[str] = []
    for module_name, attr_name in candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            attempted_errors.append(f"{module_name}.{attr_name}: import failed ({e})")
            continue
        fn = getattr(mod, attr_name, None)
        if not callable(fn):
            attempted_errors.append(f"{module_name}.{attr_name}: not callable")
            continue

        # Try common parse signatures.
        for call_style in ("positional", "source", "code", "text", "script"):
            try:
                if call_style == "positional":
                    result = fn(text)
                elif call_style == "source":
                    result = fn(source=text)
                elif call_style == "code":
                    result = fn(code=text)
                elif call_style == "text":
                    result = fn(text=text)
                else:
                    result = fn(script=text)
            except TypeError as e:
                attempted_errors.append(f"{module_name}.{attr_name}({call_style}): {e}")
                continue
            except Exception as e:
                attempted_errors.append(f"{module_name}.{attr_name}({call_style}): {e}")
                continue
            info["parse_ok"] = result is not None
            info["entrypoint"] = f"{module_name}.{attr_name}({call_style})"
            if not info["parse_ok"]:
                attempted_errors.append(f"{module_name}.{attr_name}({call_style}): returned None")
                continue
            return info

    info["error"] = "; ".join(attempted_errors[:8]) if attempted_errors else "No parser entrypoint succeeded."
    return info


def _resolve_parser_backend(pine_text: str, parser_backend: str) -> dict[str, Any]:
    requested = str(parser_backend or "auto").strip().lower()
    if requested not in {"auto", "regex", "pynescript"}:
        requested = "auto"

    parse_info: dict[str, Any] = {
        "requested": requested,
        "used": "regex",
        "fallback_to_regex": False,
        "fallback_reason": None,
        "pynescript": {
            "available": False,
            "version": None,
            "parse_ok": False,
            "entrypoint": None,
            "error": "not_attempted",
        },
    }

    if requested == "regex":
        parse_info["used"] = "regex"
        parse_info["pynescript"]["error"] = "disabled_by_config"
        return parse_info

    py_info = _parse_with_pynescript(pine_text)
    parse_info["pynescript"] = py_info
    if py_info.get("available") and py_info.get("parse_ok"):
        parse_info["used"] = "pynescript"
        return parse_info

    parse_info["used"] = "regex"
    if requested == "pynescript":
        parse_info["fallback_to_regex"] = True
        parse_info["fallback_reason"] = py_info.get("error") or "pynescript parse failed"
    return parse_info


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


def _strip_inline_comment(line: str) -> str:
    """Remove `//` comments while preserving quoted strings."""
    out = []
    in_string = False
    quote = ""
    i = 0
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if in_string:
            out.append(ch)
            if ch == quote and (i == 0 or line[i - 1] != "\\"):
                in_string = False
        else:
            if ch in ("'", '"'):
                in_string = True
                quote = ch
                out.append(ch)
            elif ch == "/" and nxt == "/":
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _extract_named_string_arg(call_text: str, arg_name: str) -> str:
    """
    Extract string argument value from call text.

    Examples:
    - id='Buy long'
    - id = "Buy long"
    """
    pattern = re.compile(
        rf"""\b{re.escape(arg_name)}\s*=\s*(['"])(.*?)\1""",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(call_text or "")
    if match:
        return str(match.group(2) or "").strip()
    return ""


def _extract_call_action(line: str) -> str:
    for action in ("entry", "exit", "close", "cancel"):
        if re.search(rf"""\bstrategy\.{action}\s*\(""", line, flags=re.IGNORECASE):
            return action
    return ""


def _parse_assignment(stripped_line: str) -> dict[str, Any] | None:
    """
    Parse Pine assignment lines into normalized assignment rows.

    Supported examples:
    - `x = expr`
    - `x := expr`
    - `float x = expr`
    - `[a, b, c] = ta.bb(...)`
    """
    line = str(stripped_line or "").strip()
    if not line:
        return None
    lowered = line.lower()
    if lowered.startswith(("if ", "for ", "while ", "switch ", "return ", "strategy.")):
        return None
    if "=>" in line:
        # Function declaration/lambda-like block.
        return None

    op = None
    split_at = -1
    if ":=" in line:
        split_at = line.find(":=")
        op = ":="
    else:
        # First plain '=' not part of comparison operators.
        m = re.search(r"(?<![=!<>])=(?!=)", line)
        if m:
            split_at = m.start()
            op = "="
    if split_at <= 0 or not op:
        return None

    lhs = line[:split_at].strip()
    rhs = line[split_at + len(op) :].strip()
    if not lhs or not rhs:
        return None

    if lhs.startswith("[") and lhs.endswith("]"):
        raw_targets = [x.strip() for x in lhs[1:-1].split(",")]
        targets = [x for x in raw_targets if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", x)]
        if not targets:
            return None
        return {"targets": targets, "op": op, "expr": rhs}

    if "(" in lhs and ")" in lhs:
        # Probably function definition or destructuring not handled here.
        return None

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", lhs)
    if not tokens:
        return None
    target = tokens[-1]
    return {"targets": [target], "op": op, "expr": rhs}


def _split_top_level_csv(text: str) -> list[str]:
    """Split comma-separated args at top-level (ignores nested scopes/strings)."""
    out: list[str] = []
    depth = 0
    in_string = False
    quote = ""
    start = 0
    src = str(text or "")
    for i, ch in enumerate(src):
        if in_string:
            if ch == quote and (i == 0 or src[i - 1] != "\\"):
                in_string = False
            continue
        if ch in ("'", '"'):
            in_string = True
            quote = ch
            continue
        if ch in "([{":
            depth += 1
            continue
        if ch in ")]}":
            depth = max(0, depth - 1)
            continue
        if ch == "," and depth == 0:
            out.append(src[start:i].strip())
            start = i + 1
    tail = src[start:].strip()
    if tail:
        out.append(tail)
    return out


def _extract_request_security_details(expr: str) -> dict[str, Any] | None:
    """
    Parse `request.security(...)` call details from an expression string.

    Returns normalized fields used by runtime/parity diagnostics.
    """
    text = str(expr or "").strip()
    match = re.match(r"^request\.security\s*\((.*)\)\s*$", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    args = _split_top_level_csv(match.group(1))
    if len(args) < 3:
        return None

    symbol_expr = args[0]
    timeframe_expr = args[1]
    expression_expr = args[2]

    gaps_expr = None
    lookahead_expr = None
    for arg in args[3:]:
        if not isinstance(arg, str):
            continue
        lowered = arg.lower()
        if lowered.startswith("gaps"):
            parts = arg.split("=", 1)
            if len(parts) == 2:
                gaps_expr = parts[1].strip()
        elif lowered.startswith("lookahead"):
            parts = arg.split("=", 1)
            if len(parts) == 2:
                lookahead_expr = parts[1].strip()

    return {
        "symbol_expr": symbol_expr,
        "timeframe_expr": timeframe_expr,
        "expression_expr": expression_expr,
        "args_count": len(args),
        "gaps_expr": gaps_expr,
        "lookahead_expr": lookahead_expr,
    }


def _build_logical_lines(text: str) -> list[dict[str, Any]]:
    """
    Build logical statements by merging multiline calls (parentheses/brackets/braces).
    """
    rows: list[dict[str, Any]] = []
    acc: list[str] = []
    start_line = 1
    start_indent = 0
    depth = 0
    in_string = False
    quote = ""

    def _append_current():
        nonlocal acc, start_line, start_indent, depth, in_string, quote
        if not acc:
            return
        merged = " ".join(part.strip() for part in acc if str(part).strip()).strip()
        if merged:
            rows.append({"line": start_line, "indent": start_indent, "text": merged})
        acc = []
        depth = 0
        in_string = False
        quote = ""

    for line_no, raw in enumerate(str(text or "").splitlines(), start=1):
        cleaned = _strip_inline_comment(raw)
        if not cleaned.strip():
            continue
        if not acc:
            start_line = line_no
            start_indent = len(cleaned) - len(cleaned.lstrip(" "))

        acc.append(cleaned)
        for i, ch in enumerate(cleaned):
            if in_string:
                if ch == quote and (i == 0 or cleaned[i - 1] != "\\"):
                    in_string = False
                continue
            if ch in ("'", '"'):
                in_string = True
                quote = ch
                continue
            if ch in "([{":
                depth += 1
                continue
            if ch in ")]}":
                depth = max(0, depth - 1)
                continue

        if depth == 0 and not in_string:
            _append_current()

    _append_current()
    return rows


def _extract_logic_artifacts(text: str) -> dict[str, Any]:
    """
    Extract deterministic logic artifacts from Pine source for codegen/runtime.

    The goal is traceable transcription of:
    - assignments
    - order calls and their surrounding `if` conditions
    """
    assignments: list[dict[str, Any]] = []
    order_rules: list[dict[str, Any]] = []
    request_security_calls: list[dict[str, Any]] = []

    # Stack entries: (indent, condition_expr)
    if_stack: list[tuple[int, str]] = []

    logical_lines = _build_logical_lines(text)
    for row in logical_lines:
        line_no = int(row.get("line", 0) or 0)
        indent = int(row.get("indent", 0) or 0)
        stripped = str(row.get("text") or "").strip()
        if not stripped:
            continue

        while if_stack and indent <= if_stack[-1][0]:
            if_stack.pop()

        if re.match(r"^\s*if\s+", stripped, flags=re.IGNORECASE):
            cond = re.sub(r"^\s*if\s+", "", stripped, flags=re.IGNORECASE).strip()
            if cond:
                if_stack.append((indent, cond))
            continue

        assign_row = _parse_assignment(stripped)
        if assign_row:
            expr = str(assign_row["expr"] or "")
            request_details = _extract_request_security_details(expr)
            if isinstance(request_details, dict):
                request_security_calls.append(
                    {
                        "line": line_no,
                        "targets": assign_row["targets"],
                        "expr": expr,
                        **request_details,
                    }
                )
            assignments.append(
                {
                    "line": line_no,
                    "targets": assign_row["targets"],
                    "op": assign_row["op"],
                    "expr": assign_row["expr"],
                }
            )

        action = _extract_call_action(stripped)
        if action:
            current_conditions = [expr for _, expr in if_stack]
            condition_expr = " and ".join(current_conditions) if current_conditions else "true"
            order_id = _extract_named_string_arg(stripped, "id")
            direction = _extract_named_string_arg(stripped, "direction")
            if not order_id:
                # Fallback: first string literal often carries id for close/cancel.
                first_string = re.search(r"""(['"])(.*?)\1""", stripped, flags=re.DOTALL)
                if first_string:
                    order_id = str(first_string.group(2) or "").strip()
            order_rules.append(
                {
                    "line": line_no,
                    "action": action,
                    "id": order_id or None,
                    "direction": direction or None,
                    "condition_expr": condition_expr,
                    "call": stripped,
                }
            )

    return {
        "assignments": assignments,
        "order_rules": order_rules,
        "request_security_calls": request_security_calls,
        "assignment_count": len(assignments),
        "order_rule_count": len(order_rules),
        "request_security_count": len(request_security_calls),
    }


def build_strategy_spec_v1_from_pine_text(
    pine_text: str,
    source_name: str = "",
    strategy_id: str = "",
    precheck_report: dict[str, Any] | None = None,
    compatibility_report: dict[str, Any] | None = None,
    parser_backend: str = "auto",
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

    parser_meta = _resolve_parser_backend(pine_text=text, parser_backend=parser_backend)

    # Current V3 extraction remains deterministic and regex-based.
    # When pynescript parse succeeds, we keep extraction identical but record parser telemetry
    # so we can progressively switch extraction paths without contract break.
    imports = _extract_imports(text)
    imports_detail = _extract_imports_detail(text)
    inputs = _extract_inputs(text)
    logic = _extract_logic_artifacts(text)

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
    if bool(parser_meta.get("fallback_to_regex")):
        fallback_reason = str(parser_meta.get("fallback_reason") or "").strip()
        warnings.append(
            f"Parser pynescript indisponible/invalide: fallback regex activé ({fallback_reason or 'raison inconnue'})."
        )

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
        "logic": logic,
        "transcription": {
            "parser_backend_requested": parser_meta.get("requested"),
            "parser_backend_used": parser_meta.get("used"),
            "fallback_to_regex": bool(parser_meta.get("fallback_to_regex", False)),
            "fallback_reason": parser_meta.get("fallback_reason"),
            "pynescript": parser_meta.get("pynescript"),
        },
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

    logic = spec.get("logic")
    if logic is not None:
        if not isinstance(logic, dict):
            errors.append("logic must be an object when provided.")
        else:
            assignments = logic.get("assignments")
            if assignments is not None:
                if not isinstance(assignments, list):
                    errors.append("logic.assignments must be an array.")
                else:
                    for idx, row in enumerate(assignments):
                        if not isinstance(row, dict):
                            errors.append(f"logic.assignments[{idx}] must be an object.")
                            continue
                        if not isinstance(row.get("line"), int):
                            errors.append(f"logic.assignments[{idx}].line must be an integer.")
                        targets = row.get("targets")
                        if not isinstance(targets, list) or any(not isinstance(x, str) for x in targets):
                            errors.append(
                                f"logic.assignments[{idx}].targets must be an array of strings."
                            )
                        if not isinstance(row.get("expr"), str) or not str(row.get("expr")).strip():
                            errors.append(f"logic.assignments[{idx}].expr must be a non-empty string.")
            order_rules = logic.get("order_rules")
            if order_rules is not None:
                if not isinstance(order_rules, list):
                    errors.append("logic.order_rules must be an array.")
                else:
                    for idx, row in enumerate(order_rules):
                        if not isinstance(row, dict):
                            errors.append(f"logic.order_rules[{idx}] must be an object.")
                            continue
                        if not isinstance(row.get("line"), int):
                            errors.append(f"logic.order_rules[{idx}].line must be an integer.")
                        if not isinstance(row.get("action"), str):
                            errors.append(f"logic.order_rules[{idx}].action must be a string.")
                        if not isinstance(row.get("condition_expr"), str):
                            errors.append(
                                f"logic.order_rules[{idx}].condition_expr must be a string."
                            )
            request_security_calls = logic.get("request_security_calls")
            if request_security_calls is not None:
                if not isinstance(request_security_calls, list):
                    errors.append("logic.request_security_calls must be an array.")
                else:
                    for idx, row in enumerate(request_security_calls):
                        if not isinstance(row, dict):
                            errors.append(
                                f"logic.request_security_calls[{idx}] must be an object."
                            )
                            continue
                        if not isinstance(row.get("line"), int):
                            errors.append(
                                f"logic.request_security_calls[{idx}].line must be an integer."
                            )
                        targets = row.get("targets")
                        if not isinstance(targets, list) or any(not isinstance(x, str) for x in targets):
                            errors.append(
                                f"logic.request_security_calls[{idx}].targets must be an array of strings."
                            )
                        for key in ("expr", "timeframe_expr", "expression_expr"):
                            if not isinstance(row.get(key), str) or not str(row.get(key)).strip():
                                errors.append(
                                    f"logic.request_security_calls[{idx}].{key} must be a non-empty string."
                                )

    transcription = spec.get("transcription")
    if transcription is not None:
        if not isinstance(transcription, dict):
            errors.append("transcription must be an object when provided.")
        else:
            for key in ("parser_backend_requested", "parser_backend_used"):
                if key in transcription and not isinstance(transcription.get(key), str):
                    errors.append(f"transcription.{key} must be a string.")
            if (
                "fallback_to_regex" in transcription
                and not isinstance(transcription.get("fallback_to_regex"), bool)
            ):
                errors.append("transcription.fallback_to_regex must be boolean.")
            pynescript_meta = transcription.get("pynescript")
            if pynescript_meta is not None and not isinstance(pynescript_meta, dict):
                errors.append("transcription.pynescript must be an object when provided.")

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
