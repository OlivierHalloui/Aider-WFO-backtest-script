"""Pine V3 UI helper functions extracted from app.py.

This module contains all Pine-related utility functions used by the
WFOE Streamlit application: file persistence, pre-check analysis,
compatibility reporting, beta readiness, parity helpers, and artifact
building for expert diagnostics and ZIP export.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import json
import os
import re
import time

import numpy as np
import pandas as pd
import streamlit as st

from domain.serialization import (
    sanitize_for_json as _sanitize_for_json,
    utc_now_iso as _utc_now_iso,
    sha256_json as _sha256_json,
    safe_float_scalar as _safe_float_scalar,
)
from pine_v3.parity import (
    validate_parity_reference_payload as _validate_parity_reference_payload,
    DEFAULT_PARITY_THRESHOLDS as _DEFAULT_PARITY_THRESHOLDS,
    DEFAULT_PARITY_DETAIL_THRESHOLDS as _DEFAULT_PARITY_DETAIL_THRESHOLDS,
    normalize_metrics as _normalize_parity_metrics,
)
from pine_v3.runtime_adapter import (
    build_order_semantics_report as _build_pine_order_semantics_report,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _repo_root_dir():
    """Return repository root from current app location (ui/ -> wfo_engine/ -> apps/ -> repo)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _read_text_file_with_fallback(path: str):
    """Read text file using UTF-8 then Latin-1 fallback."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace"), "latin-1"


# ---------------------------------------------------------------------------
# Pine file persistence
# ---------------------------------------------------------------------------

def _pine_imports_dir():
    """Return persistent folder used to store imported Pine strategy files."""
    folder = os.path.join(_repo_root_dir(), "reports", "pine_imports")
    os.makedirs(folder, exist_ok=True)
    return folder


def _pine_generated_dir():
    """Return persistent folder used to store generated Pine adapter modules."""
    folder = os.path.join(_pine_imports_dir(), "generated")
    os.makedirs(folder, exist_ok=True)
    return folder


def _persist_pine_source_text(source_text: str, source_name: str = "strategy_source.pine.txt"):
    """Persist Pine source text restored from ZIP and return local file path."""
    text = str(source_text or "")
    if not text.strip():
        return None
    safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(source_name or "strategy_source.pine.txt"))
    if not safe_name:
        safe_name = "strategy_source.pine.txt"
    if not safe_name.lower().endswith((".pine", ".txt")):
        safe_name = f"{safe_name}.txt"

    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    target_path = os.path.join(_pine_imports_dir(), f"zip_{digest}_{safe_name}")
    if not os.path.exists(target_path):
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)
    return target_path


def _persist_generated_strategy_text(source_text: str, source_name: str = "generated_strategy.py"):
    """Persist generated Python strategy module restored from ZIP and return local file path."""
    text = str(source_text or "")
    if not text.strip():
        return None
    safe_name = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(source_name or "generated_strategy.py")
    )
    if not safe_name:
        safe_name = "generated_strategy.py"
    if not safe_name.lower().endswith(".py"):
        safe_name = f"{safe_name}.py"

    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    target_path = os.path.join(_pine_generated_dir(), f"zipgen_{digest}_{safe_name}")
    if not os.path.exists(target_path):
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)
    return target_path


def _persist_uploaded_pine_file(uploaded_file):
    """Persist uploaded Pine/TXT strategy file to disk and return saved path."""
    if uploaded_file is None:
        return None, False
    try:
        file_id = getattr(uploaded_file, "file_id", f"{uploaded_file.name}:{uploaded_file.size}")
        existing_id = st.session_state.get("uploaded_pine_file_id")
        existing_path = st.session_state.get("uploaded_pine_file_path")
        if existing_id == file_id and isinstance(existing_path, str) and os.path.exists(existing_path):
            return existing_path, False

        base_name = os.path.basename(str(uploaded_file.name or "strategy.pine"))
        safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base_name)
        if not safe_name:
            safe_name = "strategy.pine"
        if not safe_name.lower().endswith((".pine", ".txt")):
            safe_name = f"{safe_name}.txt"

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        digest = hashlib.sha1(f"{file_id}_{time.time_ns()}".encode("utf-8")).hexdigest()[:12]
        target_path = os.path.join(_pine_imports_dir(), f"{timestamp}_{digest}_{safe_name}")
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state["uploaded_pine_file_id"] = file_id
        st.session_state["uploaded_pine_file_path"] = target_path
        st.session_state["pine_source_name"] = base_name
        st.session_state["pine_file_path"] = target_path
        return target_path, True
    except Exception as e:
        st.error(f"Error while saving uploaded Pine file: {e}")
        return None, False


def _persist_pine_library_text(source_text: str, source_name: str = "library_source.pine.txt"):
    """Persist a Pine library text restored from ZIP and return local file path."""
    text = str(source_text or "")
    if not text.strip():
        return None
    safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(source_name or "library_source.pine.txt"))
    if not safe_name:
        safe_name = "library_source.pine.txt"
    if not safe_name.lower().endswith((".pine", ".txt")):
        safe_name = f"{safe_name}.txt"

    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    target_path = os.path.join(_pine_imports_dir(), f"ziplib_{digest}_{safe_name}")
    if not os.path.exists(target_path):
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)
    return target_path


def _persist_uploaded_pine_library_files(uploaded_files):
    """
    Persist uploaded Pine library files and keep a session manifest.

    Returns:
        tuple[list[dict], bool]: (manifest, has_new_file)
    """
    files = list(uploaded_files or [])
    if len(files) == 0:
        existing = st.session_state.get("pine_library_files")
        if isinstance(existing, list):
            return existing, False
        st.session_state["pine_library_files"] = []
        st.session_state["pine_library_paths"] = []
        st.session_state["pine_library_names"] = []
        return [], False

    manifest = []
    has_new_file = False
    try:
        for uploaded in files:
            base_name = os.path.basename(str(getattr(uploaded, "name", "") or "library.pine"))
            safe_name = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in base_name)
            if not safe_name:
                safe_name = "library.pine"
            if not safe_name.lower().endswith((".pine", ".txt")):
                safe_name = f"{safe_name}.txt"

            raw = bytes(uploaded.getbuffer())
            source_sha1 = hashlib.sha1(raw).hexdigest()
            target_path = os.path.join(_pine_imports_dir(), f"lib_{source_sha1[:12]}_{safe_name}")
            if not os.path.exists(target_path):
                with open(target_path, "wb") as f:
                    f.write(raw)
                has_new_file = True

            manifest.append(
                {
                    "source_name": base_name,
                    "path": target_path,
                    "source_sha1": source_sha1,
                    "size_bytes": int(len(raw)),
                }
            )
    except Exception as e:
        st.error(f"Error while saving uploaded Pine libraries: {e}")
        return st.session_state.get("pine_library_files", []), False

    st.session_state["pine_library_files"] = manifest
    st.session_state["pine_library_paths"] = [str(item.get("path", "")) for item in manifest]
    st.session_state["pine_library_names"] = [str(item.get("source_name", "")) for item in manifest]
    return manifest, has_new_file


# ---------------------------------------------------------------------------
# Pine source parsing helpers
# ---------------------------------------------------------------------------

def _normalize_token(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _parse_pine_import_lines(import_lines: list[str] | None):
    """Parse Pine import lines into structured records."""
    rows = []
    for raw in list(import_lines or []):
        line = str(raw or "").strip()
        if not line:
            continue
        match = re.match(
            r"^\s*import\s+([A-Za-z0-9_./-]+)\s+as\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            rows.append(
                {
                    "raw": line,
                    "module_ref": None,
                    "alias": None,
                    "parse_ok": False,
                }
            )
            continue
        module_ref = str(match.group(1) or "").strip()
        alias = str(match.group(2) or "").strip()
        module_parts = [part for part in module_ref.split("/") if part]
        module_tail = module_parts[-1] if module_parts else ""
        if re.fullmatch(r"\d+", module_tail) and len(module_parts) >= 2:
            module_tail = module_parts[-2]
        rows.append(
            {
                "raw": line,
                "module_ref": module_ref,
                "alias": alias,
                "module_tail": module_tail,
                "parse_ok": True,
            }
        )
    return rows


def _extract_library_decl_name(source_text: str) -> str:
    """Best-effort parse of Pine `library(...)` declaration title/name."""
    text = str(source_text or "")
    if not text.strip():
        return ""
    lib_block_match = re.search(r"\blibrary\s*\((.*?)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if not lib_block_match:
        return ""
    block = lib_block_match.group(1)
    title_match = re.search(r"""title\s*=\s*(['"])(.*?)\1""", block, flags=re.IGNORECASE | re.DOTALL)
    if title_match:
        return str(title_match.group(2) or "").strip()
    first_string = re.search(r"""(['"])(.*?)\1""", block, flags=re.DOTALL)
    if first_string:
        return str(first_string.group(2) or "").strip()
    return ""


def _extract_pine_library_functions(source_text: str) -> list[str]:
    """
    Extract function names declared in a Pine library file.

    Supports common forms:
    - `export foo(args) =>`
    - `foo(args) =>`
    """
    text = str(source_text or "")
    if not text.strip():
        return []
    names = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        m = re.match(
            r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*=>",
            stripped,
            flags=re.IGNORECASE,
        )
        if m:
            names.add(str(m.group(1) or "").strip())
    return sorted([n for n in names if n])


def _extract_alias_function_calls(source_text: str) -> dict[str, list[str]]:
    """
    Extract calls of the form `Alias.func(...)` from Pine source.
    """
    text = str(source_text or "")
    calls: dict[str, set[str]] = {}
    if not text.strip():
        return {}
    for alias, func in re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        text,
        flags=re.IGNORECASE,
    ):
        alias_key = str(alias).strip()
        func_name = str(func).strip()
        if not alias_key or not func_name:
            continue
        calls.setdefault(alias_key, set()).add(func_name)
    return {k: sorted(list(v)) for k, v in calls.items()}


# ---------------------------------------------------------------------------
# Library analysis & import resolution
# ---------------------------------------------------------------------------

def _analyze_provided_libraries(provided_library_files: list[dict] | None):
    """
    Build lightweight metadata used to match Pine imports with uploaded libraries.
    """
    out = []
    for item in list(provided_library_files or []):
        if not isinstance(item, dict):
            continue
        source_name = str(item.get("source_name") or "").strip()
        path = str(item.get("path") or "").strip()
        stem = os.path.splitext(os.path.basename(source_name or path))[0]
        normalized_candidates = set()
        if stem:
            normalized_candidates.add(_normalize_token(stem))
        declared_name = ""
        declared_functions: list[str] = []
        if path and os.path.exists(path):
            try:
                txt, _ = _read_text_file_with_fallback(path)
                declared_name = _extract_library_decl_name(txt)
                declared_functions = _extract_pine_library_functions(txt)
            except Exception:
                declared_name = ""
                declared_functions = []
        if declared_name:
            normalized_candidates.add(_normalize_token(declared_name))
        out.append(
            {
                "source_name": source_name,
                "path": path,
                "source_sha1": str(item.get("source_sha1") or ""),
                "declared_name": declared_name,
                "declared_functions": declared_functions,
                "normalized_candidates": sorted([c for c in normalized_candidates if c]),
            }
        )
    return out


def _validate_python_mapping_target(target: str):
    """
    Validate mapping target:
    - `/abs/path/module.py` or `relative/path/module.py`
    - dotted module path (`package.module`) importable in current env
    """
    value = str(target or "").strip()
    if not value:
        return False, "mapping vide"
    # File path mode.
    if value.endswith(".py") or "/" in value or "\\" in value:
        candidate = value
        if not os.path.isabs(candidate):
            candidate = os.path.join(_repo_root_dir(), candidate)
        if os.path.exists(candidate):
            return True, f"path:{os.path.abspath(candidate)}"
        return False, f"fichier introuvable: {candidate}"
    # Dotted module mode.
    try:
        spec = importlib.util.find_spec(value)
        if spec is not None:
            return True, f"module:{value}"
    except Exception:
        pass
    return False, f"module non importable: {value}"


def _resolve_pine_imports(
    import_lines: list[str] | None,
    source_text: str | None = None,
    provided_library_files: list[dict] | None = None,
    import_mapping: dict | None = None,
):
    """
    Resolve Pine imports against uploaded library files + explicit Python mapping.
    """
    parsed_imports = _parse_pine_import_lines(import_lines)
    libraries = _analyze_provided_libraries(provided_library_files)
    mapping_dict = import_mapping if isinstance(import_mapping, dict) else {}
    alias_calls = _extract_alias_function_calls(str(source_text or ""))
    resolution = []
    resolved_count = 0
    functions_called_total = 0
    functions_missing_total = 0

    for item in parsed_imports:
        module_ref = item.get("module_ref")
        alias = item.get("alias")
        raw = item.get("raw")
        parse_ok = bool(item.get("parse_ok"))
        module_tail = _normalize_token(item.get("module_tail") or "")
        alias_norm = _normalize_token(alias or "")

        matched_lib = None
        for lib in libraries:
            candidates = set(lib.get("normalized_candidates") or [])
            if module_tail and module_tail in candidates:
                matched_lib = lib
                break
            if alias_norm and alias_norm in candidates:
                matched_lib = lib
                break

        mapping_target = ""
        if alias and alias in mapping_dict:
            mapping_target = str(mapping_dict.get(alias) or "").strip()
        elif module_ref and module_ref in mapping_dict:
            mapping_target = str(mapping_dict.get(module_ref) or "").strip()
        elif alias_norm and alias_norm in mapping_dict:
            mapping_target = str(mapping_dict.get(alias_norm) or "").strip()

        mapping_valid = False
        mapping_detail = ""
        if mapping_target:
            mapping_valid, mapping_detail = _validate_python_mapping_target(mapping_target)

        called_functions = alias_calls.get(str(alias or ""), []) if alias else []
        library_functions = []
        if isinstance(matched_lib, dict):
            library_functions = list(matched_lib.get("declared_functions") or [])
        library_function_set = set(str(x) for x in library_functions)
        missing_functions = [f for f in called_functions if str(f) not in library_function_set]
        found_functions_count = len([f for f in called_functions if str(f) in library_function_set])
        functions_called_total += len(called_functions)
        functions_missing_total += len(missing_functions)

        functions_ok = len(missing_functions) == 0
        resolved = parse_ok and bool(matched_lib) and bool(mapping_target) and bool(mapping_valid) and functions_ok
        if resolved:
            resolved_count += 1

        resolution.append(
            {
                "raw": raw,
                "module_ref": module_ref,
                "alias": alias,
                "parse_ok": parse_ok,
                "library_file_found": bool(matched_lib),
                "library_source_name": matched_lib.get("source_name") if isinstance(matched_lib, dict) else None,
                "library_path": matched_lib.get("path") if isinstance(matched_lib, dict) else None,
                "library_functions_count": len(library_functions),
                "called_functions": called_functions,
                "called_functions_count": len(called_functions),
                "found_functions_count": int(found_functions_count),
                "missing_functions": missing_functions,
                "missing_functions_count": len(missing_functions),
                "python_mapping_target": mapping_target or None,
                "python_mapping_valid": bool(mapping_valid),
                "python_mapping_detail": mapping_detail or None,
                "resolved": bool(resolved),
            }
        )

    return {
        "imports": resolution,
        "import_count": len(parsed_imports),
        "resolved_count": int(resolved_count),
        "unresolved_count": int(max(0, len(parsed_imports) - resolved_count)),
        "functions_called_total": int(functions_called_total),
        "functions_missing_total": int(functions_missing_total),
    }


# ---------------------------------------------------------------------------
# Pre-check & compatibility
# ---------------------------------------------------------------------------

def _precheck_pine_script_text(
    source_text: str,
    source_name: str = "",
    provided_library_files: list[dict] | None = None,
    import_mapping: dict | None = None,
):
    """Run a lightweight Pine pre-check to provide immediate actionable feedback."""
    text = str(source_text or "")
    errors = []
    warnings = []
    info = []

    line_count = text.count("\n") + (1 if text else 0)
    char_count = len(text)
    if not text.strip():
        errors.append("Le fichier est vide.")

    version_match = re.search(r"//@version\s*=\s*(\d+)", text, flags=re.IGNORECASE)
    detected_version = int(version_match.group(1)) if version_match else None
    if detected_version is None:
        errors.append("Directive `//@version=...` manquante.")
    elif detected_version != 6:
        warnings.append(
            f"Version Pine d\u00e9tect\u00e9e: v{detected_version}. WFOE V3 cible prioritairement Pine v6."
        )

    has_strategy_decl = bool(re.search(r"^\s*strategy\s*\(", text, flags=re.IGNORECASE | re.MULTILINE))
    has_indicator_decl = bool(re.search(r"^\s*indicator\s*\(", text, flags=re.IGNORECASE | re.MULTILINE))
    if not has_strategy_decl:
        errors.append("D\u00e9claration `strategy(...)` introuvable.")
    if has_indicator_decl and not has_strategy_decl:
        warnings.append("Le script semble \u00eatre un `indicator`, pas une `strategy` backtestable.")

    opens = text.count("(")
    closes = text.count(")")
    if opens != closes:
        warnings.append(
            f"D\u00e9s\u00e9quilibre de parenth\u00e8ses d\u00e9tect\u00e9: ouvrantes={opens}, fermantes={closes}."
        )

    import_lines = re.findall(r"^\s*import\s+.+$", text, flags=re.IGNORECASE | re.MULTILINE)
    provided_library_files = provided_library_files if isinstance(provided_library_files, list) else []
    import_mapping = import_mapping if isinstance(import_mapping, dict) else {}
    provided_library_names = [
        str(item.get("source_name") or "").strip()
        for item in provided_library_files
        if isinstance(item, dict)
    ]
    provided_library_names = [name for name in provided_library_names if name]
    provided_libraries_count = len(provided_library_names)
    import_resolution = _resolve_pine_imports(
        import_lines=import_lines,
        source_text=text,
        provided_library_files=provided_library_files,
        import_mapping=import_mapping,
    )
    resolved_import_count = int(import_resolution.get("resolved_count", 0))
    unresolved_import_count = int(import_resolution.get("unresolved_count", 0))
    functions_called_total = int(import_resolution.get("functions_called_total", 0))
    functions_missing_total = int(import_resolution.get("functions_missing_total", 0))
    if import_lines:
        if provided_libraries_count == 0:
            warnings.append(
                "Des imports Pine externes sont pr\u00e9sents. Aucun fichier de librairie n'est fourni: "
                "ajoute les fichiers .txt/.pine des librairies avec la strat\u00e9gie."
            )
        elif unresolved_import_count > 0:
            warnings.append(
                f"Des imports Pine externes sont pr\u00e9sents ({len(import_lines)} import(s)) et "
                f"{provided_libraries_count} fichier(s) de librairie ont \u00e9t\u00e9 fournis. "
                f"R\u00e9solution incompl\u00e8te: {resolved_import_count}/{len(import_lines)} import(s) mapp\u00e9s. "
                f"Fonctions manquantes: {functions_missing_total}/{functions_called_total}."
            )
        else:
            info.append(
                f"Imports Pine externes r\u00e9solus: {resolved_import_count}/{len(import_lines)} "
                f"(fichiers librairie + mapping Python). Fonctions v\u00e9rifi\u00e9es: "
                f"{functions_called_total - functions_missing_total}/{functions_called_total}."
            )

    detected_features = {
        "uses_request_security": bool(re.search(r"\brequest\.security\s*\(", text, flags=re.IGNORECASE)),
        "uses_request_security_lower_tf": bool(
            re.search(r"\brequest\.security_lower_tf\s*\(", text, flags=re.IGNORECASE)
        ),
        "uses_strategy_entry": bool(re.search(r"\bstrategy\.entry\s*\(", text, flags=re.IGNORECASE)),
        "uses_strategy_exit": bool(re.search(r"\bstrategy\.exit\s*\(", text, flags=re.IGNORECASE)),
        "uses_strategy_close": bool(re.search(r"\bstrategy\.close\s*\(", text, flags=re.IGNORECASE)),
        "uses_strategy_cancel": bool(re.search(r"\bstrategy\.cancel\s*\(", text, flags=re.IGNORECASE)),
        "uses_strategy_order": bool(re.search(r"\bstrategy\.order\s*\(", text, flags=re.IGNORECASE)),
        "uses_short_entry": bool(
            re.search(
                r"\bstrategy\.(?:entry|order)\s*\([^)]*strategy\.short",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ),
        "uses_order_price_controls": bool(
            re.search(
                r"\bstrategy\.(?:entry|order|exit)\s*\([^)]*\b(?:limit|stop|trail_price|trail_offset|trail_points)\s*=",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ),
        "uses_order_qty_controls": bool(
            re.search(
                r"\bstrategy\.(?:entry|order|exit)\s*\([^)]*\b(?:qty|qty_percent)\s*=",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ),
        "uses_pyramiding": bool(re.search(r"\bpyramiding\s*=", text, flags=re.IGNORECASE)),
        "uses_loops": bool(re.search(r"^\s*(for|while)\b", text, flags=re.IGNORECASE | re.MULTILINE)),
        "uses_switch": bool(re.search(r"^\s*switch\b", text, flags=re.IGNORECASE | re.MULTILINE)),
        "import_count": len(import_lines),
        "resolved_import_count": resolved_import_count,
        "unresolved_import_count": unresolved_import_count,
        "external_functions_called_count": functions_called_total,
        "external_functions_missing_count": functions_missing_total,
    }
    if detected_features["uses_request_security_lower_tf"]:
        warnings.append(
            "`request.security_lower_tf` d\u00e9tect\u00e9: support pr\u00e9vu en mode limit\u00e9, parit\u00e9 \u00e0 v\u00e9rifier."
        )

    source_sha1 = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    status = "valid" if not errors else "invalid"
    if status == "valid":
        info.append("Pr\u00e9-analyse OK: script exploitable pour la phase suivante de compatibilit\u00e9.")

    return _sanitize_for_json(
        {
            "status": status,
            "source_name": source_name or None,
            "line_count": line_count,
            "char_count": char_count,
            "detected_version": detected_version,
            "has_strategy_declaration": has_strategy_decl,
            "has_indicator_declaration": has_indicator_decl,
            "detected_features": detected_features,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "source_sha1": source_sha1,
            "import_lines": import_lines,
            "import_resolution": import_resolution.get("imports", []),
            "provided_libraries_count": provided_libraries_count,
            "provided_libraries_names": provided_library_names,
            "resolved_import_count": resolved_import_count,
            "unresolved_import_count": unresolved_import_count,
            "external_functions_called_count": functions_called_total,
            "external_functions_missing_count": functions_missing_total,
            "pine_import_mapping": import_mapping,
        }
    )


def _precheck_pine_script_file(
    file_path: str,
    provided_library_files: list[dict] | None = None,
    import_mapping: dict | None = None,
):
    """Read a Pine file from disk and return a pre-check report."""
    path = str(file_path or "").strip()
    if not path:
        return None
    if not os.path.exists(path):
        return {
            "status": "invalid",
            "source_name": os.path.basename(path) if path else None,
            "errors": [f"Fichier introuvable: {path}"],
            "warnings": [],
            "info": [],
        }
    try:
        with open(path, "rb") as f:
            raw = f.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
        return _precheck_pine_script_text(
            text,
            source_name=os.path.basename(path),
            provided_library_files=provided_library_files,
            import_mapping=import_mapping,
        )
    except Exception as e:
        return {
            "status": "invalid",
            "source_name": os.path.basename(path),
            "errors": [f"Erreur de lecture: {e}"],
            "warnings": [],
            "info": [],
        }


def _build_pine_compatibility_report(precheck_report: dict, compat_mode: str = "strict"):
    """Build Pine compatibility report aligned with Lot P0.3 (S0/S1/S2/S3)."""
    if not isinstance(precheck_report, dict) or not precheck_report:
        return None

    mode = str(compat_mode or "strict").lower()
    if mode not in ("strict", "assist", "manual"):
        mode = "strict"

    detected = precheck_report.get("detected_features", {}) or {}
    import_count = int(detected.get("import_count", 0) or 0)
    resolved_import_count = int(precheck_report.get("resolved_import_count", 0) or 0)
    unresolved_import_count = int(precheck_report.get("unresolved_import_count", 0) or 0)
    external_functions_called_count = int(
        precheck_report.get("external_functions_called_count", 0) or 0
    )
    external_functions_missing_count = int(
        precheck_report.get("external_functions_missing_count", 0) or 0
    )
    provided_libraries_count = int(precheck_report.get("provided_libraries_count", 0) or 0)
    uses_security = bool(detected.get("uses_request_security", False))
    uses_lower_tf = bool(detected.get("uses_request_security_lower_tf", False))
    uses_strategy_order = bool(detected.get("uses_strategy_order", False))
    uses_short_entry = bool(detected.get("uses_short_entry", False))
    uses_order_price_controls = bool(detected.get("uses_order_price_controls", False))
    uses_order_qty_controls = bool(detected.get("uses_order_qty_controls", False))
    uses_pyramiding = bool(detected.get("uses_pyramiding", False))
    uses_loops = bool(detected.get("uses_loops", False))
    uses_switch = bool(detected.get("uses_switch", False))
    has_strategy_decl = bool(precheck_report.get("has_strategy_declaration", False))
    version = precheck_report.get("detected_version")
    pre_errors = list(precheck_report.get("errors", []) or [])
    pre_warnings = list(precheck_report.get("warnings", []) or [])

    items = []

    def _add_item(code, label, target_level, status, detected_flag, detail="", blocking=False):
        items.append(
            {
                "code": code,
                "label": label,
                "target_level": target_level,
                "status": status,
                "detected": bool(detected_flag),
                "blocking": bool(blocking),
                "detail": str(detail or ""),
            }
        )

    # Baseline structure checks.
    if has_strategy_decl:
        _add_item("strategy_decl", "D\u00e9claration strategy(...)", "S3", "supported", True)
    else:
        _add_item(
            "strategy_decl",
            "D\u00e9claration strategy(...)",
            "S3",
            "blocked",
            False,
            detail="Le script n'expose pas de strat\u00e9gie backtestable.",
            blocking=True,
        )

    if version == 6:
        _add_item("pine_version", "Directive //@version=6", "S2", "supported", True)
    elif version is None:
        _add_item(
            "pine_version",
            "Directive //@version=6",
            "S2",
            "blocked",
            False,
            detail="Directive de version absente.",
            blocking=True,
        )
    else:
        _add_item(
            "pine_version",
            "Directive //@version=6",
            "S2",
            "partial",
            True,
            detail=f"Version d\u00e9tect\u00e9e v{version}, adaptation potentiellement n\u00e9cessaire.",
        )

    # Feature-level compatibility.
    if import_count > 0:
        detail = (
            f"{import_count} import(s) externe(s) d\u00e9tect\u00e9(s) | "
            f"r\u00e9solus={resolved_import_count}, non r\u00e9solus={unresolved_import_count}."
        )
        if external_functions_called_count > 0:
            detail += (
                f" Fonctions externes appel\u00e9es={external_functions_called_count}, "
                f"manquantes={external_functions_missing_count}."
            )
        if provided_libraries_count > 0:
            detail += f" {provided_libraries_count} fichier(s) de librairie fourni(s)."
        if unresolved_import_count == 0 and resolved_import_count == import_count:
            _add_item(
                "external_imports",
                "Imports Pine externes",
                "S1",
                "partial",
                True,
                detail=detail + " Mapping fourni (phase assist\u00e9e), validation de parit\u00e9 encore requise.",
                blocking=False,
            )
        else:
            _add_item(
                "external_imports",
                "Imports Pine externes",
                "S0",
                "blocked",
                True,
                detail=detail + " Mapping Python complet requis en mode strict.",
                blocking=True,
            )
    else:
        _add_item("external_imports", "Imports Pine externes", "S0", "not_applicable", False)

    if uses_security:
        _add_item(
            "request_security",
            "request.security",
            "S1",
            "partial",
            True,
            detail="Support MTF partiel pr\u00e9vu (lookahead/politiques \u00e0 valider).",
        )
    else:
        _add_item("request_security", "request.security", "S1", "not_applicable", False)

    if uses_lower_tf:
        _add_item(
            "request_security_lower_tf",
            "request.security_lower_tf",
            "S0",
            "blocked",
            True,
            detail="Support bas timeframe non garanti au stade actuel.",
            blocking=True,
        )
    else:
        _add_item("request_security_lower_tf", "request.security_lower_tf", "S0", "not_applicable", False)

    if uses_strategy_order:
        _add_item(
            "strategy_order",
            "strategy.order",
            "S1",
            "partial",
            True,
            detail=(
                "Support runtime partiel: les directions long/short sont interpr\u00e9t\u00e9es comme signaux "
                "d'entr\u00e9e; les nuances avanc\u00e9es d'ordres Pine restent \u00e0 valider via parit\u00e9."
            ),
            blocking=False,
        )
    else:
        _add_item("strategy_order", "strategy.order", "S0", "not_applicable", False)

    if uses_short_entry:
        _add_item(
            "short_entries",
            "Entr\u00e9es short",
            "S1",
            "partial",
            True,
            detail="Support short ajout\u00e9 (signaux short_entries/short_exits), parit\u00e9 recommand\u00e9e avant production.",
            blocking=False,
        )
    else:
        _add_item("short_entries", "Entr\u00e9es short", "S0", "not_applicable", False)

    if uses_order_price_controls:
        _add_item(
            "order_price_controls",
            "Ordres limit/stop/trailing",
            "S0",
            "blocked",
            True,
            detail=(
                "Les param\u00e8tres d'ordres pending (`limit/stop/trail_*`) ne sont pas reproduits "
                "fid\u00e8lement par le runtime beta."
            ),
            blocking=True,
        )
    else:
        _add_item("order_price_controls", "Ordres limit/stop/trailing", "S0", "not_applicable", False)

    if uses_order_qty_controls:
        _add_item(
            "order_qty_controls",
            "qty / qty_percent",
            "S1",
            "partial",
            True,
            detail=(
                "D\u00e9tection de sizing Pine explicite: support beta partiel, "
                "validation parit\u00e9 recommand\u00e9e."
            ),
            blocking=False,
        )
    else:
        _add_item("order_qty_controls", "qty / qty_percent", "S1", "not_applicable", False)

    if uses_pyramiding:
        _add_item(
            "pyramiding",
            "Pyramiding",
            "S0",
            "blocked",
            True,
            detail="Le runtime V3 beta ne reproduit pas encore le pyramiding Pine.",
            blocking=True,
        )
    else:
        _add_item("pyramiding", "Pyramiding", "S0", "not_applicable", False)

    if uses_loops or uses_switch:
        labels = []
        if uses_loops:
            labels.append("boucles")
        if uses_switch:
            labels.append("switch")
        _add_item(
            "control_flow_advanced",
            "Contr\u00f4le de flux avanc\u00e9",
            "S0",
            "blocked",
            True,
            detail=f"Transpilation V3 beta partielle: {', '.join(labels)} non valid\u00e9s.",
            blocking=True,
        )
    else:
        _add_item("control_flow_advanced", "Contr\u00f4le de flux avanc\u00e9", "S0", "not_applicable", False)

    for code, label, flag in [
        ("strategy_entry", "strategy.entry", bool(detected.get("uses_strategy_entry", False))),
        ("strategy_exit", "strategy.exit", bool(detected.get("uses_strategy_exit", False))),
        ("strategy_close", "strategy.close", bool(detected.get("uses_strategy_close", False))),
        ("strategy_cancel", "strategy.cancel", bool(detected.get("uses_strategy_cancel", False))),
    ]:
        if flag:
            _add_item(code, label, "S2", "supported", True)
        else:
            _add_item(code, label, "S2", "not_applicable", False)

    if pre_errors:
        _add_item(
            "precheck_errors",
            "Erreurs de pr\u00e9-analyse",
            "S0",
            "blocked",
            True,
            detail=f"{len(pre_errors)} erreur(s): " + " | ".join(str(e) for e in pre_errors[:3]),
            blocking=True,
        )
    else:
        _add_item("precheck_errors", "Erreurs de pr\u00e9-analyse", "S0", "supported", False)

    # Score model (deterministic and transparent).
    score_map = {
        "supported": 1.0,
        "partial": 0.5,
        "blocked": 0.0,
        "not_applicable": None,
    }
    scored = [score_map[it["status"]] for it in items if score_map.get(it["status"]) is not None]
    compatibility_score = round((sum(scored) / len(scored)) * 100, 1) if scored else 0.0

    blocking_items = [it for it in items if bool(it.get("blocking", False))]
    has_blocking_features = len(blocking_items) > 0
    is_blocking = mode == "strict" and has_blocking_features

    status = "compatible"
    if has_blocking_features:
        status = "incompatible_strict" if mode == "strict" else "incompatible_non_blocking"
    elif any(it.get("status") == "partial" for it in items):
        status = "compatible_with_warnings"

    recommendations = []
    if import_count > 0:
        if provided_libraries_count == 0:
            recommendations.append(
                "Importer les fichiers de librairie Pine (.txt/.pine) en m\u00eame temps que la strat\u00e9gie."
            )
        if unresolved_import_count > 0:
            recommendations.append(
                "Compl\u00e9ter le mapping des imports Pine vers des modules Python locaux (100% requis en strict)."
            )
        else:
            recommendations.append(
                "Mapping imports compl\u00e9t\u00e9: lancer un test de parit\u00e9 Pine/Python avant usage production."
            )
    if uses_lower_tf:
        recommendations.append(
            "Remplacer ou simplifier request.security_lower_tf pour r\u00e9duire le risque de non-parit\u00e9."
        )
    if uses_strategy_order:
        recommendations.append(
            "Valider parit\u00e9 Pine/Python sur les r\u00e8gles `strategy.order` (cas simples support\u00e9s, cas avanc\u00e9s \u00e0 v\u00e9rifier)."
        )
    if uses_short_entry:
        recommendations.append(
            "Contr\u00f4ler la parit\u00e9 des signaux short (entry/close) sur un \u00e9chantillon de r\u00e9f\u00e9rence avant run long."
        )
    if uses_order_price_controls:
        recommendations.append(
            "Retirer `limit/stop/trail_*` des ordres Pine ou fournir une version strat\u00e9gie 'market-only' pour la V3 beta."
        )
    if uses_order_qty_controls:
        recommendations.append(
            "Comparer la taille de position/trades Pine vs Python pour valider l'impact de `qty/qty_percent`."
        )
    if uses_pyramiding:
        recommendations.append(
            "D\u00e9sactiver le pyramiding pour la phase beta (requis pour fiabilit\u00e9 du replay)."
        )
    if uses_loops or uses_switch:
        recommendations.append(
            "\u00c9viter `for/while/switch` dans la strat\u00e9gie Pine cible beta ou fournir une strat\u00e9gie simplifi\u00e9e."
        )
    if pre_errors:
        recommendations.append(
            "Corriger d'abord les erreurs de pr\u00e9-analyse (`//@version`, `strategy(...)`, syntaxe)."
        )
    if mode == "strict" and has_blocking_features:
        recommendations.append(
            "Le mode strict bloque ce script. Utiliser `assist` uniquement pour diagnostic exploratoire."
        )
    if not recommendations:
        recommendations.append("Script pr\u00eat pour l'\u00e9tape suivante de compilation `strategy_spec.v1`.")

    return _sanitize_for_json(
        {
            "schema_version": "pine_compatibility.v1",
            "generated_at_utc": _utc_now_iso(),
            "source_name": precheck_report.get("source_name"),
            "compat_mode": mode,
            "status": status,
            "compatibility_score": compatibility_score,
            "has_blocking_features": has_blocking_features,
            "is_blocking": is_blocking,
            "items": items,
            "blocking_items": blocking_items,
            "import_resolution": precheck_report.get("import_resolution", []),
            "external_functions_called_count": external_functions_called_count,
            "external_functions_missing_count": external_functions_missing_count,
            "precheck_warnings": pre_warnings,
            "precheck_errors": pre_errors,
            "recommendations": recommendations,
        }
    )


# ---------------------------------------------------------------------------
# Beta readiness & order semantics
# ---------------------------------------------------------------------------

def _build_pine_beta_readiness_report(
    precheck_report: dict | None,
    compatibility_report: dict | None,
    strategy_spec: dict | None,
    strategy_spec_validation: dict | None,
    codegen_report: dict | None,
    generated_module_path: str | None = None,
):
    """Compute a deterministic beta-readiness gate for Pine V3 execution."""
    pre = precheck_report if isinstance(precheck_report, dict) else {}
    compat = compatibility_report if isinstance(compatibility_report, dict) else {}
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    spec_val = strategy_spec_validation if isinstance(strategy_spec_validation, dict) else {}
    codegen = codegen_report if isinstance(codegen_report, dict) else {}
    module_path = str(generated_module_path or "").strip()

    logic = spec.get("logic") if isinstance(spec.get("logic"), dict) else {}
    order_rules = logic.get("order_rules") if isinstance(logic.get("order_rules"), list) else []
    assignments = logic.get("assignments") if isinstance(logic.get("assignments"), list) else []

    runtime_blockers = []
    for item in (compat.get("items") or []):
        if not isinstance(item, dict):
            continue
        if bool(item.get("blocking", False)):
            runtime_blockers.append(
                {
                    "code": item.get("code"),
                    "label": item.get("label"),
                    "detail": item.get("detail"),
                }
            )

    checks = [
        {
            "id": "precheck_valid",
            "label": "Pr\u00e9-analyse Pine valide",
            "required": True,
            "passed": str(pre.get("status", "")).lower() == "valid",
            "detail": None,
        },
        {
            "id": "compatibility_non_blocking",
            "label": "Compatibilit\u00e9 non bloquante",
            "required": True,
            "passed": not bool(compat.get("is_blocking", False)),
            "detail": f"status={compat.get('status')}",
        },
        {
            "id": "strategy_spec_valid",
            "label": "strategy_spec.v1 valide",
            "required": True,
            "passed": bool(spec_val.get("valid", False)),
            "detail": f"errors={len(spec_val.get('errors', []) or [])}",
        },
        {
            "id": "logic_order_rules",
            "label": "R\u00e8gles d'ordres extraites",
            "required": True,
            "passed": len(order_rules) > 0,
            "detail": f"order_rules={len(order_rules)}, assignments={len(assignments)}",
        },
        {
            "id": "codegen_ok",
            "label": "Codegen module Python",
            "required": True,
            "passed": str(codegen.get("status", "")).lower() == "ok",
            "detail": f"status={codegen.get('status')}",
        },
        {
            "id": "generated_module_exists",
            "label": "Module g\u00e9n\u00e9r\u00e9 disponible",
            "required": True,
            "passed": bool(module_path and os.path.exists(module_path)),
            "detail": module_path or None,
        },
        {
            "id": "runtime_blockers_absent",
            "label": "Aucun blocker runtime beta",
            "required": True,
            "passed": len(runtime_blockers) == 0,
            "detail": f"blockers={len(runtime_blockers)}",
        },
    ]

    required_checks = [c for c in checks if bool(c.get("required", False))]
    required_passed = [c for c in required_checks if bool(c.get("passed", False))]
    readiness_score = round((len(required_passed) / max(1, len(required_checks))) * 100.0, 1)
    beta_ready = len(required_passed) == len(required_checks)

    next_actions = []
    for check in required_checks:
        if not bool(check.get("passed", False)):
            fallback = "\u00e0 corriger"
            next_actions.append(f"{check.get('label')}: {check.get('detail') or fallback}")
    if not next_actions and runtime_blockers:
        next_actions.extend(
            [f"{b.get('label')}: {b.get('detail')}" for b in runtime_blockers if isinstance(b, dict)]
        )

    return _sanitize_for_json(
        {
            "schema_version": "pine_beta_readiness.v1",
            "generated_at_utc": _utc_now_iso(),
            "beta_ready": bool(beta_ready),
            "status": "ready" if beta_ready else "not_ready",
            "readiness_score": readiness_score,
            "checks": checks,
            "runtime_blockers": runtime_blockers,
            "next_actions": next_actions,
            "strategy_id": (spec.get("strategy") or {}).get("id") if isinstance(spec, dict) else None,
            "compatibility_status": compat.get("status"),
            "compatibility_score": compat.get("compatibility_score"),
        }
    )


def _compute_pine_order_semantics_report(
    strategy_spec: dict | None = None,
    compat_mode: str | None = None,
    enforce_order_semantics: bool | None = None,
) -> dict:
    """Compute and sanitize runtime order-semantics diagnostics for Pine V3."""
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    if not spec:
        return {}
    runtime_cfg = {
        "pine_compat_mode": str(
            compat_mode
            if isinstance(compat_mode, str) and compat_mode.strip()
            else st.session_state.get("pine_compat_mode", "strict")
        ),
        "pine_enforce_order_semantics": bool(
            st.session_state.get("pine_enforce_order_semantics", True)
            if enforce_order_semantics is None
            else enforce_order_semantics
        ),
    }
    report = _build_pine_order_semantics_report(
        strategy_spec=spec,
        runtime_config=runtime_cfg,
    )
    return _sanitize_for_json(report if isinstance(report, dict) else {})


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------

def _get_pine_parity_thresholds_from_state() -> dict:
    """Return parity thresholds from session state with safe defaults."""
    thresholds = dict(_DEFAULT_PARITY_THRESHOLDS)
    for key, default_value in _DEFAULT_PARITY_THRESHOLDS.items():
        state_key = f"pine_parity_{key}"
        raw_value = st.session_state.get(state_key, default_value)
        try:
            thresholds[key] = float(raw_value)
        except Exception:
            thresholds[key] = float(default_value)
    return thresholds


def _get_pine_parity_detail_thresholds_from_state() -> dict:
    """Return detailed parity thresholds (events/trades) from session state."""
    thresholds = dict(_DEFAULT_PARITY_DETAIL_THRESHOLDS)
    for key, default_value in _DEFAULT_PARITY_DETAIL_THRESHOLDS.items():
        state_key = f"pine_parity_{key}"
        raw_value = st.session_state.get(state_key, default_value)
        try:
            thresholds[key] = float(raw_value)
        except Exception:
            thresholds[key] = float(default_value)
    return thresholds


def _to_iso_utc(value) -> str | None:
    """Best-effort conversion to UTC ISO timestamp."""
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    try:
        return ts.to_pydatetime().isoformat()
    except Exception:
        return str(ts)


def _extract_current_events_and_trades_for_parity(max_items: int = 30000) -> dict:
    """Extract current runtime entry/exit events and trades for detailed parity."""
    max_rows = int(max(100, max_items))
    trades_df = pd.DataFrame()
    pf = st.session_state.get("final_portfolio")
    index_ref = None

    if pf is not None:
        try:
            trades_df = pd.DataFrame(pf.trades.records)
        except Exception:
            trades_df = pd.DataFrame()
        try:
            if hasattr(pf, "wrapper"):
                index_ref = pf.wrapper.index
        except Exception:
            index_ref = None

    if trades_df.empty and isinstance(st.session_state.get("final_trades_df"), pd.DataFrame):
        trades_df = st.session_state.get("final_trades_df").copy()
    if trades_df.empty:
        return {"entries": [], "exits": [], "trades": []}

    if len(trades_df) > max_rows:
        trades_df = trades_df.head(max_rows).copy()

    def _ts_from_row(row, kind: str) -> str | None:
        direct_cols = [
            f"{kind}_ts",
            f"{kind}_time",
            f"{kind}_timestamp",
        ]
        for col in direct_cols:
            if col in row and pd.notna(row.get(col)):
                iso = _to_iso_utc(row.get(col))
                if iso:
                    return iso

        idx_col = f"{kind}_idx"
        if idx_col in row and index_ref is not None and pd.notna(row.get(idx_col)):
            try:
                idx_val = int(row.get(idx_col))
                if 0 <= idx_val < len(index_ref):
                    return _to_iso_utc(index_ref[idx_val])
            except Exception:
                return None
        return None

    entries: list[str] = []
    exits: list[str] = []
    trades: list[dict] = []

    for _, row in trades_df.iterrows():
        entry_iso = _ts_from_row(row, "entry")
        exit_iso = _ts_from_row(row, "exit")
        if entry_iso:
            entries.append(entry_iso)
        if exit_iso:
            exits.append(exit_iso)
        if entry_iso and exit_iso:
            trade_row = {"entry_time": entry_iso, "exit_time": exit_iso}
            if "pnl" in row and pd.notna(row.get("pnl")):
                try:
                    trade_row["pnl"] = float(row.get("pnl"))
                except Exception:
                    pass
            trades.append(trade_row)

    entries = sorted(set(entries))
    exits = sorted(set(exits))
    return {"entries": entries, "exits": exits, "trades": trades}


def _extract_current_metrics_for_parity(
    _extract_final_backtest_for_expert=None,
    get_current_config=None,
) -> dict:
    """Extract current runtime metrics for parity checks from final portfolio.

    The optional callable parameters allow the caller (app.py) to inject
    functions that live in app.py without creating circular imports.
    """
    metrics = {}
    pf = st.session_state.get("final_portfolio")
    if pf is not None:
        try:
            n_trades = int(len(pf.trades))
            metrics["trade_count"] = n_trades
            metrics["entry_count"] = n_trades
            metrics["exit_count"] = n_trades
        except Exception:
            pass
        try:
            metrics["total_return_pct"] = float(_safe_float_scalar(getattr(pf, "total_return", np.nan) * 100))
        except Exception:
            pass
        try:
            metrics["max_drawdown_pct"] = float(_safe_float_scalar(getattr(pf, "max_drawdown", np.nan) * 100))
        except Exception:
            pass
        return _normalize_parity_metrics(metrics)

    # Fallback for historical ZIP without final portfolio object.
    if _extract_final_backtest_for_expert is not None and get_current_config is not None:
        final_summary = _extract_final_backtest_for_expert(
            st.session_state.get("wfo_results"),
            get_current_config(),
            df_source=st.session_state.get("df"),
        )
        if isinstance(final_summary, dict):
            if isinstance(final_summary.get("strategy_n_trades"), (int, float)):
                n_trades = float(final_summary.get("strategy_n_trades"))
                metrics["trade_count"] = n_trades
                metrics["entry_count"] = n_trades
                metrics["exit_count"] = n_trades
            if isinstance(final_summary.get("strategy_total_return_pct"), (int, float)):
                metrics["total_return_pct"] = float(final_summary.get("strategy_total_return_pct"))
            if isinstance(final_summary.get("strategy_max_drawdown_pct"), (int, float)):
                metrics["max_drawdown_pct"] = float(final_summary.get("strategy_max_drawdown_pct"))
    return _normalize_parity_metrics(metrics)


def _parse_parity_reference_payload_from_text(
    raw_text: str,
    allow_legacy: bool = True,
) -> tuple[dict, dict, str | None]:
    """Parse and validate parity reference JSON text to canonical v1 payload."""
    txt = str(raw_text or "").strip()
    if not txt:
        return {}, {}, None
    try:
        data = json.loads(txt)
    except Exception as e:
        return {}, {}, f"JSON invalide: {e}"

    validation = _validate_parity_reference_payload(data, allow_legacy=allow_legacy)
    normalized_payload = (
        validation.get("normalized_payload")
        if isinstance(validation, dict) and isinstance(validation.get("normalized_payload"), dict)
        else {}
    )
    if not bool((validation or {}).get("valid", False)):
        err_lines = (validation or {}).get("errors") or []
        if not isinstance(err_lines, list):
            err_lines = [str(err_lines)]
        error_message = "; ".join([str(x) for x in err_lines if str(x).strip()]) or "R\u00e9f\u00e9rence Pine invalide."
        return normalized_payload, validation, error_message
    return normalized_payload, validation, None


def _apply_parity_reference_payload(
    payload: dict,
    validation: dict | None = None,
    update_text: bool = False,
):
    """Persist canonical parity reference payload and derived session fields."""
    if not isinstance(payload, dict):
        return
    metrics = payload.get("reference_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    st.session_state["pine_parity_reference_payload"] = payload
    st.session_state["pine_parity_reference_metrics"] = metrics
    if bool(update_text):
        st.session_state["pine_parity_reference_text"] = json.dumps(payload, indent=2, ensure_ascii=False)
    if isinstance(validation, dict):
        st.session_state["pine_parity_reference_validation"] = validation


# ---------------------------------------------------------------------------
# Expert / artifact builders
# ---------------------------------------------------------------------------

def _build_pine_artifacts_summary_for_expert():
    """Build a compact Pine artifacts summary consumable by Expert diagnostics."""
    precheck = st.session_state.get("pine_precheck_report")
    compat = st.session_state.get("pine_compatibility_report")
    spec = st.session_state.get("pine_strategy_spec")
    spec_validation = st.session_state.get("pine_strategy_spec_validation")
    trace = st.session_state.get("pine_generation_trace")
    llm_migration_report = st.session_state.get("pine_llm_migration_report")
    codegen = st.session_state.get("pine_codegen_report")
    beta_readiness = st.session_state.get("pine_beta_readiness_report")
    execution_gate = st.session_state.get("pine_execution_gate_report")
    order_semantics = st.session_state.get("pine_order_semantics_report")
    parity_report = st.session_state.get("pine_parity_report")
    request_security_diagnostics = st.session_state.get("pine_request_security_diagnostics")
    mtf_parity_proof = st.session_state.get("pine_mtf_parity_proof_report")
    parity_reference = st.session_state.get("pine_parity_reference_metrics")
    parity_reference_payload = st.session_state.get("pine_parity_reference_payload")
    parity_reference_validation = st.session_state.get("pine_parity_reference_validation")
    generated_module_path = st.session_state.get("pine_generated_module_path")
    source_text = st.session_state.get("pine_source_text")
    library_files = st.session_state.get("pine_library_files")
    import_mapping = st.session_state.get("pine_import_mapping")
    if not isinstance(library_files, list):
        library_files = []
    if not isinstance(import_mapping, dict):
        import_mapping = {}

    if (
        not any(isinstance(x, dict) and x for x in (precheck, compat, spec, spec_validation, trace))
        and not isinstance(beta_readiness, dict)
        and not isinstance(execution_gate, dict)
        and not isinstance(order_semantics, dict)
        and not isinstance(parity_report, dict)
        and not isinstance(request_security_diagnostics, dict)
        and not isinstance(mtf_parity_proof, dict)
        and not isinstance(source_text, str)
        and not library_files
    ):
        return {}

    strategy_meta = (spec.get("strategy") or {}) if isinstance(spec, dict) else {}
    source_meta = (spec.get("source") or {}) if isinstance(spec, dict) else {}
    transcription_meta = (spec.get("transcription") or {}) if isinstance(spec, dict) else {}
    source_sha1 = None
    if isinstance(precheck, dict):
        source_sha1 = precheck.get("source_sha1")
    if not source_sha1 and isinstance(source_meta, dict):
        source_sha1 = source_meta.get("source_sha1")

    source_preview = ""
    if isinstance(source_text, str) and source_text.strip():
        source_preview = "\n".join(source_text.splitlines()[:60]).strip()

    return _sanitize_for_json(
        {
            "available": True,
            "strategy_id": strategy_meta.get("id"),
            "strategy_name": strategy_meta.get("name"),
            "source_name": st.session_state.get("pine_source_name") or source_meta.get("file_name"),
            "source_sha1": source_sha1,
            "source_encoding": st.session_state.get("pine_source_encoding"),
            "precheck_status": precheck.get("status") if isinstance(precheck, dict) else None,
            "compatibility_status": compat.get("status") if isinstance(compat, dict) else None,
            "compatibility_score": compat.get("compatibility_score") if isinstance(compat, dict) else None,
            "blocking_items_count": len((compat.get("blocking_items") or [])) if isinstance(compat, dict) else 0,
            "spec_schema_version": spec.get("schema_version") if isinstance(spec, dict) else None,
            "spec_valid": bool(spec_validation.get("valid", False)) if isinstance(spec_validation, dict) else None,
            "spec_errors_count": len((spec_validation.get("errors") or [])) if isinstance(spec_validation, dict) else 0,
            "spec_parser_requested": (
                transcription_meta.get("parser_backend_requested")
                if isinstance(transcription_meta, dict)
                else None
            ),
            "spec_parser_used": (
                transcription_meta.get("parser_backend_used")
                if isinstance(transcription_meta, dict)
                else None
            ),
            "spec_parser_fallback": (
                bool(transcription_meta.get("fallback_to_regex", False))
                if isinstance(transcription_meta, dict)
                else None
            ),
            "imports": list(spec.get("imports") or []) if isinstance(spec, dict) else [],
            "import_resolution_count": len((precheck.get("import_resolution") or []))
            if isinstance(precheck, dict)
            else 0,
            "external_functions_called_count": int(precheck.get("external_functions_called_count", 0))
            if isinstance(precheck, dict)
            else 0,
            "external_functions_missing_count": int(precheck.get("external_functions_missing_count", 0))
            if isinstance(precheck, dict)
            else 0,
            "libraries_count": len(library_files),
            "libraries_names": [str(item.get("source_name") or "") for item in library_files if isinstance(item, dict)],
            "import_mapping_count": len(import_mapping),
            "import_mapping_keys": sorted([str(k) for k in import_mapping.keys()]),
            "inputs_count": len(spec.get("inputs") or []) if isinstance(spec, dict) else 0,
            "generation_trace": trace if isinstance(trace, dict) else {},
            "llm_migration_status": (
                llm_migration_report.get("status")
                if isinstance(llm_migration_report, dict)
                else None
            ),
            "codegen_status": (codegen.get("status") if isinstance(codegen, dict) else None),
            "beta_ready": bool(beta_readiness.get("beta_ready")) if isinstance(beta_readiness, dict) else None,
            "beta_readiness_score": (
                float(beta_readiness.get("readiness_score", 0.0)) if isinstance(beta_readiness, dict) else None
            ),
            "execution_gate_status": execution_gate.get("status") if isinstance(execution_gate, dict) else None,
            "execution_gate_can_run": execution_gate.get("can_run") if isinstance(execution_gate, dict) else None,
            "execution_gate_blockers_count": len((execution_gate.get("blockers") or []))
            if isinstance(execution_gate, dict)
            else 0,
            "order_semantics_status": (
                order_semantics.get("status") if isinstance(order_semantics, dict) else None
            ),
            "order_semantics_passed": (
                order_semantics.get("passed")
                if isinstance(order_semantics, dict) and "passed" in order_semantics
                else None
            ),
            "order_semantics_blockers_count": len((order_semantics.get("blockers") or []))
            if isinstance(order_semantics, dict)
            else 0,
            "order_semantics_price_controls_count": (
                int(order_semantics.get("rules_with_price_controls", 0))
                if isinstance(order_semantics, dict)
                else 0
            ),
            "order_semantics_qty_controls_count": (
                int(order_semantics.get("rules_with_qty_controls", 0))
                if isinstance(order_semantics, dict)
                else 0
            ),
            "parity_status": parity_report.get("status") if isinstance(parity_report, dict) else None,
            "parity_pass": (
                parity_report.get("parity_pass")
                if isinstance(parity_report, dict) and "parity_pass" in parity_report
                else None
            ),
            "parity_checks_count": len((parity_report.get("checks") or [])) if isinstance(parity_report, dict) else 0,
            "parity_detail_available": bool(parity_report.get("detail_available", False))
            if isinstance(parity_report, dict)
            else None,
            "parity_detail_pass": (
                parity_report.get("detail_pass")
                if isinstance(parity_report, dict) and "detail_pass" in parity_report
                else None
            ),
            "parity_reference_schema_version": (
                parity_reference_payload.get("schema_version")
                if isinstance(parity_reference_payload, dict)
                else None
            ),
            "parity_reference_source": (
                parity_reference_payload.get("source")
                if isinstance(parity_reference_payload, dict)
                else {}
            ),
            "parity_reference_valid": (
                bool(parity_reference_validation.get("valid", False))
                if isinstance(parity_reference_validation, dict)
                else None
            ),
            "parity_reference_metrics_count": (
                len((parity_reference_payload.get("reference_metrics") or {}))
                if isinstance(parity_reference_payload, dict)
                else 0
            ),
            "request_security_diagnostics_status": (
                request_security_diagnostics.get("status")
                if isinstance(request_security_diagnostics, dict)
                else None
            ),
            "request_security_diagnostics_count": (
                int(request_security_diagnostics.get("request_security_count", 0))
                if isinstance(request_security_diagnostics, dict)
                else 0
            ),
            "mtf_parity_proof_status": (
                mtf_parity_proof.get("status") if isinstance(mtf_parity_proof, dict) else None
            ),
            "mtf_parity_proof_pass": (
                mtf_parity_proof.get("proof_pass")
                if isinstance(mtf_parity_proof, dict) and "proof_pass" in mtf_parity_proof
                else None
            ),
            "parity_reference_metrics": parity_reference if isinstance(parity_reference, dict) else {},
            "generated_module_path": generated_module_path if isinstance(generated_module_path, str) else None,
            "source_preview": source_preview,
        }
    )


def _resolve_pine_source_for_artifacts():
    """Resolve Pine source text and metadata from session state for ZIP artifacts."""
    source_text = ""
    source_encoding = str(st.session_state.get("pine_source_encoding") or "")
    source_name = str(st.session_state.get("pine_source_name") or "").strip()
    source_path = str(st.session_state.get("pine_file_path") or "").strip()

    if source_path and os.path.exists(source_path):
        try:
            source_text, source_encoding = _read_text_file_with_fallback(source_path)
        except Exception:
            source_text = ""
    elif isinstance(st.session_state.get("pine_source_text"), str):
        source_text = st.session_state.get("pine_source_text") or ""

    if not source_name and source_path:
        source_name = os.path.basename(source_path)
    if not source_name:
        source_name = "strategy_source.pine.txt"

    if source_text and not source_name.lower().endswith((".pine", ".txt")):
        source_name = f"{source_name}.txt"

    source_sha1 = hashlib.sha1(source_text.encode("utf-8", errors="ignore")).hexdigest() if source_text else None
    return _sanitize_for_json(
        {
            "text": source_text,
            "encoding": source_encoding or None,
            "source_name": source_name,
            "source_path": source_path or None,
            "source_sha1": source_sha1,
            "line_count": source_text.count("\n") + (1 if source_text else 0),
            "char_count": len(source_text),
        }
    )


def _resolve_pine_libraries_for_artifacts():
    """Resolve uploaded Pine libraries from session state for ZIP artifacts."""
    raw_manifest = st.session_state.get("pine_library_files")
    if not isinstance(raw_manifest, list) or len(raw_manifest) == 0:
        return []

    out = []
    for entry in raw_manifest:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        source_name = str(entry.get("source_name") or "").strip()
        if not path or not os.path.exists(path):
            continue
        try:
            text, encoding = _read_text_file_with_fallback(path)
        except Exception:
            continue
        if not source_name:
            source_name = os.path.basename(path)
        source_sha1 = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest() if text else None
        out.append(
            {
                "source_name": source_name,
                "path": path,
                "text": text,
                "encoding": encoding,
                "source_sha1": source_sha1,
                "line_count": text.count("\n") + (1 if text else 0),
                "char_count": len(text),
            }
        )
    return _sanitize_for_json(out)


def _build_pine_generation_trace(
    source_artifact: dict,
    library_artifacts: list[dict] | None,
    import_mapping: dict | None,
    codegen_report: dict | None,
    generated_module_path: str | None,
    precheck_report: dict | None,
    compatibility_report: dict | None,
    strategy_spec: dict | None,
    strategy_spec_validation: dict | None,
):
    """Build deterministic generation trace for Pine V3 artifacts."""
    pre = precheck_report if isinstance(precheck_report, dict) else {}
    compat = compatibility_report if isinstance(compatibility_report, dict) else {}
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    spec_val = strategy_spec_validation if isinstance(strategy_spec_validation, dict) else {}
    src = source_artifact if isinstance(source_artifact, dict) else {}
    libs = library_artifacts if isinstance(library_artifacts, list) else []
    mapping = import_mapping if isinstance(import_mapping, dict) else {}
    codegen = codegen_report if isinstance(codegen_report, dict) else {}
    generated_path = str(generated_module_path or "").strip()
    llm_report = st.session_state.get("pine_llm_migration_report")
    llm_report = llm_report if isinstance(llm_report, dict) else {}
    llm_trace = llm_report.get("trace") if isinstance(llm_report.get("trace"), dict) else {}
    catalog_entry = st.session_state.get("pine_catalog_last_entry")
    catalog_entry = catalog_entry if isinstance(catalog_entry, dict) else {}

    has_any = (
        bool(src.get("text"))
        or bool(pre)
        or bool(compat)
        or bool(spec)
        or bool(spec_val)
        or bool(libs)
        or bool(mapping)
        or bool(codegen)
        or bool(generated_path)
        or bool(llm_report)
    )
    if not has_any:
        return {}

    spec_strategy = spec.get("strategy") if isinstance(spec.get("strategy"), dict) else {}
    trace = {
        "schema_version": "pine_generation_trace.v1",
        "generated_at_utc": _utc_now_iso(),
        "pipeline_stage": "P0.6",
        "strategy_mode": st.session_state.get("strategy_mode"),
        "strategy_id": st.session_state.get("strategy_id") or spec_strategy.get("id"),
        "source": {
            "name": src.get("source_name"),
            "path": src.get("source_path"),
            "encoding": src.get("encoding"),
            "sha1": src.get("source_sha1") or pre.get("source_sha1"),
            "line_count": src.get("line_count"),
            "char_count": src.get("char_count"),
        },
        "libraries": {
            "count": len(libs),
            "names": [str(item.get("source_name")) for item in libs if isinstance(item, dict)],
            "sha1_list": [str(item.get("source_sha1")) for item in libs if isinstance(item, dict) and item.get("source_sha1")],
        },
        "import_mapping": {
            "count": len(mapping),
            "keys": sorted([str(k) for k in mapping.keys()]),
        },
        "codegen": {
            "status": codegen.get("status"),
            "output_path": codegen.get("output_path") or generated_path or None,
            "module_name": codegen.get("module_name"),
            "changed": codegen.get("changed"),
        },
        "precheck": {
            "status": pre.get("status"),
            "detected_version": pre.get("detected_version"),
            "error_count": len(pre.get("errors") or []),
            "warning_count": len(pre.get("warnings") or []),
        },
        "compatibility": {
            "status": compat.get("status"),
            "score": compat.get("compatibility_score"),
            "is_blocking": bool(compat.get("is_blocking", False)),
            "blocking_items_count": len(compat.get("blocking_items") or []),
        },
        "spec": {
            "schema_version": spec.get("schema_version"),
            "strategy_id": spec_strategy.get("id"),
            "strategy_name": spec_strategy.get("name"),
            "valid": bool(spec_val.get("valid", False)) if spec_val else None,
            "validation_error_count": len(spec_val.get("errors") or []) if spec_val else None,
            "sha256": _sha256_json(spec) if spec else None,
            "imports_count": len(spec.get("imports") or []) if spec else 0,
            "inputs_count": len(spec.get("inputs") or []) if spec else 0,
        },
        "llm_used": bool(llm_report),
        "llm_migration": {
            "status": llm_report.get("status"),
            "provider": llm_trace.get("provider"),
            "model": llm_trace.get("model"),
            "candidate_valid": llm_trace.get("candidate_valid"),
            "accepted_spec_sha256": llm_trace.get("accepted_spec_sha256"),
            "errors": llm_report.get("errors", []),
            "warnings": llm_report.get("warnings", []),
        },
        "catalog": {
            "entry_id": catalog_entry.get("entry_id"),
            "strategy_id": catalog_entry.get("strategy_id"),
            "updated_at_utc": catalog_entry.get("updated_at_utc"),
        },
    }
    return _sanitize_for_json(trace)


def _build_pine_artifacts_manifest(
    source_artifact: dict,
    library_artifacts: list[dict] | None,
    import_mapping: dict | None,
    codegen_report: dict | None,
    generated_module_path: str | None,
    precheck_report: dict | None,
    compatibility_report: dict | None,
    strategy_spec: dict | None,
    strategy_spec_validation: dict | None,
    generation_trace: dict | None,
    beta_readiness_report: dict | None = None,
    execution_gate_report: dict | None = None,
    order_semantics_report: dict | None = None,
    parity_report: dict | None = None,
    request_security_diagnostics: dict | None = None,
    mtf_parity_proof_report: dict | None = None,
    parity_reference_payload: dict | None = None,
    parity_reference_validation: dict | None = None,
):
    """Build export manifest describing Pine artifacts bundled in the ZIP."""
    src = source_artifact if isinstance(source_artifact, dict) else {}
    libs = library_artifacts if isinstance(library_artifacts, list) else []
    pre = precheck_report if isinstance(precheck_report, dict) else {}
    compat = compatibility_report if isinstance(compatibility_report, dict) else {}
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    spec_val = strategy_spec_validation if isinstance(strategy_spec_validation, dict) else {}
    trace = generation_trace if isinstance(generation_trace, dict) else {}
    beta = beta_readiness_report if isinstance(beta_readiness_report, dict) else {}
    gate = execution_gate_report if isinstance(execution_gate_report, dict) else {}
    order_semantics = order_semantics_report if isinstance(order_semantics_report, dict) else {}
    parity = parity_report if isinstance(parity_report, dict) else {}
    mtf_diag = request_security_diagnostics if isinstance(request_security_diagnostics, dict) else {}
    mtf_proof = mtf_parity_proof_report if isinstance(mtf_parity_proof_report, dict) else {}
    parity_ref_payload = parity_reference_payload if isinstance(parity_reference_payload, dict) else {}
    parity_ref_validation = parity_reference_validation if isinstance(parity_reference_validation, dict) else {}
    mapping = import_mapping if isinstance(import_mapping, dict) else {}
    codegen = codegen_report if isinstance(codegen_report, dict) else {}
    generated_path = str(generated_module_path or "").strip()

    has_any = (
        bool(src.get("text"))
        or bool(pre)
        or bool(compat)
        or bool(spec)
        or bool(spec_val)
            or bool(trace)
            or bool(gate)
            or bool(order_semantics)
            or bool(libs)
        or bool(mapping)
        or bool(codegen)
        or bool(generated_path)
        or bool(mtf_diag)
        or bool(mtf_proof)
    )
    if not has_any:
        return {}

    return _sanitize_for_json(
        {
            "schema_version": "pine_artifacts_manifest.v1",
            "generated_at_utc": _utc_now_iso(),
            "has_source": bool(src.get("text")),
            "libraries_count": len(libs),
            "library_names": [str(item.get("source_name")) for item in libs if isinstance(item, dict)],
            "import_mapping_count": len(mapping),
            "has_generated_strategy_module": bool(codegen.get("output_path") or generated_path),
            "has_precheck_report": bool(pre),
            "has_compatibility_report": bool(compat),
            "has_strategy_spec": bool(spec),
            "has_strategy_spec_validation": bool(spec_val),
            "has_generation_trace": bool(trace),
            "has_llm_migration_report": bool((trace.get("llm_used")) if isinstance(trace, dict) else False),
            "has_beta_readiness_report": bool(beta),
            "has_execution_gate_report": bool(gate),
            "has_order_semantics_report": bool(order_semantics),
            "has_parity_report": bool(parity),
            "has_request_security_diagnostics": bool(mtf_diag),
            "has_mtf_parity_proof_report": bool(mtf_proof),
            "has_parity_reference_payload": bool(parity_ref_payload),
            "has_parity_reference_validation": bool(parity_ref_validation),
            "source_name": src.get("source_name"),
            "source_sha1": src.get("source_sha1") or pre.get("source_sha1"),
            "compatibility_status": compat.get("status"),
            "strategy_spec_valid": bool(spec_val.get("valid", False)) if spec_val else None,
            "beta_ready": bool(beta.get("beta_ready")) if beta else None,
            "execution_gate_status": gate.get("status") if gate else None,
            "execution_gate_can_run": gate.get("can_run") if gate else None,
            "order_semantics_status": order_semantics.get("status") if order_semantics else None,
            "order_semantics_passed": order_semantics.get("passed") if order_semantics else None,
            "parity_pass": parity.get("parity_pass") if parity else None,
            "request_security_diagnostics_status": mtf_diag.get("status") if mtf_diag else None,
            "request_security_diagnostics_count": mtf_diag.get("request_security_count") if mtf_diag else None,
            "mtf_parity_proof_status": mtf_proof.get("status") if mtf_proof else None,
            "mtf_parity_proof_pass": mtf_proof.get("proof_pass") if mtf_proof else None,
            "parity_detail_available": bool(parity.get("detail_available", False)) if parity else None,
            "parity_detail_pass": parity.get("detail_pass") if parity else None,
            "parity_reference_schema_version": parity_ref_payload.get("schema_version") if parity_ref_payload else None,
            "parity_reference_valid": parity_ref_validation.get("valid") if parity_ref_validation else None,
            "artifact_files": {
                "strategy_source": "strategy_source.pine.txt",
                "libraries_manifest": "pine_libraries_manifest.json",
                "libraries_dir": "pine_libraries/",
                "import_mapping": "import_mapping.json",
                "compatibility_report": "compatibility_report.json",
                "strategy_spec": "strategy_spec.v1.json",
                "strategy_spec_validation": "strategy_spec_validation.json",
                "generated_strategy_module": "generated_strategy.py",
                "generation_trace": "generation_trace.json",
                "llm_migration_report": "pine_llm_migration_report.json",
                "beta_readiness_report": "pine_beta_readiness_report.json",
                "execution_gate_report": "pine_execution_gate_report.json",
                "order_semantics_report": "pine_order_semantics_report.json",
                "parity_report": "pine_parity_report.json",
                "request_security_diagnostics": "pine_request_security_diagnostics.json",
                "mtf_parity_proof_report": "pine_mtf_parity_proof_report.json",
                "parity_reference_payload": "pine_parity_reference.v1.json",
                "parity_reference_validation": "pine_parity_reference_validation.json",
                "parity_reference_metrics_legacy": "pine_parity_reference_metrics.json",
            },
        }
    )
