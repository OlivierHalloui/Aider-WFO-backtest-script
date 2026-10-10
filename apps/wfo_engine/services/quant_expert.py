"""A2A interpretation client for the « Analyse quant » panel (T2).

Implements §5.2 + §6.1 of the cahier des charges:

- a **minimal JSON-RPC client** (stdlib ``urllib.request``, no new dependency)
  towards the local ``wfo-quant`` peer (A2A v1.0 ``SendMessage`` method);
- the **``quant_analysis.v1`` schema validator** (reject ``NaN``/``Infinity``,
  reject a ``diagnostics`` field, check enums, ``evidence_refs`` nomenclature and
  the ``{{ID.champ}}`` citation rule);
- a **per-run cache** keyed by the run hash + ``indicator_version``.

Design rules (from the cahier, §5):

- The application **seals the facts**; the peer only returns interpreted
  findings.  The client never sends raw metrics it does not own, and never
  trusts a bare number from the peer (citation rule).
- **Never ``0`` for a missing value**: the validator treats ``NaN``/``Infinity``
  as a hard schema violation and flags out-of-schema fields.
- Pure and re-runnable: no Streamlit / VectorBT import; the network is the only
  side effect and is isolated behind :func:`call_quant_expert`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from domain.serialization import sanitize_for_json, sha256_json, utc_now_iso

# ---------------------------------------------------------------------------
# Configuration (frozen defaults, env-overridable for tests)
# ---------------------------------------------------------------------------

DEFAULT_EXPERT_URL = "http://127.0.0.1:9924"
EXPERT_TIMEOUT_SECONDS = 900      # §6.1: application timeout (reply 2–6 min, up to ~9 min)
PROBE_TIMEOUT_SECONDS = 30        # agent-card reachability probe

ANALYSIS_SCHEMA_VERSION = "quant_analysis.v1"
RPC_METHOD = "SendMessage"        # A2A v1.0 canonical name (adapter also accepts message/send)
A2A_PROTOCOL_VERSION = "1.0"

# Default prompt bounds (declared truncation, §5.2).
DEFAULT_MAX_WINDOW_ROWS = 100
DEFAULT_MAX_TRIAL_ROWS = 20
DEFAULT_MAX_PROMPT_CHARS = 16000

# Enums (frozen, §5.2).
VERDICTS = ("GO", "WATCH", "NO_GO")
VERDICT_SCOPES = ("exploratoire", "validation_oos", "deploiement")
CONFIDENCES = ("faible", "moyenne", "elevee")
FINDING_LEVELS = ("majeur", "mineur", "info")

# A2A v1.0 task states.
STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"

# ---------------------------------------------------------------------------
# Nomenclature (§5.2) — closed, non-extensible
# ---------------------------------------------------------------------------
# evidence_refs tokens: Q1..Q8 / W<n> / M.<champ> / P.<param> / T<n> (no sub-path
# beyond the M./P. single field).
_EVIDENCE_REF = re.compile(r"^(Q[1-8]|W[0-9]+|M\.[A-Za-z0-9_]+|P\.[A-Za-z0-9_]+|T[0-9]+)$")
# {{ID.champ}} citation placeholders: a nomenclature head, optionally followed by
# one or more .field sub-paths (e.g. {{Q6.erosion_sharpe_pct}}, {{W3.sharpe_per_bar}}).
_CITATION_REF = re.compile(r"^(Q[1-8]|W[0-9]+|M|P|T[0-9]+)(\.[A-Za-z0-9_]+)*$")
_PLACEHOLDER = re.compile(r"\{\{([^{}]+)\}\}")
# bare numeric literal outside a {{...}} placeholder (citation rule).
_BARE_NUMBER = re.compile(r"(?<![\w])(?:(?:\d+[.,]\d+)|(?:\d+))(?![\w])")


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def expert_cache_key(diagnostics: Mapping[str, Any]) -> str:
    """Per-run cache key = hash of ``(run_id, input_digest, indicator_version)``.

    Invalidated at any change of run or of the indicator logic (§6.1).
    """
    manifest = diagnostics.get("manifest") if isinstance(diagnostics, Mapping) else None
    manifest = manifest if isinstance(manifest, Mapping) else {}
    return sha256_json({
        "run_id": manifest.get("run_id"),
        "input_digest": manifest.get("input_digest"),
        "indicator_version": diagnostics.get("indicator_version"),
    }) or ""


# ---------------------------------------------------------------------------
# A2A wire helpers (stdlib only)
# ---------------------------------------------------------------------------

def _new_context_id() -> str:
    return "ctx-" + uuid.uuid4().hex[:16]


def _new_task_id() -> str:
    return "task-" + uuid.uuid4().hex[:16]


def _text_from_message(message: Any) -> str:
    """Concatenate the text parts of an A2A Message (v1.0 member-presence Parts)."""
    if not isinstance(message, dict):
        return ""
    chunks = []
    for part in message.get("parts", []) or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "\n".join(chunks).strip()


def _unwrap_send_response(result: Any) -> Any:
    """Task/Message inside a v1.0 SendMessageResponse; legacy bare payloads pass through."""
    if isinstance(result, dict):
        if isinstance(result.get("task"), dict):
            return result["task"]
        if isinstance(result.get("message"), dict):
            return result["message"]
    return result


def _reply_text(payload: Any) -> str:
    """Extract the final reply text from a Task/Message payload (artifacts first)."""
    payload = _unwrap_send_response(payload)
    if not isinstance(payload, dict):
        return str(payload)
    for artifact in payload.get("artifacts", []) or []:
        txt = _text_from_message(artifact)
        if txt:
            return txt
    status = payload.get("status")
    if isinstance(status, dict):
        return _text_from_message(status.get("message") or status)
    return _text_from_message(payload)


def _http_json(url: str, *, method: str, body: Optional[dict], timeout: int) -> dict:
    headers = {"Content-Type": "application/json", "A2A-Version": A2A_PROTOCOL_VERSION}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (localhost peer)
        return json.loads(resp.read().decode("utf-8"))


def probe_agent_card(
    url: Optional[str] = None,
    timeout: int = PROBE_TIMEOUT_SECONDS,
) -> dict:
    """Best-effort reachability probe of the peer's Agent Card (§6.1).

    Returns ``{"reachable": bool, "name": str|None, "raison": str|None}``.  A
    ``404`` on the v1.0 card falls back to the v0.2 ``agent.json`` alias.
    """
    base = (url or DEFAULT_EXPERT_URL).rstrip("/")
    for path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
        try:
            card = _http_json(base + path, method="GET", body=None, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            return {"reachable": False, "name": None, "raison": f"pair injoignable (HTTP {exc.code})"}
        except urllib.error.URLError as exc:
            return {"reachable": False, "name": None, "raison": f"pair injoignable ({exc.reason})"}
        except Exception as exc:  # noqa: BLE001 — surface any transport failure
            return {"reachable": False, "name": None, "raison": f"pair injoignable ({exc})"}
        return {
            "reachable": True,
            "name": card.get("name") if isinstance(card, dict) else None,
            "raison": None,
        }
    return {"reachable": False, "name": None, "raison": "pair injoignable (pas d'agent card)"}


def _send_message(
    url: str,
    prompt: str,
    context_id: Optional[str],
    timeout: int,
) -> tuple[str, str, str]:
    """One ``SendMessage`` to the peer -> ``(reply_text, context_id, state)``.

    Raises ``urllib.error.URLError`` (incl. socket timeout) and ``ValueError`` on
    a JSON-RPC error; the caller maps these to explicit UI statuses.
    """
    ctx = context_id or _new_context_id()
    body = {
        "jsonrpc": "2.0",
        "id": _new_task_id(),
        "method": RPC_METHOD,
        "params": {
            "message": {
                "role": "ROLE_USER",
                "parts": [{"text": prompt, "mediaType": "text/plain"}],
                "messageId": uuid.uuid4().hex,
                "contextId": ctx,
            }
        },
    }
    raw = _http_json(url.rstrip("/"), method="POST", body=body, timeout=timeout)
    if "error" in raw:
        err = raw["error"] if isinstance(raw["error"], dict) else {}
        raise ValueError(err.get("message") or str(raw["error"]))
    payload = _unwrap_send_response(raw.get("result", {}))
    reply = _reply_text(payload)
    reply_ctx, state = ctx, ""
    if isinstance(payload, dict):
        reply_ctx = payload.get("contextId") or ctx
        status = payload.get("status")
        if isinstance(status, dict):
            state = status.get("state") or ""
    return reply, reply_ctx, state


# ---------------------------------------------------------------------------
# Prompt builder (§5.2 — compact, bounded, with declared truncation)
# ---------------------------------------------------------------------------

def _window_sort_key(wid: Any):
    """Numeric-aware sort key so ``W<n>`` ordinals follow the natural window order."""
    try:
        return (0, int(wid))
    except (TypeError, ValueError):
        return (1, str(wid))


def _paired_windows(wfo_results: Mapping[str, Any]) -> list:
    """Merge IS/OOS per-window rows into paired entries ``{window, is, oos, params_match}``.

    ``params_match`` reports whether the IS and OOS rows of the same window carry
    the same selected parameters (``params_sha``).  A mismatch is surfaced, never
    silently presented as an apparied pair (§5.0).
    """
    is_perf = wfo_results.get("in_sample_performance") or []
    oos_perf = wfo_results.get("out_of_sample_performance") or []
    by_is = {r.get("window"): r for r in is_perf if isinstance(r, dict)}
    by_oos = {r.get("window"): r for r in oos_perf if isinstance(r, dict)}
    merged = []
    for wid in sorted(set(by_is) | set(by_oos), key=_window_sort_key):
        is_row, oos_row = by_is.get(wid), by_oos.get(wid)
        entry: dict[str, Any] = {"window": wid}
        if is_row is not None:
            entry["is"] = is_row
        if oos_row is not None:
            entry["oos"] = oos_row
        # params_match requires two NON-EMPTY, equal fingerprints — two absent
        # fingerprints must never pass as "matched" (None == None).
        is_sha = is_row.get("params_sha") if is_row is not None else None
        oos_sha = oos_row.get("params_sha") if oos_row is not None else None
        entry["params_match"] = bool(is_sha and oos_sha and is_sha == oos_sha)
        merged.append(entry)
    return merged


def _top_trials(wfo_results: Mapping[str, Any], all_trials: Optional[pd.DataFrame], limit: int) -> list:
    """A small, relevant set of trials: ``all_trials`` head, else per-window winners."""
    if isinstance(all_trials, pd.DataFrame) and not all_trials.empty:
        return all_trials.head(limit).to_dict("records")
    out = []
    for window in wfo_results.get("window_results") or []:
        if not isinstance(window, dict):
            continue
        if out and len(out) >= limit:
            break
        bp = window.get("best_params")
        if isinstance(bp, dict):
            out.append({"best_params": bp, "window": window.get("window_info", {}).get("window")})
    return out[:limit]


_SEMANTICS_REMINDER = (
    "Rappel des sémantiques : le Sharpe est NON-annualisé (moyenne/écart-type par barre, ddof=1) ; "
    "calc_avg_pl = mean(trades.returns)*100 ; unités figées : return_pct, win_rate_pct, sharpe_per_bar. "
    "Les rendements sont nets de frais (voir Q8.convention)."
)

_INSTRUCTION = (
    "Consigne : distingue FAITS (fournis ci-dessus), INFÉRENCES et HYPOTHÈSES ; ne recopie PAS les "
    "nombres déjà fournis — interprète-les. Cite tout chiffre sous la forme littérale {{ID.champ}} "
    "(jamais une valeur numérique nue). Réponds UNIQUEMENT avec un JSON conforme au schéma "
    "quant_analysis.v1, en français, en une seule réponse, sans question en retour."
)


def build_expert_prompt(
    diagnostics: Mapping[str, Any],
    *,
    wfo_results: Optional[Mapping[str, Any]] = None,
    all_trials: Optional[pd.DataFrame] = None,
    max_window_rows: int = DEFAULT_MAX_WINDOW_ROWS,
    max_trial_rows: int = DEFAULT_MAX_TRIAL_ROWS,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> str:
    """Build the compact, bounded prompt for ``wfo-quant`` (§5.2).

    Includes the manifest, the sealed Q1–Q8 aggregates, the local pre-verdict,
    the paired IS/OOS per-window tables, a few relevant trials, the semantics
    reminder and the answering instruction.  Truncation is **declared**.
    """
    manifest = diagnostics.get("manifest") if isinstance(diagnostics, Mapping) else None
    manifest = manifest if isinstance(manifest, Mapping) else {}
    indicators = diagnostics.get("indicators") if isinstance(diagnostics, Mapping) else None
    indicators = indicators if isinstance(indicators, Mapping) else {}
    pre_verdict = diagnostics.get("pre_verdict") if isinstance(diagnostics, Mapping) else None
    pre_verdict = pre_verdict if isinstance(pre_verdict, Mapping) else {}

    windows = _paired_windows(wfo_results if isinstance(wfo_results, Mapping) else {})
    windows_truncated = len(windows) > max_window_rows
    windows = windows[:max_window_rows]
    trials = _top_trials(
        wfo_results if isinstance(wfo_results, Mapping) else {},
        all_trials,
        max_trial_rows,
    )
    trials_truncated = bool(all_trials is not None and len(all_trials) > max_trial_rows)

    header = "ANALYSE QUANT — INTERPRÉTATION (wfo-quant)"
    tail = "\n" + _SEMANTICS_REMINDER + "\n\n" + _INSTRUCTION

    # Priority order matters: the essential evidence blocks (manifest, local
    # pre-verdict, paired IS/OOS tables, relevant trials) come FIRST; the
    # potentially voluminous Q1–Q8 aggregates come LAST so they are what gets
    # truncated when the budget is tight — never the per-window evidence (§5.2).
    blocks: list[tuple[str, str]] = [
        ("MANIFESTE §5.0", json.dumps(sanitize_for_json(manifest), ensure_ascii=False, indent=1)),
        ("PRÉ-VERDICT LOCAL §5.3", json.dumps(sanitize_for_json(pre_verdict), ensure_ascii=False, indent=1)),
        (
            f"TABLEAUX PAR FENÊTRE (IS/OOS appariés) — {len(windows)} fenêtre(s)"
            + (" — TRONQUÉ à " + str(max_window_rows) if windows_truncated else ""),
            json.dumps(sanitize_for_json(windows), ensure_ascii=False, indent=1),
        ),
        (
            f"ESSAIS PERTINENTS — {len(trials)} ligne(s)" + (" — TRONQUÉ" if trials_truncated else ""),
            json.dumps(sanitize_for_json(trials), ensure_ascii=False, indent=1),
        ),
        ("INDICATEURS Q1–Q8 (agrégats scellés)", json.dumps(sanitize_for_json(indicators), ensure_ascii=False, indent=1)),
    ]

    # Reserve room for the semantics + instruction tail AND the truncation note
    # first, so a voluminous data block can never push the answering instruction
    # out of the prompt nor overflow the declared budget.
    note_reserve = 160
    data_budget = max(0, max_prompt_chars - len(header) - len(tail) - note_reserve)
    data, dropped = _assemble_within_budget(blocks, data_budget)

    prompt = header + data
    if dropped:
        note = "\n[TRONQUÉ : " + "; ".join(dropped) + "]"
        if len(note) > note_reserve:
            note = note[:note_reserve - 1] + "]"
        prompt += note
    prompt += tail

    # Guarantee the instruction tail survives: trim the data/note region first,
    # never the tail — unless the tail alone overflows an absurd budget.
    if len(prompt) > max_prompt_chars:
        overflow = len(prompt) - max_prompt_chars
        keep = max(0, len(prompt) - len(tail) - overflow)
        prompt = prompt[:keep] + tail
        if len(prompt) > max_prompt_chars:
            prompt = prompt[:max_prompt_chars]
    return prompt


def _assemble_within_budget(blocks: list[tuple[str, str]], budget: int) -> tuple[str, list[str]]:
    """Greedily include ``(label, body)`` blocks within ``budget`` chars.

    Returns ``(data_text, dropped_labels)``.  When a block does not fit, its body
    is truncated at a line boundary (JSON with ``indent=1`` is line-oriented) and
    the remaining blocks are dropped and reported.  The instruction tail is always
    preserved because the caller reserves it *before* calling this helper.
    """
    parts: list[str] = []
    used = 0
    dropped: list[str] = []
    for i, (label, body) in enumerate(blocks):
        chunk = f"\n\n[{label}]\n{body}"
        if used + len(chunk) <= budget:
            parts.append(chunk)
            used += len(chunk)
            continue
        header_chunk = f"\n\n[{label}]\n"
        avail = budget - used
        if avail > len(header_chunk) + 40:  # enough room for a useful prefix
            parts.append(header_chunk + _truncate_json(body, avail - len(header_chunk)))
            dropped.append(f"{label} (tronqué)")
        else:
            dropped.append(label)
        dropped.extend(l for l, _ in blocks[i + 1:])
        break
    return "".join(parts), dropped


def _truncate_json(text: str, budget: int) -> str:
    """Truncate a (line-oriented) JSON string to ``budget`` chars at a line boundary."""
    if len(text) <= budget:
        return text
    prefix = text[:budget]
    cut = prefix.rfind("\n")
    if cut > 0:
        prefix = prefix[:cut]
    return prefix + "\n…"


# ---------------------------------------------------------------------------
# Citation rule + reference resolution (§5.2)
# ---------------------------------------------------------------------------

def parse_citation_references(text: Any) -> dict:
    """Parse ``{{ID.champ}}`` placeholders and bare numeric literals.

    Returns ``{"references": [...], "bare_numbers": [...]}``.  ``references`` are
    the raw inner tokens (validated separately); ``bare_numbers`` are numeric
    literals found *outside* a placeholder (→ ``chiffre_non_reference``).
    """
    if not isinstance(text, str):
        return {"references": [], "bare_numbers": []}
    references = [m.group(1).strip() for m in _PLACEHOLDER.finditer(text)]
    stripped = _PLACEHOLDER.sub(" ", text)
    bare = [m.group(0) for m in _BARE_NUMBER.finditer(stripped)]
    return {"references": references, "bare_numbers": bare}


def substitute_citations(
    text: Any,
    sealed: Mapping[str, Any],
    render: Optional[Callable[[Any], str]] = None,
) -> tuple[str, list[str]]:
    """Replace ``{{ID.champ}}`` with the sealed value (§5.2 source-of-truth rule).

    The application owns the numbers: a citation is rendered as **the value it
    resolves to**, with the reference kept next to it so the reader can trace
    it.  An unresolvable reference is left verbatim and reported in the second
    return value — it is never silently dropped nor invented.

    ``render`` formats a resolved value (defaults to a plain scalar repr) so the
    caller keeps its own unit/« indisponible » policy.
    """
    if not isinstance(text, str):
        return ("" if text is None else str(text)), []
    fmt = render or (lambda v: str(v))
    unresolved: list[str] = []

    def _sub(match: "re.Match[str]") -> str:
        ref = match.group(1).strip()
        value = resolve_reference(ref, sealed)
        if value is None:
            unresolved.append(ref)
            return match.group(0)
        return f"{fmt(value)} «{ref}»"

    return _PLACEHOLDER.sub(_sub, text), unresolved


def build_sealed_context(
    diagnostics: Mapping[str, Any],
    *,
    wfo_results: Optional[Mapping[str, Any]] = None,
    all_trials: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Assemble the sealed facts used to resolve ``{{ID.champ}}`` references."""
    manifest = diagnostics.get("manifest") if isinstance(diagnostics, Mapping) else {}
    manifest = manifest if isinstance(manifest, Mapping) else {}
    windows = _paired_windows(wfo_results if isinstance(wfo_results, Mapping) else {})
    return {
        "manifest": manifest,
        "indicators": diagnostics.get("indicators") or {},
        "pre_verdict": diagnostics.get("pre_verdict") or {},
        "windows": windows,
        "trials": all_trials,
        "param_grid": param_grid or {},
    }


def _resolve(base: Any, path: list[str]) -> tuple[bool, Any]:
    """Walk ``path`` into ``base``.  ``found`` is False on a dead end.

    A resolved-but-``None`` leaf counts as found here; callers decide what that
    means (``reference_exists`` treats it as absent, matching its historical
    behaviour).
    """
    cur = base
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif isinstance(cur, list) and key.isdigit() and int(key) < len(cur):
            cur = cur[int(key)]
        else:
            return False, None
    return True, cur


def _descend(base: Any, path: list[str]) -> bool:
    found, value = _resolve(base, path)
    return found and value is not None


def resolve_reference(ref: str, sealed: Mapping[str, Any]) -> Any:
    """Resolve ``{{ID.champ}}`` to the sealed value (§5.2).

    Returns the value, or ``None`` when the reference is out of nomenclature,
    unresolvable, or resolves to ``None``.  Same resolution semantics as
    :func:`reference_exists` — the two must never drift.
    """
    if not isinstance(ref, str) or not _CITATION_REF.match(ref):
        return None
    parts = ref.split(".")
    head = parts[0]
    try:
        if head[:1] == "Q" and head[1:].isdigit():
            return _resolve((sealed.get("indicators") or {}).get(head), parts[1:])[1]
        if head == "M":
            return _resolve(sealed.get("manifest"), parts[1:])[1]
        if head == "P":
            if parts[1] in (sealed.get("param_grid") or {}):
                return (sealed.get("param_grid") or {}).get(parts[1])
            return None
        if head[:1] == "W" and head[1:].isdigit():
            n = int(head[1:])
            windows = sealed.get("windows") or []
            # W<n> is a 1-based ORDINAL into the (numeric-aware sorted) window
            # list — never an id match, which would accept W10 over 2 windows.
            idx = n - 1
            if 0 <= idx < len(windows):
                return _resolve(windows[idx], parts[1:])[1]
            return None
        if head[:1] == "T" and head[1:].isdigit():
            idx = int(head[1:]) - 1
            trials_obj = sealed.get("trials")
            if not isinstance(trials_obj, pd.DataFrame):
                return None
            trials: pd.DataFrame = trials_obj
            if not (0 <= idx < len(trials)):
                return None
            return _resolve(trials.iloc[idx].to_dict(), parts[1:])[1]
    except Exception:  # noqa: BLE001 — resolution must never raise
        return None
    return None


def reference_exists(ref: str, sealed: Mapping[str, Any]) -> bool:
    """Return True iff ``ref`` **exists** against the sealed facts (§5.2).

    Distinct from :func:`resolve_reference`, which returns the *value*: a
    ``P.<param>`` grid entry present with a ``None`` value still exists as a
    reference.  Non-regression guard:
    ``reference_exists("P.foo", {"param_grid": {"foo": None}}) is True``.
    """
    if not isinstance(ref, str) or not _CITATION_REF.match(ref):
        return False
    parts = ref.split(".")
    head = parts[0]
    try:
        if head[:1] == "Q" and head[1:].isdigit():
            return _descend((sealed.get("indicators") or {}).get(head), parts[1:])
        if head == "M":
            return _descend(sealed.get("manifest"), parts[1:])
        if head == "P":
            return parts[1] in (sealed.get("param_grid") or {})
        if head[:1] == "W" and head[1:].isdigit():
            n = int(head[1:])
            windows = sealed.get("windows") or []
            # W<n> is a 1-based ORDINAL into the (numeric-aware sorted) window
            # list — never an id match, which would accept W10 over 2 windows.
            idx = n - 1
            if 0 <= idx < len(windows):
                return _descend(windows[idx], parts[1:])
            return False
        if head[:1] == "T" and head[1:].isdigit():
            idx = int(head[1:]) - 1
            trials_obj = sealed.get("trials")
            if not isinstance(trials_obj, pd.DataFrame):
                return False
            trials: pd.DataFrame = trials_obj
            if not (0 <= idx < len(trials)):
                return False
            return _descend(trials.iloc[idx].to_dict(), parts[1:])
    except Exception:  # noqa: BLE001 — resolution must never raise
        return False
    return False


# ---------------------------------------------------------------------------
# Schema validation (§5.2 / §5.3)
# ---------------------------------------------------------------------------

def _nan_inf_paths(obj: Any, path: str = "$") -> list[str]:
    """Return the JSON-pointer-ish paths of any NaN/Infinity leaf."""
    out: list[str] = []
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            out.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_nan_inf_paths(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_nan_inf_paths(v, f"{path}[{i}]"))
    return out


_REQUIRED_FIELDS = (
    "schema_version", "run_id", "input_digest", "indicator_version", "generated_at",
    "verdict", "verdict_scope", "confidence", "verdict_justification",
    "blocking_findings", "limitations", "findings", "recommandations",
    "accord_avec_preverdict", "preverdict_local",
)


def _validate_evidence_refs(refs: Any) -> list[str]:
    """Return invalid ``evidence_refs`` tokens (out of the closed nomenclature)."""
    if not isinstance(refs, (list, tuple)):
        return [str(refs)] if refs is not None else []
    invalid = []
    for ref in refs:
        if not isinstance(ref, str) or not _EVIDENCE_REF.match(ref):
            invalid.append(str(ref))
    return invalid


def invalid_evidence_refs(refs: Any) -> list[str]:
    """``evidence_refs`` outside the closed nomenclature (§5.2).

    Public entry point for the UI, which marks the carrying ``findings`` row as
    ``reference_invalide``.  Same semantics as the validator used internally.
    """
    return _validate_evidence_refs(refs)


_DATETIME_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}")


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _check_citation(txt: Any, where: str, errors: list, warnings: list, sealed: Optional[Mapping[str, Any]]) -> None:
    if not isinstance(txt, str):
        return
    parsed = parse_citation_references(txt)
    if parsed["bare_numbers"]:
        warnings.append(f"{where} : chiffre non référencé {parsed['bare_numbers'][:3]} (chiffre_non_reference)")
    if sealed is not None:
        for ref in parsed["references"]:
            if not reference_exists(ref, sealed):
                warnings.append(f"{where} : référence introuvable {{{{{ref}}}}}")


def validate_quant_analysis(
    payload: Any,
    *,
    diagnostics: Optional[Mapping[str, Any]] = None,
    sealed: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Validate a peer reply against the ``quant_analysis.v1`` schema.

    Returns ``{"ok": bool, "errors": [...], "warnings": [...], "accord_local": bool|None}``.

    **errors** (blocking): wrong schema, missing/wrong-typed required fields, bad
    enums, ``NaN``/``Infinity`` anywhere, non-conformant ``evidence_refs``,
    run-identifier mismatch (``run_id`` / ``indicator_version`` / ``input_digest``),
    a ``preverdict_local`` differing from the sealed local pre-verdict.

    **warnings** (non-blocking): a ``diagnostics`` field (out of schema, ignored),
    bare numeric literals (``chiffre_non_reference``), unresolvable ``{{ID.champ}}``
    placeholders.

    ``accord_local`` is computed from the sealed pre-verdict — the peer's
    ``accord_avec_preverdict`` boolean is never trusted (§5.2).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(payload, dict):
        return {"ok": False, "errors": ["réponse non objet (JSON attendu)"], "warnings": [], "accord_local": None}

    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        errors.append(f"schema_version attendu {ANALYSIS_SCHEMA_VERSION!r}")

    for field in _REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"champ requis manquant : {field!r}")

    # NaN / Infinity never leak (§5.3) — hard rejection.
    nan_paths = _nan_inf_paths(payload)
    if nan_paths:
        errors.append(f"NaN/Infinity interdit ({len(nan_paths)} valeur(s), ex. {nan_paths[0]})")

    # diagnostics is an INPUT block, never part of the peer's OUTPUT (§5.2).
    if "diagnostics" in payload:
        warnings.append("champ 'diagnostics' hors schéma (bloc d'entrée) — ignoré")

    # Enums.
    if payload.get("verdict") not in VERDICTS:
        errors.append(f"verdict hors enum {VERDICTS}")
    if payload.get("verdict_scope") not in VERDICT_SCOPES:
        errors.append(f"verdict_scope hors enum {VERDICT_SCOPES}")
    if payload.get("confidence") not in CONFIDENCES:
        errors.append(f"confidence hors enum {CONFIDENCES}")
    if payload.get("preverdict_local") not in VERDICTS:
        errors.append(f"preverdict_local hors enum {VERDICTS}")
    if not isinstance(payload.get("accord_avec_preverdict"), bool):
        errors.append("accord_avec_preverdict doit être un booléen")
    if not isinstance(payload.get("verdict_justification"), str) or not payload.get("verdict_justification"):
        errors.append("verdict_justification doit être une chaîne non vide")
    generated_at = payload.get("generated_at")
    if not isinstance(generated_at, str) or not _DATETIME_PREFIX.match(generated_at):
        errors.append("generated_at doit être un horodatage ISO (ex. 2026-10-09T…)")

    # Run identifiers must match the sealed manifest (blocking — the peer echoes
    # them, it does not author them).
    if diagnostics is not None:
        manifest = diagnostics.get("manifest") if isinstance(diagnostics, Mapping) else {}
        manifest = manifest if isinstance(manifest, Mapping) else {}
        if "run_id" in payload and manifest.get("run_id") and payload["run_id"] != manifest["run_id"]:
            errors.append("run_id du pair ≠ run_id du manifeste")
        if "input_digest" in payload and manifest.get("input_digest") and payload["input_digest"] != manifest["input_digest"]:
            errors.append("input_digest du pair ≠ input_digest du manifeste")
        if "indicator_version" in payload and diagnostics.get("indicator_version") and payload["indicator_version"] != diagnostics["indicator_version"]:
            errors.append("indicator_version du pair ≠ indicator_version scellée")

    for key in ("blocking_findings", "limitations", "findings", "recommandations"):
        if key in payload and not isinstance(payload[key], list):
            errors.append(f"{key!r} doit être une liste")

    # Inner-object schema: findings.
    for i, item in enumerate(_as_list(payload.get("findings"))):
        if not isinstance(item, dict):
            errors.append(f"findings[{i}] doit être un objet")
            continue
        if not isinstance(item.get("id"), str) or not item.get("id"):
            errors.append(f"findings[{i}].id doit être une chaîne non vide")
        if item.get("niveau") not in FINDING_LEVELS:
            errors.append(f"findings[{i}].niveau hors enum {FINDING_LEVELS}")
        if not isinstance(item.get("constat"), str):
            errors.append(f"findings[{i}].constat doit être une chaîne")
        if not isinstance(item.get("interpretation"), str):
            errors.append(f"findings[{i}].interpretation doit être une chaîne")
        if "condition" in item and item["condition"] is not None and not isinstance(item["condition"], str):
            errors.append(f"findings[{i}].condition doit être une chaîne ou null")
        refs = item.get("evidence_refs")
        if not isinstance(refs, list):
            errors.append(f"findings[{i}].evidence_refs doit être une liste de chaînes")
        else:
            invalid = _validate_evidence_refs(refs)
            if invalid:
                errors.append(f"findings[{i}].evidence_refs hors nomenclature : {invalid}")
        _check_citation(item.get("constat"), f"findings[{i}].constat", errors, warnings, sealed)
        _check_citation(item.get("interpretation"), f"findings[{i}].interpretation", errors, warnings, sealed)

    # Inner-object schema: blocking_findings.
    for i, item in enumerate(_as_list(payload.get("blocking_findings"))):
        if not isinstance(item, dict):
            errors.append(f"blocking_findings[{i}] doit être un objet")
            continue
        if not isinstance(item.get("id"), str) or not item.get("id"):
            errors.append(f"blocking_findings[{i}].id doit être une chaîne non vide")
        if not isinstance(item.get("constat"), str):
            errors.append(f"blocking_findings[{i}].constat doit être une chaîne")
        refs = item.get("evidence_refs")
        if not isinstance(refs, list):
            errors.append(f"blocking_findings[{i}].evidence_refs doit être une liste de chaînes")
        else:
            invalid = _validate_evidence_refs(refs)
            if invalid:
                errors.append(f"blocking_findings[{i}].evidence_refs hors nomenclature : {invalid}")
        _check_citation(item.get("constat"), f"blocking_findings[{i}].constat", errors, warnings, sealed)

    # Inner-object schema: recommandations.
    for i, item in enumerate(_as_list(payload.get("recommandations"))):
        if not isinstance(item, dict):
            errors.append(f"recommandations[{i}] doit être un objet")
            continue
        if not isinstance(item.get("action"), str):
            errors.append(f"recommandations[{i}].action doit être une chaîne")
        if not isinstance(item.get("priorite"), int):
            errors.append(f"recommandations[{i}].priorite doit être un entier")
        if not isinstance(item.get("motif"), str):
            errors.append(f"recommandations[{i}].motif doit être une chaîne")
        if not isinstance(item.get("critere_de_validation"), str):
            errors.append(f"recommandations[{i}].critere_de_validation doit être une chaîne")

    # verdict_justification citation rule.
    _check_citation(payload.get("verdict_justification"), "verdict_justification", errors, warnings, sealed)

    # Coherence with the sealed local pre-verdict (the peer echoes it, never rewrites it).
    accord_local = None
    if diagnostics is not None:
        pre = diagnostics.get("pre_verdict") if isinstance(diagnostics, Mapping) else {}
        pre = pre if isinstance(pre, Mapping) else {}
        local_verdict = pre.get("verdict")
        if local_verdict in VERDICTS:
            if "preverdict_local" in payload and payload["preverdict_local"] != local_verdict:
                errors.append(f"preverdict_local ({payload.get('preverdict_local')}) ≠ pré-verdict local scellé ({local_verdict})")
            accord_local = payload.get("verdict") == local_verdict

    return {"ok": not errors, "errors": errors, "warnings": warnings, "accord_local": accord_local}


# ---------------------------------------------------------------------------
# End-of-run hook (§5.2 « Analyse automatique en fin de run »)
# ---------------------------------------------------------------------------

def auto_analyze_run(
    wfo_results: Optional[Mapping[str, Any]] = None,
    *,
    diagnostics: Optional[Mapping[str, Any]] = None,
    all_trials: Optional[pd.DataFrame] = None,
    final_trades: Optional[pd.DataFrame] = None,
    per_bar_returns: Optional[Mapping[str, Any]] = None,
    oos_trades: Optional[pd.DataFrame] = None,
    config: Optional[Mapping[str, Any]] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
    cache: Optional[dict] = None,
    enabled: bool = True,
    runner: Optional[Callable[..., dict]] = None,
) -> dict:
    """End-of-run hook shared with ``services/run_service.py`` (§5.2).

    Computes the §5.1 diagnostics when not supplied, then runs the same
    interpretation path as the button.  ``enabled=False`` (checkbox unticked) is
    a no-op returning ``{}``, so the toggle really disables the analysis.  The
    automatic path keeps ``force=False``: the cache applies (§6.1).

    ``config``, ``param_grid`` and ``final_trades`` are **required** for the §5.0
    integrity check to pass (costs provenance, ``P.<param>`` references and the
    final-trade evidence).  Missing them yields ``non_evaluable`` — and
    :func:`call_quant_expert` then issues **no** A2A call (§5.3 case a).
    """
    if not enabled:
        return {}
    if diagnostics is None:
        from services.quant_indicators import compute_quant_indicators

        source = wfo_results if isinstance(wfo_results, Mapping) else {}
        diagnostics = compute_quant_indicators(
            source,
            all_trials=all_trials,
            final_trades=final_trades,
            per_bar_returns=per_bar_returns,
            oos_trades=oos_trades,
            config=config,
            param_grid=param_grid,
        )

    call = runner or call_quant_expert
    out = call(
        diagnostics,
        wfo_results=wfo_results,
        all_trials=all_trials,
        param_grid=param_grid,
        cache=cache,
        force=False,
    )
    return out if isinstance(out, dict) else dict(out or {})


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def call_quant_expert(
    diagnostics: Mapping[str, Any],
    *,
    wfo_results: Optional[Mapping[str, Any]] = None,
    all_trials: Optional[pd.DataFrame] = None,
    param_grid: Optional[Mapping[str, Any]] = None,
    force: bool = False,
    context_id: Optional[str] = None,
    cache: Optional[dict] = None,
    url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> dict:
    """Run the full interpretation flow towards ``wfo-quant`` (§6.1).

    Returns a structured result:

    ``status`` — ``ok`` | ``non_evaluable`` | ``unreachable`` | ``timeout`` |
    ``input_required`` | ``empty`` | ``invalid_schema`` | ``error`` ;
    ``context_id`` — preserved on timeout for resumption (§6.1) ;
    ``analysis`` — the validated ``quant_analysis.v1`` dict (only when ``ok``) ;
    ``errors`` / ``warnings`` — validator output ; ``raison`` / ``marche_a_suivre``
    — French user-facing status and next step ; ``cached`` — cache hit ;
    ``cache_key`` — the per-run key.

    ``force`` bypasses the cache ; ``context_id`` resumes a prior (timed-out)
    conversation.
    """
    key = expert_cache_key(diagnostics)
    result: dict[str, Any] = {
        "status": "error",
        "context_id": context_id,
        "analysis": None,
        "errors": [],
        "warnings": [],
        "raison": None,
        "marche_a_suivre": None,
        "cached": False,
        "cache_key": key,
    }

    # §5.0 / §7 — a failed integrity check blocks ANY A2A call (and any cache
    # read): no reliable facts to interpret.
    integrity = diagnostics.get("integrity") if isinstance(diagnostics, Mapping) else None
    integrity = integrity if isinstance(integrity, Mapping) else {}
    if not integrity.get("ok", False):
        result["status"] = "non_evaluable"
        result["raison"] = "contrôle d'intégrité en échec : " + ("; ".join(integrity.get("causes", [])) or "cause inconnue")
        result["marche_a_suivre"] = "Corriger les données du run (intégrité §5.0) avant toute analyse."
        return result

    if cache is not None and not force and key in cache:
        hit = dict(cache[key])
        hit["cached"] = True
        return hit

    target_url = (url or DEFAULT_EXPERT_URL).rstrip("/")
    eff_timeout = timeout if timeout is not None else EXPERT_TIMEOUT_SECONDS

    probe = probe_agent_card(target_url)
    if not probe["reachable"]:
        result["status"] = "unreachable"
        result["raison"] = probe["raison"] or "pair injoignable"
        result["marche_a_suivre"] = (
            "Vérifier que le profil wfo-quant est démarré sur 127.0.0.1:9924 "
            "(approvals.mode: off, platforms.a2a.enabled: true), puis relancer."
        )
        return result

    prompt = build_expert_prompt(
        diagnostics, wfo_results=wfo_results, all_trials=all_trials
    )

    # Generate the context up-front so it is preserved (and resumable) even when
    # the send fails/times out (§6.1).
    ctx = context_id or _new_context_id()
    result["context_id"] = ctx

    try:
        reply, reply_ctx, state = _send_message(target_url, prompt, ctx, eff_timeout)
    except OSError as exc:
        reason = getattr(exc, "reason", exc)
        is_timeout = (
            isinstance(reason, TimeoutError)
            or isinstance(exc, TimeoutError)
            or "timed out" in str(exc).lower()
            or "timeout" in str(exc).lower()
        )
        if is_timeout:
            result["status"] = "timeout"
            result["raison"] = f"délai dépassé ({eff_timeout} s)"
            result["marche_a_suivre"] = (
                "Le context_id a été conservé pour reprise — cliquer « Relancer » "
                "pour reprendre sans perdre le fil de l'analyse."
            )
        else:
            result["status"] = "unreachable"
            result["raison"] = f"pair injoignable ({reason})"
            result["marche_a_suivre"] = "Vérifier la connexion au pair, puis relancer."
        return result
    except ValueError as exc:
        result["status"] = "error"
        result["raison"] = str(exc)
        result["marche_a_suivre"] = "Erreur renvoyée par le pair — relancer ou vérifier sa configuration."
        return result

    result["context_id"] = reply_ctx

    if state == STATE_INPUT_REQUIRED:
        result["status"] = "input_required"
        result["raison"] = "le pair demande une saisie (input-required)"
        result["marche_a_suivre"] = (
            "Désactiver le gate d'approbation (approvals.mode: off) sur le profil "
            "wfo-quant, puis relancer."
        )
        return result

    if not reply or not reply.strip():
        result["status"] = "empty"
        result["raison"] = "réponse vide du pair"
        result["marche_a_suivre"] = "Relancer l'analyse."
        return result

    try:
        payload = json.loads(reply)
    except json.JSONDecodeError:
        result["status"] = "invalid_schema"
        result["raison"] = "réponse non-JSON du pair (hors schéma)"
        result["marche_a_suivre"] = "Relancer l'analyse ; le pair doit répondre un JSON quant_analysis.v1."
        return result

    sealed = build_sealed_context(
        diagnostics, wfo_results=wfo_results, all_trials=all_trials, param_grid=param_grid
    )
    validation = validate_quant_analysis(payload, diagnostics=diagnostics, sealed=sealed)
    result["errors"] = validation["errors"]
    result["warnings"] = validation["warnings"]
    result["accord_local"] = validation.get("accord_local")

    if not validation["ok"]:
        result["status"] = "invalid_schema"
        result["raison"] = "réponse hors schéma : " + "; ".join(validation["errors"][:3])
        result["marche_a_suivre"] = "Relancer l'analyse ; corriger le schéma de réponse du pair."
        return result

    # The ``diagnostics`` block is an INPUT, never part of the accepted artifact
    # (§5.2) — strip it even though it was only flagged as a warning.  The peer's
    # ``accord_avec_preverdict`` boolean is replaced by the locally-computed
    # agreement, so T4 has a single displayable source of truth.
    analysis = dict(payload)
    analysis.pop("diagnostics", None)
    accord_local = validation.get("accord_local")
    if accord_local is not None:
        analysis["accord_avec_preverdict"] = accord_local
        analysis["accord_local"] = accord_local

    result["status"] = "ok"
    result["analysis"] = analysis
    result["raison"] = None
    result["marche_a_suivre"] = None

    if cache is not None:
        cache[key] = {k: v for k, v in result.items() if k != "cached"}
    return result
