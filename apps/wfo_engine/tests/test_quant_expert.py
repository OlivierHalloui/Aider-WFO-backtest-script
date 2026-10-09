"""Unit tests for services/quant_expert.py (T2).

Covers the A2A client (mock pair over a local HTTP server), the
``quant_analysis.v1`` schema validator (NaN / diagnostics field / enums /
``evidence_refs`` nomenclature), the ``{{ID.champ}}`` citation rule, the cache
key + ``force`` bypass, the bounded prompt builder and the reference resolver.
No VectorBT / Streamlit dependency — stdlib + numpy/pandas only.
"""

from __future__ import annotations

import http.server
import json
import threading
import urllib.error

import pandas as pd
import pytest

from services import quant_expert as e
from services.quant_expert import (
    ANALYSIS_SCHEMA_VERSION,
    _paired_windows,
    build_expert_prompt,
    build_sealed_context,
    call_quant_expert,
    expert_cache_key,
    parse_citation_references,
    probe_agent_card,
    reference_exists,
    validate_quant_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _diagnostics(run_id="run-1", input_digest="abc123"):
    return {
        "schema_version": "quant_indicators.v1",
        "indicator_version": "1.0.0",
        "generated_at": "2026-10-09T00:00:00+00:00",
        "manifest": {
            "run_id": run_id,
            "input_digest": input_digest,
            "timeframe": "5s",
            "n_windows": 2,
        },
        "integrity": {"ok": True, "causes": [], "checks": {}},
        "indicators": {
            "Q6": {"median_erosion_sharpe_pct": {"value": 12.5}},
            "Q7": {"final_sharpe_recomputed": {"available": True}},
        },
        "pre_verdict": {
            "verdict": "WATCH",
            "scope": "exploratoire",
            "status": "ok",
            "criteria": {},
            "rationale": [],
        },
    }


def _valid_analysis():
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "run_id": "run-1",
        "input_digest": "abc123",
        "indicator_version": "1.0.0",
        "generated_at": "2026-10-09T00:00:00+00:00",
        "verdict": "WATCH",
        "verdict_scope": "exploratoire",
        "confidence": "moyenne",
        "verdict_justification": "Érosion modérée ({{Q6.median_erosion_sharpe_pct.value}}).",
        "blocking_findings": [],
        "limitations": ["échantillon OOS limité"],
        "findings": [
            {
                "id": "F1",
                "niveau": "majeur",
                "constat": "Érosion modérée ({{Q6.median_erosion_sharpe_pct.value}}).",
                "evidence_refs": ["Q6", "W1"],
                "interpretation": "plateau stable",
                "condition": None,
            }
        ],
        "recommandations": [
            {"action": "élargir l'échantillon", "priorite": 1, "motif": "robustesse", "critere_de_validation": "GO"}
        ],
        "accord_avec_preverdict": True,
        "preverdict_local": "WATCH",
    }


def _start_pair(artifact_text=None, state="TASK_STATE_COMPLETED"):
    """Start a mock A2A peer on a free port -> (server, thread, base_url, box)."""
    box = {"artifact_text": artifact_text, "state": state, "hits": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: D102
            pass

        def _reply(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
                card = {"name": "wfo-quant-test", "url": "http://127.0.0.1:0", "version": "1.0.0"}
                self._reply(200, json.dumps(card).encode())
            else:
                self._reply(404, b"")

        def do_POST(self):  # noqa: N802
            box["hits"] += 1
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            task = {"id": "task-1", "contextId": "ctx-test", "status": {"state": box["state"]}}
            if box["state"] == "TASK_STATE_INPUT_REQUIRED":
                task["status"]["message"] = {
                    "role": "ROLE_AGENT",
                    "parts": [{"text": "[INPUT_REQUIRED] précision requise", "mediaType": "text/plain"}],
                }
            elif box["artifact_text"] is not None:
                task["artifacts"] = [
                    {"artifactId": "a1", "parts": [{"text": box["artifact_text"], "mediaType": "text/plain"}]}
                ]
            body = json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"task": task}}).encode()
            self._reply(200, body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, thread, base, box


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def test_cache_key_stable():
    d1 = _diagnostics()
    d2 = _diagnostics()
    assert expert_cache_key(d1) == expert_cache_key(d2)


def test_cache_key_varies_with_run_and_version():
    base = _diagnostics()
    assert expert_cache_key(_diagnostics(run_id="run-2")) != expert_cache_key(base)
    assert expert_cache_key(_diagnostics(input_digest="zzz")) != expert_cache_key(base)
    changed = _diagnostics()
    changed["indicator_version"] = "2.0.0"
    assert expert_cache_key(changed) != expert_cache_key(base)


# ---------------------------------------------------------------------------
# Agent card probe
# ---------------------------------------------------------------------------

def test_probe_agent_card_reachable():
    server, thread, base, _ = _start_pair()
    try:
        result = probe_agent_card(base)
        assert result["reachable"] is True
        assert result["name"] == "wfo-quant-test"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_probe_agent_card_unreachable():
    # port 1 on localhost is (essentially) never bound.
    result = probe_agent_card("http://127.0.0.1:1")
    assert result["reachable"] is False
    assert result["raison"]


# ---------------------------------------------------------------------------
# call_quant_expert — happy path + cache
# ---------------------------------------------------------------------------

def test_call_quant_expert_ok():
    server, thread, base, box = _start_pair(artifact_text=json.dumps(_valid_analysis()))
    try:
        result = call_quant_expert(_diagnostics(), url=base)
        assert result["status"] == "ok"
        assert result["analysis"]["schema_version"] == ANALYSIS_SCHEMA_VERSION
        assert result["analysis"]["verdict"] == "WATCH"
        assert result["context_id"] == "ctx-test"
        assert box["hits"] == 1
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_call_quant_expert_cached_then_force():
    server, thread, base, box = _start_pair(artifact_text=json.dumps(_valid_analysis()))
    cache = {}
    try:
        r1 = call_quant_expert(_diagnostics(), url=base, cache=cache)
        assert r1["status"] == "ok" and r1["cached"] is False
        r2 = call_quant_expert(_diagnostics(), url=base, cache=cache)
        assert r2["status"] == "ok" and r2["cached"] is True
        assert box["hits"] == 1  # cache hit -> no second network call
        r3 = call_quant_expert(_diagnostics(), url=base, cache=cache, force=True)
        assert r3["cached"] is False
        assert box["hits"] == 2  # force -> second network call
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# call_quant_expert — error paths
# ---------------------------------------------------------------------------

def test_call_quant_expert_unreachable():
    result = call_quant_expert(_diagnostics(), url="http://127.0.0.1:1")
    assert result["status"] == "unreachable"
    assert result["raison"]
    assert result["marche_a_suivre"]


def test_call_quant_expert_timeout(monkeypatch):
    import services.quant_expert as mod

    def _boom(url, prompt, context_id, timeout):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(mod, "_send_message", _boom)
    result = call_quant_expert(_diagnostics(), context_id="ctx-abc")
    assert result["status"] == "timeout"
    assert result["context_id"] == "ctx-abc"  # preserved for resumption
    assert "Relancer" in (result["marche_a_suivre"] or "")


def test_call_quant_expert_input_required():
    server, thread, base, _ = _start_pair(state="TASK_STATE_INPUT_REQUIRED")
    try:
        result = call_quant_expert(_diagnostics(), url=base)
        assert result["status"] == "input_required"
        assert "approvals.mode" in (result["marche_a_suivre"] or "")
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_call_quant_expert_empty_reply():
    server, thread, base, _ = _start_pair(artifact_text="")
    try:
        result = call_quant_expert(_diagnostics(), url=base)
        assert result["status"] == "empty"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_call_quant_expert_non_json_reply():
    server, thread, base, _ = _start_pair(artifact_text="pas du JSON")
    try:
        result = call_quant_expert(_diagnostics(), url=base)
        assert result["status"] == "invalid_schema"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_call_quant_expert_nan_rejected():
    payload = _valid_analysis()
    payload["confidence"] = float("nan")
    server, thread, base, _ = _start_pair(artifact_text=json.dumps(payload))
    try:
        result = call_quant_expert(_diagnostics(), url=base)
        assert result["status"] == "invalid_schema"
        assert any("NaN/Infinity" in err for err in result["errors"])
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_validate_accepts_conformant():
    result = validate_quant_analysis(_valid_analysis(), diagnostics=_diagnostics())
    assert result["ok"] is True
    assert result["errors"] == []


def test_validate_rejects_non_dict():
    result = validate_quant_analysis("nope")
    assert result["ok"] is False
    assert result["errors"]


def test_validate_rejects_missing_fields():
    result = validate_quant_analysis({"schema_version": ANALYSIS_SCHEMA_VERSION})
    assert result["ok"] is False
    assert any("requis" in err for err in result["errors"])


def test_validate_rejects_bad_enum():
    payload = _valid_analysis()
    payload["verdict"] = "MAYBE"
    result = validate_quant_analysis(payload)
    assert result["ok"] is False
    assert any("verdict hors enum" in err for err in result["errors"])


def test_validate_rejects_nan():
    payload = _valid_analysis()
    payload["verdict_justification"] = float("nan")
    result = validate_quant_analysis(payload)
    assert result["ok"] is False
    assert any("NaN/Infinity" in err for err in result["errors"])


def test_validate_flags_diagnostics_field():
    payload = _valid_analysis()
    payload["diagnostics"] = {"Q1": 123}
    result = validate_quant_analysis(payload)
    assert result["ok"] is True  # out-of-schema field is ignored, not blocking
    assert any("diagnostics" in w for w in result["warnings"])


def test_validate_rejects_bad_evidence_ref():
    payload = _valid_analysis()
    payload["findings"][0]["evidence_refs"] = ["Q6", "Z9"]
    result = validate_quant_analysis(payload)
    assert result["ok"] is False
    assert any("nomenclature" in err for err in result["errors"])


def test_validate_flags_bare_number_citation():
    payload = _valid_analysis()
    payload["findings"][0]["constat"] = "érosion de 35 % observée"
    result = validate_quant_analysis(payload, sealed=build_sealed_context(_diagnostics()))
    assert any("chiffre_non_reference" in w for w in result["warnings"])


def test_validate_flags_unresolvable_reference():
    payload = _valid_analysis()
    payload["findings"][0]["constat"] = "voir {{Q6.champ_inexistant}}"
    result = validate_quant_analysis(payload, sealed=build_sealed_context(_diagnostics()))
    assert any("référence introuvable" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Citation rule + reference resolution
# ---------------------------------------------------------------------------

def test_parse_citation_references():
    text = "érosion {{Q6.median_erosion_sharpe_pct.value}} sur {{W3.sharpe_per_bar}} soit 12,5 %."
    parsed = parse_citation_references(text)
    assert parsed["references"] == ["Q6.median_erosion_sharpe_pct.value", "W3.sharpe_per_bar"]
    assert "12,5" in parsed["bare_numbers"]


def test_parse_citation_references_no_bare_number_inside_placeholder():
    # digits inside {{...}} must not be flagged as bare numbers
    text = "réf {{W3.sharpe_per_bar}} seule"
    parsed = parse_citation_references(text)
    assert parsed["bare_numbers"] == []


def test_reference_exists_resolution():
    sealed = {
        "manifest": {"timeframe": "5s"},
        "indicators": {"Q6": {"median_erosion_sharpe_pct": {"value": 12.5}}},
        "pre_verdict": {"verdict": "WATCH"},
        "windows": [{"window": 1, "oos": {"return": 5.0}}],
        "trials": pd.DataFrame([{"timeperiod": 10}, {"timeperiod": 12}]),
        "param_grid": {"timeperiod": (5, 15)},
    }
    assert reference_exists("Q6.median_erosion_sharpe_pct.value", sealed) is True
    assert reference_exists("M.timeframe", sealed) is True
    assert reference_exists("P.timeperiod", sealed) is True
    assert reference_exists("W1.oos.return", sealed) is True
    assert reference_exists("T2", sealed) is True
    assert reference_exists("Q6.nonexistent", sealed) is False
    assert reference_exists("P.nonexistent", sealed) is False
    assert reference_exists("W9", sealed) is False


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def test_build_prompt_contains_blocks():
    prompt = build_expert_prompt(_diagnostics())
    assert "[MANIFESTE §5.0]" in prompt
    assert "[PRÉ-VERDICT LOCAL §5.3]" in prompt
    assert "[INDICATEURS Q1–Q8" in prompt
    assert "NON-annualisé" in prompt
    assert "{{ID.champ}}" in prompt
    assert "quant_analysis.v1" in prompt


def test_build_prompt_declares_window_truncation():
    wfo = {
        "in_sample_performance": [{"window": i, "return": 1.0, "n_trades": 10} for i in range(5)],
        "out_of_sample_performance": [{"window": i, "return": 1.0, "n_trades": 10} for i in range(5)],
    }
    prompt = build_expert_prompt(_diagnostics(), wfo_results=wfo, max_window_rows=3)
    assert "TRONQUÉ à 3" in prompt


def test_build_prompt_respects_char_budget():
    prompt = build_expert_prompt(_diagnostics(), max_prompt_chars=1000)
    assert len(prompt) <= 1000
    # the instruction tail is preserved even under a tight budget
    assert "Réponds UNIQUEMENT" in prompt


def test_build_prompt_respects_ceiling_and_keeps_instruction():
    # finding 2: a voluminous indicators block must not push the instruction out
    big = _diagnostics()
    big["indicators"] = {"Q1": {"trials": [{"i": i, "v": "x" * 50} for i in range(500)]}}
    prompt = build_expert_prompt(big, max_prompt_chars=16000)
    assert len(prompt) <= 16000
    assert "Réponds UNIQUEMENT" in prompt
    assert "NON-annualisé" in prompt
    assert "TRONQUÉ" in prompt


# ---------------------------------------------------------------------------
# Findings round 2 — coherence / inner schema / sealing
# ---------------------------------------------------------------------------

def test_validate_run_identifier_mismatch():
    payload = _valid_analysis()
    payload["run_id"] = "other-run"
    result = validate_quant_analysis(payload, diagnostics=_diagnostics())
    assert result["ok"] is False
    assert any("run_id" in err for err in result["errors"])


def test_validate_indicator_version_mismatch():
    payload = _valid_analysis()
    payload["indicator_version"] = "9.9.9"
    result = validate_quant_analysis(payload, diagnostics=_diagnostics())
    assert result["ok"] is False
    assert any("indicator_version" in err for err in result["errors"])


def test_validate_preverdict_local_mismatch():
    payload = _valid_analysis()
    payload["preverdict_local"] = "GO"  # sealed is WATCH
    result = validate_quant_analysis(payload, diagnostics=_diagnostics())
    assert result["ok"] is False
    assert any("preverdict_local" in err for err in result["errors"])


def test_accord_local_computed_not_trusted():
    result = validate_quant_analysis(_valid_analysis(), diagnostics=_diagnostics())
    assert result["accord_local"] is True
    disagree = _valid_analysis()
    disagree["verdict"] = "NO_GO"  # peer disagrees with the sealed WATCH
    result2 = validate_quant_analysis(disagree, diagnostics=_diagnostics())
    assert result2["accord_local"] is False


def test_validate_evidence_refs_not_list_no_crash():
    # finding 3: evidence_refs: 3 must not raise, but invalid_schema
    payload = _valid_analysis()
    payload["findings"][0]["evidence_refs"] = 3
    result = validate_quant_analysis(payload)
    assert result["ok"] is False
    assert any("evidence_refs" in err for err in result["errors"])


def test_call_quant_expert_evidence_refs_int_via_http():
    payload = _valid_analysis()
    payload["findings"][0]["evidence_refs"] = 3
    server, thread, base, _ = _start_pair(artifact_text=json.dumps(payload))
    try:
        result = call_quant_expert(_diagnostics(), url=base)
        assert result["status"] == "invalid_schema"
        assert any("evidence_refs" in err for err in result["errors"])
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_validate_inner_object_schema():
    # finding 4: inner objects are enforced (niveau / interpretation / action / generated_at)
    payload = _valid_analysis()
    payload["findings"][0]["niveau"] = "invalide"
    payload["findings"][0]["interpretation"] = 3
    payload["recommandations"][0]["action"] = 2
    payload["generated_at"] = "x"
    result = validate_quant_analysis(payload)
    assert result["ok"] is False
    assert any("niveau" in err for err in result["errors"])
    assert any("interpretation" in err for err in result["errors"])
    assert any("action" in err for err in result["errors"])
    assert any("generated_at" in err for err in result["errors"])


def test_diagnostics_stripped_from_analysis():
    payload = _valid_analysis()
    payload["diagnostics"] = {"Q1": 123}
    server, thread, base, _ = _start_pair(artifact_text=json.dumps(payload))
    try:
        result = call_quant_expert(_diagnostics(), url=base)
        assert result["status"] == "ok"
        assert "diagnostics" not in result["analysis"]
        assert "accord_local" in result["analysis"]
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_reference_exists_trial_field():
    # finding 5: T<n>.<field> must verify the requested field, not just the row
    sealed = {
        "manifest": {},
        "indicators": {},
        "pre_verdict": {},
        "windows": [],
        "trials": pd.DataFrame([{"timeperiod": 10}]),
        "param_grid": {},
    }
    assert reference_exists("T1.timeperiod", sealed) is True
    assert reference_exists("T1.nonexistent", sealed) is False


def test_paired_windows_params_match_and_sort():
    wfo = {
        "in_sample_performance": [
            {"window": 10, "params_sha": "a", "return": 1.0},
            {"window": 2, "params_sha": "a", "return": 1.0},
        ],
        "out_of_sample_performance": [
            {"window": 10, "params_sha": "b", "return": 1.0},
            {"window": 2, "params_sha": "a", "return": 1.0},
        ],
    }
    wins = _paired_windows(wfo)
    assert [w["window"] for w in wins] == [2, 10]  # numeric-aware sort
    by_id = {w["window"]: w for w in wins}
    assert by_id[2]["params_match"] is True   # same params_sha
    assert by_id[10]["params_match"] is False  # differing params_sha, surfaced


# ---------------------------------------------------------------------------
# Corrections des 5 findings de revue (wfo-reviewer)
# ---------------------------------------------------------------------------

def test_integrity_ko_no_network_no_cache(monkeypatch):
    """Finding 1 — intégrité KO : aucun appel réseau, aucune lecture de cache."""
    d = _diagnostics()
    d["integrity"] = {"ok": False, "causes": ["run_present: wfo_results absent ou vide"], "checks": {}}
    key = expert_cache_key(d)
    # un cache PIÉGÉ : s'il était lu, il renverrait ce résultat « ok »
    cache = {key: {"status": "ok", "analysis": {"verdict": "GO"}, "cached": False}}

    def _forbidden(*args, **kwargs):
        raise AssertionError("aucun appel réseau ne doit être émis (intégrité KO)")

    monkeypatch.setattr(e, "_send_message", _forbidden)
    monkeypatch.setattr(e, "probe_agent_card", _forbidden)

    result = call_quant_expert(d, cache=cache)

    assert result["status"] == "non_evaluable"
    assert result["analysis"] is None
    assert result["cached"] is False          # cache NON lu
    assert cache[key]["analysis"] == {"verdict": "GO"}  # cache intact
    assert "run_present" in (result["raison"] or "")
    assert result["marche_a_suivre"]


def test_paired_windows_params_match_absent_sha_is_false():
    """Finding 2 — params_match ne doit jamais passer sur None == None."""
    # les deux params_sha absents -> False (et non True par None == None)
    both_absent = {
        "in_sample_performance": [{"window": 1, "return": 1.0}],
        "out_of_sample_performance": [{"window": 1, "return": 1.0}],
    }
    assert _paired_windows(both_absent)[0]["params_match"] is False

    # un seul des deux présent -> False
    one_missing = {
        "in_sample_performance": [{"window": 1, "params_sha": "a"}],
        "out_of_sample_performance": [{"window": 1}],
    }
    assert _paired_windows(one_missing)[0]["params_match"] is False

    # chaînes vides -> False (empreinte non vide exigée)
    empty_sha = {
        "in_sample_performance": [{"window": 1, "params_sha": ""}],
        "out_of_sample_performance": [{"window": 1, "params_sha": ""}],
    }
    assert _paired_windows(empty_sha)[0]["params_match"] is False

    # empreintes non vides et identiques -> True (seul cas accepté)
    ok = {
        "in_sample_performance": [{"window": 1, "params_sha": "a"}],
        "out_of_sample_performance": [{"window": 1, "params_sha": "a"}],
    }
    assert _paired_windows(ok)[0]["params_match"] is True


def test_build_prompt_keeps_window_and_trials_blocks_with_huge_q1():
    """Finding 3 — un gros bloc Q1 ne doit pas évincer tableaux ni essais, ni la consigne."""
    big = _diagnostics()
    big["indicators"] = {"Q1": {"blob": "x" * 16000}}
    wfo = {
        "in_sample_performance": [{"window": i, "return": 1.0, "n_trades": 10} for i in range(3)],
        "out_of_sample_performance": [{"window": i, "return": 1.0, "n_trades": 10} for i in range(3)],
    }
    trials = pd.DataFrame([{"timeperiod": 10}, {"timeperiod": 12}])

    prompt = build_expert_prompt(
        big, wfo_results=wfo, all_trials=trials, max_prompt_chars=16000
    )

    assert "TABLEAUX PAR FENÊTRE" in prompt
    assert "ESSAIS PERTINENTS" in prompt
    assert "Réponds UNIQUEMENT" in prompt       # la consigne survit
    assert "NON-annualisé" in prompt            # le rappel de sémantiques aussi
    assert len(prompt) <= 16000
    # c'est bien le bloc Q1 qui est sacrifié, jamais les preuves par fenêtre
    assert "TRONQUÉ" in prompt


def test_reference_exists_w10_ordinal_not_id_fallback():
    """Finding 4 — W<n> est un ordinal : W10 refusé avec 2 fenêtres, même si l'id 10 existe."""
    sealed = {
        "manifest": {},
        "indicators": {},
        "pre_verdict": {},
        "windows": [
            {"window": 1, "oos": {"return": 5.0}},
            {"window": 10, "oos": {"return": 6.0}},   # l'id 10 existe, mais c'est la 2e fenêtre
        ],
        "trials": None,
        "param_grid": {},
    }
    # W10 = 10e fenêtre -> hors bornes (2 fenêtres) -> rejeté, aucun repli par id
    assert reference_exists("W10.oos.return", sealed) is False
    assert reference_exists("W10", sealed) is False
    # la résolution reste strictement ordinale
    assert reference_exists("W1.oos.return", sealed) is True
    assert reference_exists("W2.oos.return", sealed) is True   # 2e fenêtre = id 10
    assert reference_exists("W3.oos.return", sealed) is False


def test_accord_avec_preverdict_overwritten_by_computed_accord():
    """Finding 5 — accord_avec_preverdict est écrasé par l'accord calculé localement."""
    payload = _valid_analysis()
    payload["verdict"] = "NO_GO"                     # diffère du pré-verdict scellé (WATCH)
    payload["preverdict_local"] = "WATCH"            # écho correct du scellé -> validation OK
    payload["accord_avec_preverdict"] = True         # le pair affirme à tort l'accord
    payload["verdict_justification"] = "Désaccord ({{Q6.median_erosion_sharpe_pct.value}})."
    payload["findings"][0]["constat"] = "Désaccord ({{Q6.median_erosion_sharpe_pct.value}})."

    server, thread, base, _ = _start_pair(artifact_text=json.dumps(payload))
    try:
        result = call_quant_expert(_diagnostics(), url=base)
        assert result["status"] == "ok"
        # une seule valeur de référence : l'accord calculé, jamais le booléen du pair
        assert result["accord_local"] is False
        assert result["analysis"]["accord_avec_preverdict"] is False
        assert result["analysis"]["accord_local"] is False
    finally:
        server.shutdown()
        thread.join(timeout=5)
