"""Expert-analysis rendering helpers for Streamlit UI."""

from __future__ import annotations

import numpy as np
import streamlit as st


def render_deterministic_alerts(alerts):
    if not alerts:
        st.info("Aucune alerte déterministe déclenchée.")
        return
    severity_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    sorted_alerts = sorted(
        [a for a in alerts if isinstance(a, dict)],
        key=lambda a: severity_rank.get(str(a.get("severity", "low")).lower(), 0),
        reverse=True,
    )
    for alert in sorted_alerts:
        sev = str(alert.get("severity", "low")).lower()
        msg = str(alert.get("message", "Alerte"))
        typ = str(alert.get("type", "other"))
        rule = str(alert.get("rule", ""))
        text = f"[{typ}] {msg}"
        if rule:
            text += f"\nRègle: {rule}"
        if sev in {"critical", "high"}:
            st.error(text)
        elif sev == "medium":
            st.warning(text)
        else:
            st.info(text)


def render_interpretation_guide(points):
    """Display a compact interpretation guide below a chart."""
    if not points:
        return
    clean_points = []
    for point in points:
        text = str(point).strip()
        if text:
            clean_points.append(text)
    if not clean_points:
        return
    st.caption("Guide d'interprétation")
    st.markdown("\n".join([f"- {p}" for p in clean_points]))


def expert_value_to_text(value, digits=2, suffix=""):
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            val = float(value)
            if np.isfinite(val):
                fmt = f"{{:.{int(digits)}f}}"
                return f"{fmt.format(val)}{suffix}"
        except Exception:
            pass
    if value is None:
        return "insufficient_data"
    txt = str(value).strip()
    return txt if txt else "insufficient_data"


def build_expert_markdown_report(result_json, meta=None):
    """Build a readable markdown report from expert JSON output + run metadata."""
    meta = meta or {}
    r = result_json if isinstance(result_json, dict) else {}
    lines = []

    run_id = str(r.get("run_id") or meta.get("run_id") or "n/a")
    mode = str(r.get("mode") or "n/a")
    detail = str(r.get("detail_level") or "n/a")
    lines.append("# Rapport Expert IA")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Niveau de détail: `{detail}`")
    lines.append("")

    deterministic_alerts = meta.get("deterministic_alerts", []) if isinstance(meta, dict) else []
    if isinstance(deterministic_alerts, list) and deterministic_alerts:
        lines.append("## Alertes automatiques (règles déterministes)")
        for a in deterministic_alerts:
            if not isinstance(a, dict):
                continue
            sev = expert_value_to_text(a.get("severity")).upper()
            typ = expert_value_to_text(a.get("type"))
            msg = expert_value_to_text(a.get("message"))
            rule = expert_value_to_text(a.get("rule"))
            lines.append(f"- [{sev}] {typ}: {msg}")
            if rule != "insufficient_data":
                lines.append(f"  - Règle: {rule}")
        lines.append("")

    fb_assessment = r.get("final_backtest_assessment", {})
    fb_meta = meta.get("final_backtest", {}) if isinstance(meta, dict) else {}
    if (isinstance(fb_assessment, dict) and fb_assessment) or (isinstance(fb_meta, dict) and fb_meta):
        lines.append("## Final Backtest")
        if isinstance(fb_assessment, dict) and fb_assessment:
            lines.append(f"- Synthèse IA: {expert_value_to_text(fb_assessment.get('summary'))}")
            lines.append(f"- Return stratégie: **{expert_value_to_text(fb_assessment.get('strategy_return_pct'))}%**")
            lines.append(f"- Return buy&hold: **{expert_value_to_text(fb_assessment.get('buy_hold_return_pct'))}%**")
            lines.append(
                f"- Sur/Sous-performance vs buy&hold: **{expert_value_to_text(fb_assessment.get('outperformance_vs_buy_hold_pct'))} pts**"
            )
            lines.append(f"- Max Drawdown: **{expert_value_to_text(fb_assessment.get('max_drawdown_pct'))}%**")
            lines.append(f"- Sharpe: **{expert_value_to_text(fb_assessment.get('sharpe'))}**")
            lines.append(f"- Win Rate: **{expert_value_to_text(fb_assessment.get('win_rate_pct'))}%**")
            lines.append(f"- Nombre de trades: **{expert_value_to_text(fb_assessment.get('n_trades'))}**")
            comment = expert_value_to_text(fb_assessment.get("comment"))
            if comment != "insufficient_data":
                lines.append(f"- Commentaire: {comment}")
        elif isinstance(fb_meta, dict) and fb_meta:
            lines.append(f"- Return stratégie: **{expert_value_to_text(fb_meta.get('strategy_total_return_pct'))}%**")
            lines.append(f"- Return buy&hold: **{expert_value_to_text(fb_meta.get('buy_hold_return_pct'))}%**")
            lines.append(
                f"- Sur/Sous-performance vs buy&hold: **{expert_value_to_text(fb_meta.get('outperformance_vs_buy_hold_pct'))} pts**"
            )
            lines.append(f"- Max Drawdown: **{expert_value_to_text(fb_meta.get('strategy_max_drawdown_pct'))}%**")
            lines.append(f"- Sharpe: **{expert_value_to_text(fb_meta.get('strategy_sharpe'))}**")
            lines.append(f"- Win Rate: **{expert_value_to_text(fb_meta.get('strategy_win_rate_pct'))}%**")
            lines.append(f"- Nombre de trades: **{expert_value_to_text(fb_meta.get('strategy_n_trades'))}**")
        lines.append("")

    ga = r.get("global_assessment", {})
    if isinstance(ga, dict) and ga:
        lines.append("## Évaluation globale")
        lines.append(f"- Qualité: **{expert_value_to_text(ga.get('quality_score'), digits=1, suffix='/100')}**")
        lines.append(f"- Robustesse: **{expert_value_to_text(ga.get('robustness_score'), digits=1, suffix='/100')}**")
        lines.append(f"- Risque de sur-optimisation: **{expert_value_to_text(ga.get('overfitting_risk_score'), digits=1, suffix='/100')}**")
        lines.append(f"- Confiance: **{expert_value_to_text(ga.get('confidence_score'), digits=1, suffix='/100')}**")
        lines.append("")

    findings = r.get("key_findings", [])
    if isinstance(findings, list) and findings:
        lines.append("## Constats clés")
        for idx, f in enumerate(findings, start=1):
            if not isinstance(f, dict):
                continue
            title = str(f.get("title") or f"Constat {idx}")
            sev = str(f.get("severity") or "n/a")
            exp = str(f.get("explanation") or "insufficient_data")
            lines.append(f"{idx}. **{title}** (sévérité: `{sev}`)")
            lines.append(f"   - Explication: {exp}")
            ev = f.get("evidence", [])
            if isinstance(ev, list) and ev:
                lines.append("   - Preuves:")
                for e in ev[:5]:
                    lines.append(f"     - {str(e)}")
        lines.append("")

    strategy_alignment = r.get("strategy_alignment", {})
    if isinstance(strategy_alignment, dict) and strategy_alignment:
        lines.append("## Alignement avec la logique stratégie")
        lines.append(f"- Cohérence logique d'entrée: **{expert_value_to_text(strategy_alignment.get('entry_logic_fit'))}**")
        lines.append(f"- Cohérence logique de sortie: **{expert_value_to_text(strategy_alignment.get('exit_logic_fit'))}**")
        comments = strategy_alignment.get("comments", [])
        if isinstance(comments, list) and comments:
            for c in comments[:6]:
                lines.append(f"- {str(c)}")
        lines.append("")

    gen = r.get("is_oos_generalization", {})
    if isinstance(gen, dict) and gen:
        lines.append("## Généralisation IS/OOS")
        lines.append(f"- Gap moyen Return (IS-OOS): **{expert_value_to_text(gen.get('return_gap_mean'))}**")
        lines.append(f"- Gap moyen Sharpe (IS-OOS): **{expert_value_to_text(gen.get('sharpe_gap_mean'))}**")
        lines.append(f"- Tendance du gap: **{expert_value_to_text(gen.get('gap_trend'))}**")
        lines.append(f"- Commentaire: {expert_value_to_text(gen.get('comment'))}")
        lines.append("")

    ada = r.get("adaptive_diagnostics", {})
    if isinstance(ada, dict) and ada:
        lines.append("## Diagnostic Adaptatif")
        lines.append(
            f"- Réduction moyenne de l'espace de recherche: **{expert_value_to_text(ada.get('search_space_reduction_ratio_mean'))}**"
        )
        lines.append(f"- État de convergence: **{expert_value_to_text(ada.get('convergence_state'))}**")
        ex = ada.get("exploration_vs_exploitation", {})
        if isinstance(ex, dict) and ex:
            lines.append(f"- Exploration/Exploitation: **{expert_value_to_text(ex.get('assessment'))}**")
            lines.append(f"- Commentaire: {expert_value_to_text(ex.get('comment'))}")
        lines.append("")

    actions = r.get("recommended_actions", [])
    if isinstance(actions, list) and actions:
        lines.append("## Plan d'action recommandé")
        sortable = []
        for i, a in enumerate(actions):
            if isinstance(a, dict):
                pr = a.get("priority", i + 1)
                try:
                    pr = int(pr)
                except Exception:
                    pr = i + 1
                sortable.append((pr, a))
        sortable.sort(key=lambda x: x[0])
        for pr, a in sortable:
            lines.append(f"{pr}. **{expert_value_to_text(a.get('action'))}**")
            lines.append(f"   - Bénéfice attendu: {expert_value_to_text(a.get('expected_benefit'))}")
            lines.append(f"   - Risque: {expert_value_to_text(a.get('risk'))}")
            lines.append(f"   - Coût estimé: `{expert_value_to_text(a.get('estimated_cost'))}`")
        lines.append("")

    alerts = r.get("alerts", [])
    if isinstance(alerts, list) and alerts:
        lines.append("## Alertes")
        for a in alerts:
            if not isinstance(a, dict):
                continue
            lines.append(
                f"- [{expert_value_to_text(a.get('severity')).upper()}] "
                f"{expert_value_to_text(a.get('type'))}: {expert_value_to_text(a.get('message'))}"
            )
        lines.append("")

    limits = r.get("limitations", [])
    if isinstance(limits, list) and limits:
        lines.append("## Limites")
        for l in limits:
            lines.append(f"- {expert_value_to_text(l)}")
        lines.append("")

    disclaimer = r.get("disclaimer")
    if disclaimer:
        lines.append("## Note")
        lines.append(str(disclaimer))

    return "\n".join(lines).strip()


def render_expert_human_report(result_json, meta=None):
    report_md = build_expert_markdown_report(result_json, meta=meta)
    if not report_md:
        st.info("Le rapport Expert est vide ou non interprétable.")
        return ""
    st.markdown(report_md)
    return report_md
