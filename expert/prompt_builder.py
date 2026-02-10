from __future__ import annotations

import json
from typing import Any, Dict

from .models import AnalysisMode, DetailLevel, ExpertInputData, ExpertRequest


class ExpertPromptBuilder:
    """Builds deterministic prompts for expert interpretation."""

    def build_system_prompt(self, mode: AnalysisMode, detail_level: DetailLevel) -> str:
        return (
            "Tu es un expert quant senior en validation de strategies de trading. "
            "Tu interpretes des resultats WFO/adaptatifs. "
            "N'invente rien: utilise uniquement les donnees fournies. "
            "Si une information manque, ecris 'insufficient_data'. "
            "Reponds en langue francaise, avec tolerance aux anglicismes de la profession (trading/quant). "
            "Utilise explicitement le contexte strategie (logique entree/sortie, modules actifs, roles des parametres) "
            "pour expliquer les observations. "
            "Confronte les alertes deterministes aux constats statistiques et indique accord/desaccord motive. "
            "L'analyse doit couvrir explicitement les resultats du final backtest et les comparer au buy&hold si disponible. "
            "Priorise robustesse OOS, risque de sur-optimisation, stabilite parametrique, "
            "et actions concretes de configuration. "
            "Reponds strictement en JSON valide sans markdown ni texte hors JSON. "
            f"Mode d'analyse: {mode}. Niveau de detail: {detail_level}."
        )

    def build_user_prompt(self, data: ExpertInputData, request: ExpertRequest, output_schema: Dict[str, Any]) -> str:
        payload = {
            "run_context": {
                "run_id": data.context.run_id,
                "optimization_regime": data.context.optimization_regime,
                "timeframe": data.context.timeframe,
                "start_date": data.context.start_date,
                "end_date": data.context.end_date,
                "selected_params": data.context.selected_params,
            },
            "oos_performance": data.out_of_sample_performance,
            "is_performance": data.in_sample_performance,
            "best_params_by_window": data.best_params,
            "trials_compact": data.all_trials,
            "adaptive_guidance": data.adaptive_guidance,
            "adaptive_summary": data.adaptive_summary,
            "price_features": data.price_features,
            "strategy_context": data.strategy_context,
            "deterministic_alerts": data.deterministic_alerts,
            "final_backtest": data.final_backtest,
            "request": {
                "mode": request.mode,
                "detail_level": request.detail_level,
                "include_raw_evidence": request.include_raw_evidence,
                "user_question": request.user_question or "",
            },
        }

        schema_hint = {
            "schema_version": "expert.v1",
            "run_id": "string",
            "mode": "summary|diagnostic|action_plan|alerts",
            "detail_level": "short|standard|expert",
            "global_assessment": {
                "quality_score": "0-100",
                "robustness_score": "0-100",
                "overfitting_risk_score": "0-100",
                "confidence_score": "0-100",
            },
            "key_findings": [
                {
                    "id": "F1",
                    "severity": "low|medium|high|critical",
                    "title": "string",
                    "explanation": "string",
                    "evidence": ["string"],
                    "impacted_windows": [1],
                    "impacted_params": ["param_name"],
                }
            ],
            "strategy_alignment": {
                "entry_logic_fit": "strong|medium|weak|insufficient_data",
                "exit_logic_fit": "strong|medium|weak|insufficient_data",
                "comments": ["string"],
            },
            "final_backtest_assessment": {
                "summary": "string",
                "strategy_return_pct": "number|insufficient_data",
                "buy_hold_return_pct": "number|insufficient_data",
                "outperformance_vs_buy_hold_pct": "number|insufficient_data",
                "max_drawdown_pct": "number|insufficient_data",
                "sharpe": "number|insufficient_data",
                "win_rate_pct": "number|insufficient_data",
                "n_trades": "number|insufficient_data",
                "comment": "string",
            },
            "is_oos_generalization": {
                "return_gap_mean": "number|insufficient_data",
                "sharpe_gap_mean": "number|insufficient_data",
                "gap_trend": "improving|degrading|flat|unclear",
                "comment": "string",
            },
            "adaptive_diagnostics": {
                "search_space_reduction_ratio_mean": "number|insufficient_data",
                "convergence_state": "early|progressing|plateau|unstable|unclear",
                "exploration_vs_exploitation": {
                    "assessment": "balanced|too_exploratory|too_exploitative|unclear",
                    "comment": "string",
                },
            },
            "recommended_actions": [
                {
                    "priority": 1,
                    "action": "string",
                    "expected_benefit": "string",
                    "risk": "string",
                    "estimated_cost": "low|medium|high",
                }
            ],
            "alerts": [
                {
                    "type": "overfitting|instability|insufficient_trials|inconclusive|other",
                    "severity": "low|medium|high|critical",
                    "message": "string",
                }
            ],
            "next_run_config_patch": {
                "adaptive_keep_ratio": "number|no_change",
                "adaptive_exploration_ratio": "number|no_change",
                "adaptive_trials_per_cycle": "number|no_change",
                "adaptive_decay": "number|no_change",
                "notes": "string",
            },
            "limitations": ["string"],
            "disclaimer": "string",
        }

        return (
            "Analyse les donnees ci-dessous et renvoie un JSON unique conforme au schema cible.\n\n"
            f"Schema minimal requis: {json.dumps(output_schema, ensure_ascii=True)}\n\n"
            f"Template recommande: {json.dumps(schema_hint, ensure_ascii=True)}\n\n"
            f"Donnees: {json.dumps(payload, ensure_ascii=True, default=self._json_safe)}"
        )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)
