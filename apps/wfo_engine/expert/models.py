from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Provider = Literal["openai", "grok", "gemini"]
DetailLevel = Literal["short", "standard", "expert"]
AnalysisMode = Literal["summary", "diagnostic", "action_plan", "alerts"]


@dataclass
class LLMConfig:
    """Provider/model call settings used by the Expert gateway."""
    provider: Provider
    model: str
    api_key: str
    base_url: Optional[str] = None
    temperature: float = 0.2
    timeout_s: float = 90.0
    max_tokens: int = 1800
    retries: int = 2


@dataclass
class ExpertRunContext:
    """High-level metadata describing the optimization run under analysis."""
    run_id: str
    optimization_regime: str
    timeframe: str
    start_date: str
    end_date: str
    selected_params: List[str] = field(default_factory=list)


@dataclass
class ExpertInputData:
    """Structured dataset sent to the Expert analyzer for interpretation."""
    context: ExpertRunContext
    out_of_sample_performance: List[Dict[str, Any]]
    in_sample_performance: List[Dict[str, Any]]
    best_params: List[Dict[str, Any]]
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    adaptive_guidance: List[Dict[str, Any]] = field(default_factory=list)
    adaptive_summary: Dict[str, Any] = field(default_factory=dict)
    price_features: Dict[str, Any] = field(default_factory=dict)
    strategy_context: Dict[str, Any] = field(default_factory=dict)
    deterministic_alerts: List[Dict[str, Any]] = field(default_factory=list)
    final_backtest: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertRequest:
    """User request options controlling Expert analysis mode and detail level."""
    mode: AnalysisMode
    detail_level: DetailLevel
    user_question: Optional[str]
    include_raw_evidence: bool = True


@dataclass
class ExpertResponse:
    """Normalized Expert output persisted and rendered in the UI."""
    run_id: str
    status: Literal["ok", "partial", "error"]
    result_json: Dict[str, Any]
    raw_text: str
    model_info: Dict[str, Any]
    timings_ms: Dict[str, int]
    warnings: List[str] = field(default_factory=list)
