from __future__ import annotations

from .analyzer import ExpertAnalyzer
from .models import ExpertInputData, ExpertRequest, ExpertResponse, LLMConfig
from .storage import ExpertStorage


class ExpertService:
    """Thin service layer combining analysis execution and persistence."""

    def __init__(self, analyzer: ExpertAnalyzer, storage: ExpertStorage):
        self.analyzer = analyzer
        self.storage = storage

    def run(
        self,
        data: ExpertInputData,
        request: ExpertRequest,
        llm_config: LLMConfig,
        persist: bool = True,
        system_prompt_override: str | None = None,
        user_prompt_override: str | None = None,
    ) -> ExpertResponse:
        """Execute expert analysis and optionally persist report + audit trail."""
        response = self.analyzer.analyze(
            data=data,
            request=request,
            llm_config=llm_config,
            system_prompt_override=system_prompt_override,
            user_prompt_override=user_prompt_override,
        )
        if persist:
            self.storage.save_report(response)
            self.storage.save_audit_line(response)
        return response
