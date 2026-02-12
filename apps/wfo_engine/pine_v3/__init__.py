"""Pine v3 support package (spec generation and validation)."""

from .spec import (
    build_strategy_spec_v1_from_pine_text,
    validate_strategy_spec_v1,
)
from .mtf import (
    resample_ohlcv,
    realign_series_to_base,
    request_security_series,
    request_security_signal,
)
from .codegen import (
    build_generated_strategy_source,
    generate_strategy_module_from_spec,
)
from .parity import (
    build_parity_report,
    build_parity_reference_payload,
    validate_parity_reference_payload,
    normalize_metrics as normalize_parity_metrics,
    normalize_events as normalize_parity_events,
    normalize_trade_pairs as normalize_parity_trades,
    DEFAULT_PARITY_THRESHOLDS,
    DEFAULT_PARITY_DETAIL_THRESHOLDS,
    PARITY_REFERENCE_SCHEMA_VERSION,
    PARITY_REFERENCE_VALIDATION_SCHEMA_VERSION,
)
from .execution_gate import (
    build_execution_gate_report,
)
from .mtf_parity import (
    build_mtf_parity_proof_report,
)

__all__ = [
    "build_strategy_spec_v1_from_pine_text",
    "validate_strategy_spec_v1",
    "resample_ohlcv",
    "realign_series_to_base",
    "request_security_series",
    "request_security_signal",
    "build_generated_strategy_source",
    "generate_strategy_module_from_spec",
    "build_parity_report",
    "build_parity_reference_payload",
    "validate_parity_reference_payload",
    "normalize_parity_metrics",
    "normalize_parity_events",
    "normalize_parity_trades",
    "DEFAULT_PARITY_THRESHOLDS",
    "DEFAULT_PARITY_DETAIL_THRESHOLDS",
    "PARITY_REFERENCE_SCHEMA_VERSION",
    "PARITY_REFERENCE_VALIDATION_SCHEMA_VERSION",
    "build_execution_gate_report",
    "build_mtf_parity_proof_report",
]
