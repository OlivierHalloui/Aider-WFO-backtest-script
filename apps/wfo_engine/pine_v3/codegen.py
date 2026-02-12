"""Deterministic code generation for Pine V3 strategy modules."""

from __future__ import annotations

from pathlib import Path
import hashlib
import pprint
import re
from typing import Any


def _sanitize_token(value: str, default: str = "pine_strategy") -> str:
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or default


def build_generated_strategy_source(
    strategy_spec: dict[str, Any],
    import_mapping: dict[str, Any] | None = None,
    import_resolution: list[dict[str, Any]] | None = None,
) -> str:
    """Render a generated Python adapter module from `strategy_spec.v1`."""
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    strategy = spec.get("strategy") if isinstance(spec.get("strategy"), dict) else {}
    strategy_id = str(strategy.get("id") or "pine_strategy_test").strip() or "pine_strategy_test"
    strategy_name = str(strategy.get("name") or "Imported Pine Strategy").strip() or "Imported Pine Strategy"
    safe_id = _sanitize_token(strategy_id, default="pine_strategy")
    mapping = import_mapping if isinstance(import_mapping, dict) else {}
    resolution = import_resolution if isinstance(import_resolution, list) else []

    mapping_json = pprint.pformat(mapping, sort_dicts=False, width=100)
    resolution_json = pprint.pformat(resolution, sort_dicts=False, width=100)
    spec_json = pprint.pformat(spec, sort_dicts=False, width=100)

    return (
        '"""Auto-generated Pine adapter module (WFOE V3)."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n"
        "from pine_v3.runtime_adapter import GeneratedPineRuntimeAdapter\n\n"
        f"GENERATED_FROM_STRATEGY_ID = {strategy_id!r}\n"
        f"GENERATED_FROM_STRATEGY_NAME = {strategy_name!r}\n"
        f"GENERATED_ADAPTER_ID = {safe_id!r}\n\n"
        f"GENERATED_STRATEGY_SPEC = {spec_json}\n\n"
        f"EXTERNAL_IMPORT_MAPPING = {mapping_json}\n\n"
        f"EXTERNAL_IMPORT_RESOLUTION = {resolution_json}\n\n"
        "class GeneratedPineAdapter:\n"
        "    strategy_mode = 'pine_imported'\n"
        f"    strategy_id = {strategy_id!r}\n\n"
        "    def __init__(self, runtime_config: dict[str, Any] | None = None):\n"
        "        self.runtime_config = runtime_config\n\n"
        "    def _merged_config(self) -> dict[str, Any]:\n"
        "        cfg = dict(self.runtime_config or {})\n"
        "        if EXTERNAL_IMPORT_MAPPING and not isinstance(cfg.get('pine_import_mapping'), dict):\n"
        "            cfg['pine_import_mapping'] = dict(EXTERNAL_IMPORT_MAPPING)\n"
        "        if EXTERNAL_IMPORT_RESOLUTION:\n"
        "            pre = cfg.get('pine_precheck_report')\n"
        "            if not isinstance(pre, dict):\n"
        "                pre = {}\n"
        "            if not isinstance(pre.get('import_resolution'), list):\n"
        "                pre['import_resolution'] = EXTERNAL_IMPORT_RESOLUTION\n"
        "            cfg['pine_precheck_report'] = pre\n"
        "        if GENERATED_STRATEGY_SPEC and not isinstance(cfg.get('pine_strategy_spec'), dict):\n"
        "            cfg['pine_strategy_spec'] = GENERATED_STRATEGY_SPEC\n"
        "        return cfg\n\n"
        "    def get_param_space(self) -> dict[str, Any]:\n"
        "        base = GeneratedPineRuntimeAdapter(\n"
        "            strategy_id=self.strategy_id,\n"
        "            runtime_config=self._merged_config(),\n"
        "            strategy_spec=GENERATED_STRATEGY_SPEC,\n"
        "        )\n"
        "        return base.get_param_space()\n\n"
        "    def generate_signals(self, df, params: dict[str, Any]):\n"
        "        base = GeneratedPineRuntimeAdapter(\n"
        "            strategy_id=self.strategy_id,\n"
        "            runtime_config=self._merged_config(),\n"
        "            strategy_spec=GENERATED_STRATEGY_SPEC,\n"
        "        )\n"
        "        return base.generate_signals(df, params)\n\n"
        "    def run_backtest(self, df, params: dict[str, Any], timeframe: str = '5s', return_portfolio: bool = True):\n"
        "        base = GeneratedPineRuntimeAdapter(\n"
        "            strategy_id=self.strategy_id,\n"
        "            runtime_config=self._merged_config(),\n"
        "            strategy_spec=GENERATED_STRATEGY_SPEC,\n"
        "        )\n"
        "        return base.run_backtest(df, params, timeframe=timeframe, return_portfolio=return_portfolio)\n\n"
        "def create_adapter(runtime_config: dict[str, Any] | None = None) -> GeneratedPineAdapter:\n"
        "    return GeneratedPineAdapter(runtime_config=runtime_config)\n"
    )


def generate_strategy_module_from_spec(
    strategy_spec: dict[str, Any],
    output_dir: str,
    import_mapping: dict[str, Any] | None = None,
    import_resolution: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate and persist `generated_strategy.py` from spec + mappings."""
    spec = strategy_spec if isinstance(strategy_spec, dict) else {}
    output_root = Path(str(output_dir)).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    strategy = spec.get("strategy") if isinstance(spec.get("strategy"), dict) else {}
    source = spec.get("source") if isinstance(spec.get("source"), dict) else {}
    strategy_id = str(strategy.get("id") or "pine_strategy_test").strip() or "pine_strategy_test"
    source_sha1 = str(source.get("source_sha1") or "").strip()
    safe_id = _sanitize_token(strategy_id, default="pine_strategy")
    suffix = source_sha1[:12] if source_sha1 else hashlib.sha1(strategy_id.encode("utf-8")).hexdigest()[:12]
    module_name = f"generated_{safe_id}_{suffix}.py"
    target = output_root / module_name

    rendered = build_generated_strategy_source(
        strategy_spec=spec,
        import_mapping=import_mapping,
        import_resolution=import_resolution,
    )
    changed = True
    if target.exists():
        try:
            current = target.read_text(encoding="utf-8")
            changed = current != rendered
        except Exception:
            changed = True
    if changed:
        target.write_text(rendered, encoding="utf-8")

    return {
        "status": "ok",
        "output_path": str(target),
        "module_name": module_name,
        "changed": bool(changed),
        "strategy_id": strategy_id,
        "source_sha1": source_sha1 or None,
    }
