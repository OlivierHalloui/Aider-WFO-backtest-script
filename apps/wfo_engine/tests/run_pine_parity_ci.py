"""CLI runner for Pine V3 parity CI campaign (P2.3)."""

from __future__ import annotations

import argparse
import json
import os
import sys

TEST_DIR = os.path.dirname(__file__)
WFO_ENGINE_DIR = os.path.abspath(os.path.join(TEST_DIR, ".."))
if WFO_ENGINE_DIR not in sys.path:
    sys.path.append(WFO_ENGINE_DIR)

from pine_v3.parity_ci import run_parity_ci_campaign, write_parity_ci_report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pine V3 parity CI campaign and write a JSON report.")
    parser.add_argument(
        "--output",
        default=os.path.join("reports", "ci", "pine_parity_ci_report.json"),
        help="Path to output report JSON.",
    )
    parser.add_argument(
        "--require-runtime",
        action="store_true",
        help="Fail if runtime scenarios (vectorbtpro) cannot be executed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = run_parity_ci_campaign(require_runtime=bool(args.require_runtime))
    output_path = write_parity_ci_report(report, args.output)

    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "output": output_path,
                "total_scenarios": summary.get("total_scenarios"),
                "passed": summary.get("passed"),
                "failed": summary.get("failed"),
                "blockers": summary.get("blockers"),
                "runtime_executed": report.get("runtime_executed"),
            },
            ensure_ascii=False,
        )
    )

    return 0 if str(report.get("status")) == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

