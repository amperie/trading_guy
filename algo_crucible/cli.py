from __future__ import annotations

import argparse
import json

from algo_crucible.orchestrator import CrucibleOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Algo crucible orchestration")
    parser.add_argument("--platform-config", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--milestone", default="1", choices=["1"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = CrucibleOrchestrator(args.platform_config, args.workload_config).run_milestone1(rerun=args.rerun)
    print(json.dumps({
        "status": result.get("status"),
        "crucible_run_id": result.get("crucible_run_id"),
        "run_dir": result.get("run_dir"),
        "summary": result.get("summary"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
