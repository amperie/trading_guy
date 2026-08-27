from __future__ import annotations

import argparse
import json

from algo_crucible.orchestrator import CrucibleOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Algo crucible orchestration")
    parser.add_argument("--platform-config", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--milestone", default="1", choices=["1", "3", "4", "5"])
    parser.add_argument("--local", action="store_true", help="Disable Ray for stages that support parallel jobs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    orchestrator = CrucibleOrchestrator(args.platform_config, args.workload_config)
    if args.milestone == "1":
        result = orchestrator.run_milestone1(rerun=args.rerun)
    elif args.milestone == "3":
        result = orchestrator.run_walk_forward_oos(rerun=args.rerun, use_ray=not args.local)
    elif args.milestone == "4":
        result = orchestrator.run_hpo_stage(rerun=args.rerun)
    else:
        result = orchestrator.run_regime_gate_stage(rerun=args.rerun)
    print(json.dumps({
        "status": result.get("status"),
        "crucible_run_id": result.get("crucible_run_id"),
        "run_dir": result.get("run_dir"),
        "summary": result.get("summary"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
