from __future__ import annotations

import argparse
import json

from algo_crucible.orchestrator import CrucibleOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Algo crucible orchestration")
    parser.add_argument("--platform-config", required=True)
    parser.add_argument("--workload-config", required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--milestone", default="1", choices=["1", "3", "4", "5", "6", "7", "8", "9"])
    parser.add_argument("--local", action="store_true", help="Disable Ray for stages that support parallel jobs")
    parser.add_argument("--create-promoted-folder", action="store_true", help="Write promotion packet into trading/promoted")
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
    elif args.milestone == "5":
        result = orchestrator.run_regime_gate_stage(rerun=args.rerun)
    elif args.milestone == "6":
        result = orchestrator.run_plateau_stage(rerun=args.rerun, use_ray=not args.local)
    elif args.milestone == "7":
        result = orchestrator.run_perturbation_stage(rerun=args.rerun, use_ray=not args.local)
    elif args.milestone == "8":
        result = orchestrator.run_confirmation_stage(
            rerun=args.rerun,
            use_ray=not args.local,
            create_promoted_folder=args.create_promoted_folder,
        )
    else:
        result = orchestrator.run_paper_replay_stage(rerun=args.rerun)
    print(json.dumps({
        "status": result.get("status"),
        "crucible_run_id": result.get("crucible_run_id"),
        "run_dir": result.get("run_dir"),
        "summary": result.get("summary"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
