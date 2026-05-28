from __future__ import annotations

import argparse

from trading.launchers.mlflow_hpo_launcher import run_launcher


def cmd_hpo_from_mlflow(args: argparse.Namespace):
    run_launcher(
        args.run_url,
        args.account,
        tracking_uri=getattr(args, "tracking_uri", None),
        editor=getattr(args, "editor", None),
        algorithm_param_overrides=getattr(args, "algorithm_param", None),
    )


def cmd_hpo_split_from_mlflow(args: argparse.Namespace):
    run_launcher(
        args.run_url,
        args.account,
        tracking_uri=getattr(args, "tracking_uri", None),
        editor=getattr(args, "editor", None),
        split_validation=True,
        algorithm_param_overrides=getattr(args, "algorithm_param", None),
    )
