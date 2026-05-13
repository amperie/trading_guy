from __future__ import annotations

import argparse

from trading.launchers.mlflow_hpo_launcher import run_launcher


def cmd_hpo_from_mlflow(args: argparse.Namespace):
    run_launcher(
        args.run_url,
        args.account,
        tracking_uri=getattr(args, "tracking_uri", None),
        editor=getattr(args, "editor", None),
    )
