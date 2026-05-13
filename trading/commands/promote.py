from __future__ import annotations

import argparse


def cmd_promote(args: argparse.Namespace):
    from trading.launchers.mlflow_promote_launcher import promote_run

    bundle = promote_run(
        args.run_url,
        tracking_uri=getattr(args, "tracking_uri", None),
        name=getattr(args, "name", None),
    )
    print(f"Promoted run {bundle.source_run_id} -> {bundle.config_path}")
    print(f"Manifest: {bundle.manifest_path}")
