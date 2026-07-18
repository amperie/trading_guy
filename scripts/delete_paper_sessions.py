"""Delete MongoDB trading sessions whose IDs start with a prefix.

Defaults to a dry run against the configured live trading database. Pass --yes
to delete matching sessions and all session-partitioned data.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config_manager import ConfigManager

SESSION_COLLECTION = "sessions"
RELATED_COLLECTIONS = (
    "ticks",
    "portfolio_snapshots",
    "orders",
    "signals",
    "optimization_events",
)


def _state_store_cfg() -> dict:
    return ConfigManager().get("state_store") or {}


def _query(prefix: str) -> dict:
    return {"_id": {"$regex": f"^{re.escape(prefix)}"}}


def matching_session_ids(db, prefix: str) -> list[str]:
    return [
        str(doc["_id"])
        for doc in db[SESSION_COLLECTION].find(_query(prefix), {"_id": 1}).sort("_id", 1)
    ]


def counts_by_collection(db, session_ids: list[str]) -> dict[str, int]:
    if not session_ids:
        return {SESSION_COLLECTION: 0, **{name: 0 for name in RELATED_COLLECTIONS}}

    counts = {SESSION_COLLECTION: len(session_ids)}
    related_query = {"session_id": {"$in": session_ids}}
    for name in RELATED_COLLECTIONS:
        counts[name] = db[name].count_documents(related_query)
    return counts


def delete_sessions(db, session_ids: list[str]) -> dict[str, int]:
    if not session_ids:
        return counts_by_collection(db, session_ids)

    deleted = {}
    related_query = {"session_id": {"$in": session_ids}}
    for name in RELATED_COLLECTIONS:
        deleted[name] = db[name].delete_many(related_query).deleted_count
    deleted[SESSION_COLLECTION] = db[SESSION_COLLECTION].delete_many(
        {"_id": {"$in": session_ids}}
    ).deleted_count
    return deleted


def build_parser() -> argparse.ArgumentParser:
    cfg = _state_store_cfg()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connection-uri",
        default=cfg.get("connection_uri", "mongodb://localhost:27017"),
    )
    parser.add_argument(
        "--database",
        default=cfg.get("default_live_database") or cfg.get("database", "live_trading"),
    )
    parser.add_argument("--prefix", default="paper_")
    parser.add_argument("--yes", action="store_true", help="Actually delete matched data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = MongoClient(args.connection_uri)
    db = client[args.database]

    session_ids = matching_session_ids(db, args.prefix)
    counts = delete_sessions(db, session_ids) if args.yes else counts_by_collection(db, session_ids)

    action = "Deleted" if args.yes else "Dry run"
    print(f"{action}: {len(session_ids)} session(s) in {args.database} matching {args.prefix!r}")
    for session_id in session_ids:
        print(f"  {session_id}")
    print("Counts:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    if not args.yes:
        print("No data deleted. Re-run with --yes to delete.")


if __name__ == "__main__":
    main()
