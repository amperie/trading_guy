from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    algorithm_class: str
    portfolio_class: str
    algorithm_params: dict[str, Any]
    portfolio_params: dict[str, Any]


@dataclass
class CandidateResult:
    candidate: Candidate
    overall_scorecard: dict[str, Any]
    regime_scorecard: list[dict[str, Any]]
    artifacts: dict[str, str] = field(default_factory=dict)
