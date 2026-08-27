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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            candidate_id=data["candidate_id"],
            algorithm_class=data["algorithm_class"],
            portfolio_class=data["portfolio_class"],
            algorithm_params=data["algorithm_params"],
            portfolio_params=data["portfolio_params"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "algorithm_class": self.algorithm_class,
            "portfolio_class": self.portfolio_class,
            "algorithm_params": self.algorithm_params,
            "portfolio_params": self.portfolio_params,
        }


@dataclass
class CandidateResult:
    candidate: Candidate
    overall_scorecard: dict[str, Any]
    regime_scorecard: list[dict[str, Any]]
    artifacts: dict[str, str] = field(default_factory=dict)
