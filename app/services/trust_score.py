from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrustScore:
    retrieval_confidence: float
    source_agreement: float
    coverage_score: float
    score_spread: float
    overall: float
    level: str  # "high" | "medium" | "low"
    source_count: int

    def to_dict(self) -> dict:
        return {
            "retrieval_confidence": self.retrieval_confidence,
            "source_agreement": self.source_agreement,
            "coverage_score": self.coverage_score,
            "score_spread": self.score_spread,
            "overall": self.overall,
            "level": self.level,
            "source_count": self.source_count,
        }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_trust_score(scores: list[float]) -> TrustScore:
    """Compute trust score from a list of similarity scores."""
    if not scores:
        return TrustScore(
            retrieval_confidence=0.0,
            source_agreement=0.0,
            coverage_score=0.0,
            score_spread=0.0,
            overall=0.0,
            level="low",
            source_count=0,
        )

    n = len(scores)

    # Retrieval Confidence: average similarity score
    retrieval_confidence = _clamp(sum(scores) / n)

    # Source Agreement: 1 - (stdev / 0.5)
    if n == 1:
        source_agreement = 1.0
    else:
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        stdev = math.sqrt(variance)
        source_agreement = _clamp(1.0 - (stdev / 0.5))

    # Coverage Score: fraction of sources with score >= 0.6
    coverage_score = _clamp(sum(1 for s in scores if s >= 0.6) / n)

    # Score Spread: 1 - (max - min)
    if n == 1:
        score_spread = 1.0
    else:
        score_spread = _clamp(1.0 - (max(scores) - min(scores)))

    # Overall: weighted sum
    overall = _clamp(
        0.35 * retrieval_confidence
        + 0.25 * source_agreement
        + 0.25 * coverage_score
        + 0.15 * score_spread
    )

    # Level
    if overall >= 0.8:
        level = "high"
    elif overall >= 0.6:
        level = "medium"
    else:
        level = "low"

    return TrustScore(
        retrieval_confidence=round(retrieval_confidence, 4),
        source_agreement=round(source_agreement, 4),
        coverage_score=round(coverage_score, 4),
        score_spread=round(score_spread, 4),
        overall=round(overall, 4),
        level=level,
        source_count=n,
    )
