"""Auditable evidence records and deterministic consensus scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


STANCE_VALUES = {
    "support",
    "oppose",
    "mixed",
    "uncertain",
    "not_applicable",
}
STUDY_WEIGHTS = {
    "experimental": 1.0,
    "observational": 0.8,
    "simulation": 0.6,
    "theory": 0.6,
    "unknown": 0.4,
    "review": 0.0,
    "systematic_review": 0.0,
    "meta_analysis": 0.0,
    "commentary": 0.0,
}
QUALITY_WEIGHTS = {"high": 1.0, "medium": 0.75, "low": 0.4, "unknown": 0.5}


@dataclass(frozen=True)
class EvidenceItem:
    paper_id: str
    title: str
    eligible: bool
    exclusion_reason: str
    study_type: str
    stance: str
    claim: str
    excerpt: str
    quality: str
    directness: float
    confidence: float
    independence_factor: float
    limitations: tuple[str, ...]
    population: str = "not reported"
    intervention: str = "not reported"
    comparator: str = "not reported"
    outcome: str = "not reported"
    effect_direction: str = "not reported"
    sample_size: str = "not reported"
    statistical_evidence: str = "not reported"
    cohort_key: str = ""


@dataclass(frozen=True)
class ConsensusScore:
    score: int | None
    direction: str
    directional_agreement: float
    evidence_certainty: float
    effective_independent_studies: float
    weighted_support: float
    weighted_oppose: float
    weighted_mixed: float
    raw_counts: dict[str, int]

    @classmethod
    def insufficient(cls, raw_counts: dict[str, int]) -> "ConsensusScore":
        return cls(
            score=None,
            direction="insufficient_evidence",
            directional_agreement=0.0,
            evidence_certainty=0.0,
            effective_independent_studies=0.0,
            weighted_support=0.0,
            weighted_oppose=0.0,
            weighted_mixed=0.0,
            raw_counts=raw_counts,
        )


def _clamp(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _raw_counts(items: Sequence[EvidenceItem]) -> dict[str, int]:
    counts = {stance: 0 for stance in sorted(STANCE_VALUES)}
    counts["excluded"] = 0
    for item in items:
        stance = item.stance if item.stance in STANCE_VALUES else "uncertain"
        counts[stance] += 1
        if not item.eligible:
            counts["excluded"] += 1
    return counts


def calculate_consensus(items: Sequence[EvidenceItem]) -> ConsensusScore:
    """Calculate a reproducible score; uncertain evidence is never treated as 50%."""
    raw_counts = _raw_counts(items)
    support = 0.0
    oppose = 0.0
    mixed_weight = 0.0
    quality_numerator = 0.0
    quality_denominator = 0.0
    effective_studies = 0.0

    for item in items:
        if not item.eligible or item.stance not in {"support", "oppose", "mixed"}:
            continue
        study_weight = STUDY_WEIGHTS.get(item.study_type, STUDY_WEIGHTS["unknown"])
        if study_weight <= 0:
            continue
        quality_weight = QUALITY_WEIGHTS.get(item.quality, QUALITY_WEIGHTS["unknown"])
        directness = _clamp(item.directness)
        confidence = _clamp(item.confidence)
        independence = _clamp(item.independence_factor)
        base_weight = study_weight * directness * confidence * independence
        weight = base_weight * quality_weight

        quality_numerator += weight
        quality_denominator += base_weight
        effective_studies += independence
        if item.stance == "support":
            support += weight
        elif item.stance == "oppose":
            oppose += weight
        else:
            mixed_weight += weight
            support += weight / 2
            oppose += weight / 2

    directional_total = support + oppose
    if directional_total <= 0:
        return ConsensusScore.insufficient(raw_counts)

    agreement = max(support, oppose) / directional_total
    weighted_quality = (
        quality_numerator / quality_denominator if quality_denominator else 0.0
    )
    certainty = weighted_quality * min(1.0, effective_studies / 8.0)
    if support > oppose:
        direction = "support"
    elif oppose > support:
        direction = "oppose"
    else:
        direction = "mixed"

    return ConsensusScore(
        score=round(100 * agreement * certainty),
        direction=direction,
        directional_agreement=agreement,
        evidence_certainty=certainty,
        effective_independent_studies=effective_studies,
        weighted_support=support,
        weighted_oppose=oppose,
        weighted_mixed=mixed_weight,
        raw_counts=raw_counts,
    )
