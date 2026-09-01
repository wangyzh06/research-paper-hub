"""Auditable evidence records and deterministic consensus scoring."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, replace
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


@dataclass(frozen=True)
class ConsensusAnalysis:
    evidence: tuple[EvidenceItem, ...]
    score: ConsensusScore
    narrative: str


class ConsensusAnalyzer:
    """Use an LLM for bounded evidence extraction, never for score calculation."""

    def __init__(self, llm_factory, max_concurrency: int = 4):
        self.llm_factory = llm_factory
        self.max_concurrency = max(1, max_concurrency)
        self._stop_reason: str | None = None

    @staticmethod
    def _unavailable(paper: dict, reason: str) -> EvidenceItem:
        return EvidenceItem(
            paper_id=str(paper.get("id") or paper.get("doi") or paper.get("title") or ""),
            title=paper.get("title") or "Untitled",
            eligible=False,
            exclusion_reason=reason,
            study_type="unknown",
            stance="not_applicable",
            claim="",
            excerpt="",
            quality="unknown",
            directness=0.0,
            confidence=0.0,
            independence_factor=1.0,
            limitations=("abstract_only", "extraction_failed"),
        )

    @staticmethod
    async def _close_client(client) -> None:
        close = getattr(client, "aclose", None)
        if close:
            try:
                await close()
            except Exception:
                pass

    @staticmethod
    def _text(payload: dict, key: str, default: str = "not reported") -> str:
        value = payload.get(key)
        if value is None or value == "":
            return default
        return str(value).strip() or default

    def _parse_evidence(self, raw: str, paper: dict) -> EvidenceItem:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("evidence response must be a JSON object")
        eligible = payload.get("eligible", False)
        if not isinstance(eligible, bool):
            raise ValueError("eligible must be boolean")

        study_type = self._text(payload, "study_type", "unknown").casefold()
        stance = self._text(payload, "stance", "uncertain").casefold()
        quality = self._text(payload, "quality", "unknown").casefold()
        if study_type not in STUDY_WEIGHTS:
            raise ValueError("invalid study_type")
        if stance not in STANCE_VALUES:
            raise ValueError("invalid stance")
        if quality not in QUALITY_WEIGHTS:
            raise ValueError("invalid quality")

        limitations_value = payload.get("limitations", [])
        if not isinstance(limitations_value, list):
            raise ValueError("limitations must be an array")
        limitations = [str(value).strip() for value in limitations_value if str(value).strip()]
        if "abstract_only" not in limitations:
            limitations.append("abstract_only")

        abstract = paper.get("abstract") or ""
        excerpt = self._text(payload, "excerpt", "")
        if excerpt and excerpt not in abstract:
            excerpt = ""
            limitations.append("untraceable_excerpt")

        return EvidenceItem(
            paper_id=str(paper.get("id") or paper.get("doi") or paper.get("title") or ""),
            title=paper.get("title") or "Untitled",
            eligible=eligible,
            exclusion_reason=self._text(payload, "exclusion_reason", ""),
            study_type=study_type,
            stance=stance,
            claim=self._text(payload, "claim", ""),
            excerpt=excerpt,
            quality=quality,
            directness=_clamp(payload.get("directness", 0.0)),
            confidence=_clamp(payload.get("confidence", 0.0)),
            independence_factor=1.0,
            limitations=tuple(dict.fromkeys(limitations)),
            population=self._text(payload, "population"),
            intervention=self._text(payload, "intervention"),
            comparator=self._text(payload, "comparator"),
            outcome=self._text(payload, "outcome"),
            effect_direction=self._text(payload, "effect_direction"),
            sample_size=self._text(payload, "sample_size"),
            statistical_evidence=self._text(payload, "statistical_evidence"),
            cohort_key=self._text(payload, "cohort_key", ""),
        )

    async def extract_one(self, question: str, paper: dict) -> EvidenceItem:
        abstract = paper.get("abstract") or ""
        if not abstract.strip():
            return self._unavailable(paper, "no abstract available")
        if self._stop_reason:
            return self._unavailable(paper, self._stop_reason)

        schema = (
            "Return one JSON object with keys: eligible (boolean), exclusion_reason, "
            "study_type (experimental|observational|simulation|theory|review|"
            "systematic_review|meta_analysis|commentary|unknown), population, "
            "intervention, comparator, outcome, stance "
            "(support|oppose|mixed|uncertain|not_applicable), claim, effect_direction, "
            "sample_size, statistical_evidence, quality (high|medium|low|unknown), "
            "directness (0..1), confidence (0..1), excerpt, limitations (array), "
            "and cohort_key. Use 'not reported' for facts absent from the abstract. "
            "The excerpt must be copied exactly from the supplied abstract."
        )
        messages = [
            {"role": "system", "content": "Extract evidence conservatively from abstracts only. " + schema},
            {
                "role": "user",
                "content": (
                    f"Research question: {question}\nTitle: {paper.get('title', '')}\n"
                    f"Abstract: {abstract}"
                ),
            },
        ]
        client = self.llm_factory()
        if (
            hasattr(client, "api_key")
            and not client.api_key
            and getattr(client, "provider", "") != "ollama"
        ):
            self._stop_reason = "missing LLM API key"
            await self._close_client(client)
            return self._unavailable(paper, self._stop_reason)
        last_error = "invalid structured response"
        invalid_response = ""
        try:
            for attempt in range(2):
                request_messages = messages
                if attempt:
                    request_messages = [
                        {"role": "system", "content": "Repair the response as strict JSON. " + schema},
                        {"role": "user", "content": f"Invalid response:\n{invalid_response}"},
                    ]
                try:
                    raw = await client.complete_strict(
                        request_messages,
                        temperature=0,
                        response_format={"type": "json_object"},
                    )
                    invalid_response = str(raw)
                    return self._parse_evidence(invalid_response, paper)
                except Exception as error:
                    category = getattr(error, "category", "")
                    last_error = (
                        f"evidence extraction failed: {category or type(error).__name__}"
                    )
                    if category in {"authentication", "rate_limit", "invalid_configuration"}:
                        self._stop_reason = last_error
                        break
            return self._unavailable(paper, last_error)
        finally:
            await self._close_client(client)

    @staticmethod
    def _apply_duplicate_cohort_weights(
        evidence: Sequence[EvidenceItem],
    ) -> tuple[EvidenceItem, ...]:
        seen = set()
        adjusted = []
        for item in evidence:
            key = item.cohort_key.strip().casefold()
            if key and key in seen:
                limitations = tuple(dict.fromkeys((*item.limitations, "duplicate_cohort")))
                item = replace(item, independence_factor=0.25, limitations=limitations)
            elif key:
                seen.add(key)
            adjusted.append(item)
        return tuple(adjusted)

    @staticmethod
    def _fallback_narrative(score: ConsensusScore) -> str:
        if score.score is None:
            return "现有摘要未提供足够的方向性证据，无法计算共识度。"
        return (
            f"主要证据方向为 {score.direction}；方向一致率为 "
            f"{score.directional_agreement:.1%}，证据确定性为 "
            f"{score.evidence_certainty:.1%}。共识分为 {score.score}/100。"
        )

    async def analyze(self, question: str,
                      papers: Sequence[dict]) -> ConsensusAnalysis:
        self._stop_reason = None
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def bounded(paper):
            async with semaphore:
                if self._stop_reason:
                    return self._unavailable(paper, self._stop_reason)
                return await self.extract_one(question, paper)

        extracted = await asyncio.gather(*(bounded(paper) for paper in papers))
        evidence = self._apply_duplicate_cohort_weights(extracted)
        score = calculate_consensus(evidence)
        fixed_values = asdict(score)
        evidence_summary = [
            {
                "title": item.title,
                "eligible": item.eligible,
                "exclusion_reason": item.exclusion_reason,
                "study_type": item.study_type,
                "stance": item.stance,
                "claim": item.claim,
                "quality": item.quality,
                "limitations": item.limitations,
            }
            for item in evidence
        ]
        prompt = (
            "Write a concise Chinese evidence synthesis. The following calculated "
            "values are immutable: do not recalculate, replace, or infer another score. "
            "State the abstract-only limitation and distinguish unavailable facts.\n"
            + json.dumps(fixed_values, ensure_ascii=False, sort_keys=True)
            + "\nExtracted evidence:\n"
            + json.dumps(evidence_summary, ensure_ascii=False)
        )
        narrative = self._fallback_narrative(score)
        if not self._stop_reason:
            client = self.llm_factory()
            try:
                generated = await client.complete_strict(
                    [
                        {"role": "system", "content": "Synthesize evidence without changing supplied metrics."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                if str(generated).strip():
                    narrative = str(generated).strip()
            except Exception:
                pass
            finally:
                await self._close_client(client)
        return ConsensusAnalysis(evidence, score, narrative)
