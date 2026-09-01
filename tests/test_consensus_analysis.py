import json

import pytest

from consensus import ConsensusAnalyzer


class SequenceLLM:
    def __init__(self, responses, calls=None):
        self.responses = iter(responses)
        self.calls = calls if calls is not None else []

    async def complete_strict(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


class QueueFactory:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self):
        return SequenceLLM([self.responses.pop(0)], self.calls)


PAPER = {
    "id": "p1",
    "title": "Trial",
    "abstract": "Treatment improved outcome in 80 participants.",
}
VALID_DATA = {
    "eligible": True,
    "exclusion_reason": "",
    "study_type": "experimental",
    "population": "participants",
    "intervention": "treatment",
    "comparator": "control",
    "outcome": "outcome",
    "stance": "support",
    "claim": "Treatment improved outcome",
    "effect_direction": "positive",
    "sample_size": "80",
    "statistical_evidence": "not reported",
    "quality": "medium",
    "directness": 1.0,
    "confidence": 0.9,
    "excerpt": "Treatment improved outcome in 80 participants.",
    "limitations": ["abstract_only"],
    "cohort_key": "",
}
VALID = json.dumps(VALID_DATA)


@pytest.mark.asyncio
async def test_invalid_json_is_repaired_once_and_returns_traceable_evidence():
    analyzer = ConsensusAnalyzer(lambda: SequenceLLM(["not-json", VALID]))

    evidence = await analyzer.extract_one(
        "Does treatment improve outcome?", PAPER
    )

    assert evidence.stance == "support"
    assert evidence.sample_size == "80"
    assert evidence.excerpt in PAPER["abstract"]


@pytest.mark.asyncio
async def test_unreported_statistics_remain_unreported():
    analyzer = ConsensusAnalyzer(lambda: SequenceLLM([VALID]))

    evidence = await analyzer.extract_one(
        "Does treatment improve outcome?", PAPER
    )

    assert evidence.statistical_evidence == "not reported"
    assert "abstract_only" in evidence.limitations


@pytest.mark.asyncio
async def test_two_failed_parses_return_visible_unavailable_evidence():
    analyzer = ConsensusAnalyzer(lambda: SequenceLLM(["bad", "still bad"]))

    evidence = await analyzer.extract_one("question", PAPER)

    assert evidence.eligible is False
    assert evidence.stance == "not_applicable"
    assert "extraction_failed" in evidence.limitations
    assert evidence.exclusion_reason


@pytest.mark.asyncio
async def test_untraceable_excerpt_is_removed_and_flagged():
    payload = {**VALID_DATA, "excerpt": "This sentence is fabricated."}
    analyzer = ConsensusAnalyzer(
        lambda: SequenceLLM([json.dumps(payload)])
    )

    evidence = await analyzer.extract_one("question", PAPER)

    assert evidence.excerpt == ""
    assert "untraceable_excerpt" in evidence.limitations


@pytest.mark.asyncio
async def test_duplicate_cohorts_are_downweighted_after_first_record():
    first = json.dumps({**VALID_DATA, "cohort_key": "trial-80"})
    second = json.dumps({**VALID_DATA, "cohort_key": "trial-80"})
    factory = QueueFactory([first, second, "Fixed narrative"])
    analyzer = ConsensusAnalyzer(factory, max_concurrency=1)

    analysis = await analyzer.analyze(
        "question", [PAPER, {**PAPER, "id": "p2", "title": "Follow-up"}]
    )

    assert [item.independence_factor for item in analysis.evidence] == [1.0, 0.25]
    assert "duplicate_cohort" in analysis.evidence[1].limitations
    assert analysis.score.effective_independent_studies == 1.25


@pytest.mark.asyncio
async def test_narrative_receives_fixed_calculated_score():
    papers = [{**PAPER, "id": f"p{i}", "title": f"Trial {i}"} for i in range(8)]
    high_confidence = json.dumps({**VALID_DATA, "quality": "high", "confidence": 1.0})
    factory = QueueFactory([high_confidence] * 8 + ["Narrative based on fixed values"])
    analyzer = ConsensusAnalyzer(factory, max_concurrency=1)

    analysis = await analyzer.analyze("question", papers)

    assert analysis.score.score == 100
    narrative_prompt = factory.calls[-1]["messages"][-1]["content"]
    assert '"score": 100' in narrative_prompt
    assert analysis.narrative == "Narrative based on fixed values"
