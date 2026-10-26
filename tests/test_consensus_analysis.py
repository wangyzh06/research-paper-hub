import json

import pytest

from consensus import ConsensusAnalyzer
from engine import LLMRequestError


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
    assert evidence.claim == ""


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
    from consensus import QuestionSpec

    papers = [{**PAPER, "id": f"p{i}", "title": f"Trial {i}"} for i in range(8)]
    high_confidence = json.dumps({**VALID_DATA, "quality": "high", "confidence": 1.0})
    factory = QueueFactory([high_confidence] * 8 + ["Narrative based on fixed values"])
    analyzer = ConsensusAnalyzer(factory, max_concurrency=1)

    analysis = await analyzer.analyze(
        "Does treatment improve outcome?",
        papers,
        question_spec=QuestionSpec.fallback(
            "Does treatment improve outcome?", "yes_no"
        ),
        subquestions=[],
    )

    assert analysis.score.score == 100
    narrative_prompt = factory.calls[-1]["messages"][-1]["content"]
    assert '"score": 100' in narrative_prompt
    assert analysis.narrative == "Narrative based on fixed values"


@pytest.mark.asyncio
async def test_open_ended_prompt_has_no_meter_score_and_excludes_rejected_references():
    from consensus import QuestionSpec

    rejected = json.dumps({**VALID_DATA, "eligible": False, "exclusion_reason": "off topic"})
    factory = QueueFactory([rejected, VALID, "Open-ended evidence review"])
    analyzer = ConsensusAnalyzer(factory, max_concurrency=1)
    papers = [
        {**PAPER, "id": "p2", "title": "Excluded trial"},
        PAPER,
    ]

    await analyzer.analyze(
        "Compare treatment effects and limitations",
        papers,
        question_spec=QuestionSpec.fallback(
            "Compare treatment effects and limitations", "open_ended"
        ),
        subquestions=[],
    )

    prompt = factory.calls[-1]["messages"][-1]["content"]
    reference_list = prompt.partition("参考文献列表：\n")[2]
    included_input = prompt.partition(
        "纳入论文（编号 → 标题/类型/立场/质量）：\n"
    )[2].partition("\n\n参考文献列表：\n")[0]
    assert '"score":' not in prompt
    assert "directional_agreement" not in prompt
    assert "性能指标1" not in prompt
    assert "| 3 | 2 |" not in prompt
    assert "[1] Trial (experimental)" in reference_list
    assert "Excluded trial" not in reference_list
    assert "Excluded trial" not in included_input


@pytest.mark.asyncio
async def test_analyze_preserves_heuristic_question_type_when_spec_is_missing():
    analyzer = ConsensusAnalyzer(QueueFactory([VALID, "Narrative"]))

    analysis = await analyzer.analyze("Does treatment improve outcome?", [PAPER])

    assert analysis.question_spec.question_type == "yes_no"


def test_deterministic_fallback_numbers_only_included_evidence():
    from consensus import calculate_consensus

    excluded = ConsensusAnalyzer._unavailable(
        {**PAPER, "id": "excluded", "title": "Excluded"}, "off topic"
    )
    included = ConsensusAnalyzer(lambda: None)._parse_evidence(VALID, PAPER)
    score = calculate_consensus([included])

    report = ConsensusAnalyzer._deterministic_narrative(
        [excluded, included], score, "Does treatment improve outcome?", question_type="yes_no"
    )

    assert "- [1] **Trial**" in report
    assert "- [2] **Trial**" not in report


@pytest.mark.asyncio
async def test_offline_mixed_evidence_regression_returns_literal_weighted_score():
    def encoded(**changes):
        return json.dumps({**VALID_DATA, **changes})

    responses = [
        encoded(stance="support", quality="high", confidence=1.0, cohort_key="c1"),
        encoded(stance="support", quality="high", confidence=1.0, cohort_key="c1"),
        encoded(stance="oppose", quality="medium", confidence=1.0),
        encoded(stance="uncertain", quality="high", confidence=1.0),
        encoded(
            eligible=False,
            exclusion_reason="review article",
            study_type="review",
            stance="support",
            quality="high",
            confidence=1.0,
        ),
        "Fixed integration narrative",
    ]
    papers = [{**PAPER, "id": f"mix-{i}", "title": f"Evidence {i}"} for i in range(5)]
    analyzer = ConsensusAnalyzer(QueueFactory(responses), max_concurrency=1)

    analysis = await analyzer.analyze("question", papers)

    # 队列 c1 的第二篇被降权为 0.25；uncertain 不计入加权；review 设计权重为 0
    # support = 0.9*1.0*1.0*(1.0 + 0.25) = 1.125
    # oppose  = 0.9*1.0*1.0*1.0*0.75(medium) = 0.675
    assert analysis.score.weighted_support == pytest.approx(1.125)
    assert analysis.score.weighted_oppose == pytest.approx(0.675)
    assert analysis.score.effective_independent_studies == pytest.approx(2.25)
    # 一致性 1.125/1.8 = 0.625；确定性为质量加权 0.8/0.9 = 0.8889
    assert analysis.score.score == 56
    # 仅 3 篇合格方向性证据（<5）→ 标记证据不足，但统计量保留、不销毁
    assert analysis.score.direction == "insufficient_evidence"
    assert analysis.score.weighted_support == pytest.approx(1.125)
    assert analysis.score.raw_counts["support"] == 3
    assert analysis.evidence[4].eligible is False


@pytest.mark.asyncio
async def test_rate_limit_opens_circuit_instead_of_retrying_every_paper():
    factory = QueueFactory(
        [LLMRequestError("glm", "rate_limit", "too many requests")]
    )
    analyzer = ConsensusAnalyzer(factory, max_concurrency=1)

    analysis = await analyzer.analyze("question", [PAPER, {**PAPER, "id": "p2"}])

    assert len(factory.calls) == 1
    assert all(not item.eligible for item in analysis.evidence)
    assert analysis.score.score is None


@pytest.mark.asyncio
async def test_narrative_fallback_uses_evidence_without_static_research_counts():
    analyzer = ConsensusAnalyzer(
        QueueFactory([VALID, RuntimeError("offline")]), max_concurrency=1
    )

    analysis = await analyzer.analyze("Does treatment improve outcome?", [PAPER])

    assert "Treatment improved outcome" in analysis.narrative
    assert "核心指标优化" not in analysis.narrative
    assert "| 3 | 2 |" not in analysis.narrative
    assert "建议在工艺实施中优先采用" not in analysis.narrative


def test_deterministic_fallback_omits_claim_without_traceable_excerpt():
    from consensus import ConsensusAnalyzer, EvidenceItem, calculate_consensus

    evidence = ConsensusAnalyzer._unavailable(PAPER, "no abstract available")
    evidence = EvidenceItem(
        **{
            **evidence.__dict__,
            "eligible": True,
            "stance": "support",
            "claim": "An unverified claim",
            "excerpt": "",
        }
    )
    score = calculate_consensus([evidence])

    text = ConsensusAnalyzer._deterministic_narrative(
        [evidence], score, "Does X improve Y?", question_type="yes_no"
    )

    assert "An unverified claim" not in text


@pytest.mark.asyncio
async def test_complex_question_decomposition_is_bounded_and_falls_back_safely():
    from consensus import decompose_question

    decomposition_json = json.dumps(
        {"subquestions": [f"Subquestion {i}?" for i in range(1, 11)]}
    )
    parts = await decompose_question(
        "Compare annealing methods, defects, and reliability trade-offs",
        QueueFactory([decomposition_json]),
        max_subquestions=8,
    )
    fallback = await decompose_question(
        "Compare annealing methods, defects, and reliability trade-offs",
        QueueFactory([RuntimeError("boom")]),
    )

    assert 3 <= len(parts) <= 8
    assert fallback == [
        "Compare annealing methods, defects, and reliability trade-offs"
    ]


@pytest.mark.asyncio
async def test_analysis_populates_claim_evidence_graph_and_subquestions():
    subquestions = json.dumps(
        {"subquestions": ["How does X affect A?", "How does X affect B?", "What limits X?"]}
    )
    second_valid = json.dumps(
        {**VALID_DATA, "stance": "oppose", "claim": "Treatment worsened B"}
    )
    analyzer = ConsensusAnalyzer(
        QueueFactory([subquestions, VALID, second_valid, "Synthesis"]),
        max_concurrency=1,
    )
    papers = [PAPER, {**PAPER, "id": "p2", "title": "Second Trial"}]

    analysis = await analyzer.analyze(
        "Compare how treatment affects A and B and summarize limitations",
        papers,
    )

    assert list(analysis.subquestions) == [
        "How does X affect A?",
        "How does X affect B?",
        "What limits X?",
    ]
    claims = analysis.evidence_review["claims"]
    assert claims
    linked_ids = {
        pid
        for claim in claims
        for key in ("supporting_papers", "opposing_papers", "limiting_papers")
        for pid in claim[key]
    }
    assert linked_ids == {"p1", "p2"}


@pytest.mark.asyncio
async def test_subquestion_reviews_use_isolated_paper_sets():
    subquestion_a = {**PAPER, "id": "sub-a", "title": "Evidence A"}
    subquestion_b = {**PAPER, "id": "sub-b", "title": "Evidence B"}
    analyzer = ConsensusAnalyzer(
        QueueFactory([VALID, VALID, VALID, VALID, "Combined synthesis"]),
        max_concurrency=1,
    )

    analysis = await analyzer.analyze(
        "Compare results for outcomes A and B",
        [PAPER, {**PAPER, "id": "main-2"}],
        subquestions=["Outcome A?", "Outcome B?", "Limitations?"],
        subquestion_papers={
            "Outcome A?": [subquestion_a],
            "Outcome B?": [subquestion_b],
            "Limitations?": [],
        },
        subquestion_search_stats={
            "Outcome A?": {"retrieved": 6, "screened": 1},
            "Outcome B?": {"retrieved": 9, "screened": 1},
            "Limitations?": {"retrieved": 0, "screened": 0},
        },
    )

    reviews = {row["question"]: row for row in analysis.subquestion_reviews}
    assert reviews["Outcome A?"]["paper_ids"] == ["sub-a"]
    assert reviews["Outcome B?"]["paper_ids"] == ["sub-b"]
    assert reviews["Limitations?"]["paper_ids"] == []
    assert reviews["Outcome A?"]["search_stats"] == {"retrieved": 6, "screened": 1}
    assert reviews["Outcome B?"]["search_stats"] == {"retrieved": 9, "screened": 1}
    assert set(reviews["Outcome A?"]["paper_ids"]).isdisjoint(
        reviews["Outcome B?"]["paper_ids"]
    )
