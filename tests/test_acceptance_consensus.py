"""ChatGPT 规划验收测试：<5研究门控、Meta去重、问题类型识别、确定性统计。"""
import pytest

from consensus import (
    ConsensusScore,
    EvidenceItem,
    QuestionSpec,
    apply_min_study_gate,
    calculate_consensus,
    heuristic_question_type,
    is_meter_displayable,
    needs_search_expansion,
)


def item(stance, *, quality="high", independence=1.0, eligible=True,
         study_type="experimental", confidence=1.0, directness=1.0):
    return EvidenceItem(
        paper_id=stance,
        title=stance,
        eligible=eligible,
        exclusion_reason="",
        study_type=study_type,
        stance=stance,
        claim="claim",
        excerpt="evidence",
        quality=quality,
        directness=directness,
        confidence=confidence,
        independence_factor=independence,
        limitations=("abstract_only",),
    )


# === 少于5篇研究 → 标记证据不足，但保留统计量（用户决策：不足=搜索策略问题）===
def test_four_supporting_studies_are_gated_to_insufficient():
    evidence = [item("support") for _ in range(4)]
    score = calculate_consensus(evidence)
    gated = apply_min_study_gate(score, evidence, minimum=5)

    assert score.score == 100  # 确定性：4 篇满分研究，确定性不再被数量打折
    assert gated.direction == "insufficient_evidence"  # 标记为不足
    assert gated.score == 100  # 统计量保留，不再销毁成 None（避免白屏）
    assert not is_meter_displayable(gated, minimum=5)  # 但不展示 Meter 数字
    assert needs_search_expansion(gated, minimum=5)  # 应扩充检索


def test_five_supporting_studies_pass_the_gate():
    evidence = [item("support") for _ in range(5)]
    score = calculate_consensus(evidence)
    gated = apply_min_study_gate(score, evidence, minimum=5)

    assert gated.score is not None
    assert gated.direction == "support"
    assert is_meter_displayable(gated, minimum=5)
    assert not needs_search_expansion(gated, minimum=5)


def test_duplicate_fraction_does_not_satisfy_five_independent_study_gate():
    evidence = [item("support") for _ in range(4)] + [
        item("support", independence=0.25)
    ]
    gated = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)

    assert gated.effective_independent_studies == 4.25
    assert gated.direction == "insufficient_evidence"
    assert not is_meter_displayable(gated, minimum=5)
    assert needs_search_expansion(gated, minimum=5)


def test_low_confidence_records_do_not_qualify_for_meter():
    evidence = [item("support", confidence=0.01) for _ in range(5)]
    score = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)

    assert score.direction == "insufficient_evidence"
    assert not is_meter_displayable(score, minimum=5)
    assert needs_search_expansion(score, minimum=5)


def test_meter_qualified_counts_ignore_excluded_records_and_weight_duplicates():
    import consensus

    count_fn = getattr(consensus, "qualified_stance_counts", None)
    assert callable(count_fn), "qualified Meter distribution API is missing"
    evidence = [item("support", eligible=False) for _ in range(4)] + [
        item("oppose"), item("oppose", independence=0.25)
    ]

    counts = count_fn(evidence)

    assert counts["support"] == 0.0
    assert counts["oppose"] == 1.25
    assert counts["effective_independent_studies"] == 1.25


# === Meta 分析不参与独立计票 ===
def test_meta_analysis_does_not_count_as_independent_vote():
    review = item("support", study_type="meta_analysis")
    result = calculate_consensus([review])

    assert result.score is None
    assert result.effective_independent_studies == 0.0


def test_meta_analysis_and_its_trials_do_not_double_count():
    # 2 篇原始试验 + 1 篇 Meta 分析（纳入同样试验）
    trials = [item("support") for _ in range(2)]
    meta = item("support", study_type="meta_analysis")
    result = calculate_consensus(trials + [meta])

    # Meta 分析权重为 0，只计 2 篇独立试验
    assert result.effective_independent_studies == 2.0


# === 不确定证据不当 50% ===
def test_uncertain_evidence_never_becomes_neutral_support():
    result = calculate_consensus([item("uncertain")])

    assert result.score is None
    assert result.weighted_support == 0.0
    assert result.weighted_oppose == 0.0


# === 确定性：同一输入永远同一结果 ===
def test_same_evidence_input_yields_identical_score():
    evidence = [item("support") for _ in range(6)] + [item("oppose") for _ in range(2)]
    first = calculate_consensus(evidence)
    second = calculate_consensus(evidence)

    assert first.score == second.score
    assert first.directional_agreement == second.directional_agreement
    assert first.evidence_certainty == second.evidence_certainty


# === 问题类型识别 ===
@pytest.mark.parametrize("question,expected", [
    ("是否退火能改善 GaInP/GaAs HBT 性能？", "yes_no"),
    ("Does BERT outperform LSTM?", "yes_no"),
    ("Is exercise effective for depression?", "yes_no"),
    ("比较不同 MOCVD 生长工艺的优劣", "open_ended"),
    ("梳理 HBT 退火机制的研究现状", "open_ended"),
])
def test_heuristic_question_type(question, expected):
    assert heuristic_question_type(question) == expected


def test_question_spec_fallback_uses_heuristic():
    spec = QuestionSpec.fallback("Does X improve Y?", "yes_no")

    assert spec.question_type == "yes_no"
    assert spec.population == "not reported"


def test_complex_question_detection_catches_multiple_english_outcomes():
    from consensus import is_complex_question

    assert is_complex_question("Does X improve outcome A and reduce outcome B?")
    assert not is_complex_question("Does X improve outcome A?")


# === 重复队列不重复计数 ===
def test_duplicate_cohort_halves_independence():
    evidence = [
        item("support", independence=1.0),
        item("support", independence=0.25),  # 同队列后续论文
    ]
    result = calculate_consensus(evidence)

    assert result.effective_independent_studies == 1.25


# === 证据不足 → 触发“扩充检索”，而不是白屏或伪造结论（用户决策）===
def test_insufficient_evidence_requests_search_expansion():
    evidence = [item("support") for _ in range(3)]
    score = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)

    assert needs_search_expansion(score)
    # 关键：仍然保留立场分布与统计量，页面不会“什么都没有”
    assert score.raw_counts["support"] == 3
    assert score.score is not None


def test_uncertain_evidence_also_requests_search_expansion():
    """不确定证据不产生分数，同样应提示扩充检索而非静默失败。"""
    score = calculate_consensus([item("uncertain") for _ in range(6)])

    assert score.score is None
    assert needs_search_expansion(score)


def test_sufficient_evidence_does_not_request_expansion():
    evidence = [item("support") for _ in range(6)]
    score = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)

    assert not needs_search_expansion(score)
    assert is_meter_displayable(score, minimum=5)


def test_meta_only_evidence_is_not_meter_displayable():
    """Meta 分析权重为 0，不能单独支撑 Meter，应提示扩充原始研究。"""
    evidence = [item("support", study_type="meta_analysis") for _ in range(6)]
    score = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)

    assert not is_meter_displayable(score, minimum=5)
    assert needs_search_expansion(score)


# === 证据确定性只反映“证据有多强”，不再暗中按研究数量打折 ===
@pytest.mark.parametrize("count", [1, 2, 4, 8, 16])
def test_certainty_is_independent_of_study_count(count):
    evidence = [item("support") for _ in range(count)]
    result = calculate_consensus(evidence)

    assert result.directional_agreement == 1.0
    assert result.evidence_certainty == 1.0
    assert result.score == 100


# === 证据不足时，叙述不得把不可靠数字当作结论 ===
def test_narrative_flags_insufficient_evidence_instead_of_asserting_score():
    from consensus import ConsensusAnalyzer

    evidence = [item("support") for _ in range(3)]
    score = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)
    assert score.score == 100  # 统计量仍在

    text = ConsensusAnalyzer._fallback_narrative(score)

    assert "证据不足" in text
    assert "扩充检索" in text
    # 不得出现“共识分为 X/100”这类定论式表述
    assert "共识分为" not in text
    # 不得把内部的 insufficient_evidence 标签直接暴露给用户
    assert "insufficient_evidence" not in text


def test_narrative_reports_score_when_evidence_is_sufficient():
    from consensus import ConsensusAnalyzer

    evidence = [item("support") for _ in range(6)]
    score = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)
    text = ConsensusAnalyzer._fallback_narrative(score)

    assert "共识分为 100/100" in text
    assert "证据不足" not in text


def test_ineligible_and_low_confidence_studies_do_not_count_toward_gate():
    evidence = [item("support", eligible=False) for _ in range(5)]
    evidence += [item("support", confidence=0.49) for _ in range(5)]
    score = apply_min_study_gate(calculate_consensus(evidence), evidence, minimum=5)

    assert score.raw_counts["support"] == 10
    assert score.raw_counts["excluded"] == 5
    assert score.direction == "insufficient_evidence"
