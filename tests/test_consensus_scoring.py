from consensus import EvidenceItem, calculate_consensus


def item(stance, *, quality="high", independence=1.0, eligible=True,
         confidence=1.0, directness=1.0, study_type="experimental"):
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


def test_no_directional_evidence_returns_unavailable_not_fifty():
    result = calculate_consensus([item("uncertain")])

    assert result.score is None
    assert result.direction == "insufficient_evidence"


def test_eight_high_quality_independent_supporting_studies_score_one_hundred():
    result = calculate_consensus([item("support") for _ in range(8)])

    assert result.direction == "support"
    assert result.directional_agreement == 1.0
    assert result.evidence_certainty == 1.0
    assert result.score == 100


def test_four_support_and_four_oppose_have_fifty_percent_directional_agreement():
    result = calculate_consensus(
        [item("support") for _ in range(4)]
        + [item("oppose") for _ in range(4)]
    )

    assert result.direction == "mixed"
    assert result.directional_agreement == 0.5
    assert result.score == 50


def test_duplicate_cohort_reduces_effective_study_count():
    result = calculate_consensus(
        [item("support"), item("support", independence=0.25)]
    )

    # 重复队列降低“有效独立研究数”，这与分数解耦
    assert result.effective_independent_studies == 1.25
    # 两篇方向一致且质量/设计满分 → 一致率 100%、确定性 100%
    # 研究数量不足由 apply_min_study_gate 标注，不再在分数里暗中打折
    assert result.directional_agreement == 1.0
    assert result.evidence_certainty == 1.0
    assert result.score == 100


def test_reviews_do_not_contribute_directional_weight():
    review = item("support")
    review = EvidenceItem(**{**review.__dict__, "study_type": "review"})

    result = calculate_consensus([review])

    assert result.score is None


def test_ineligible_studies_remain_counted_but_not_scored():
    result = calculate_consensus([item("support", eligible=False)])

    assert result.score is None
    assert result.raw_counts["support"] == 1
    assert result.raw_counts["excluded"] == 1


def test_evidence_certainty_decreases_with_confidence_directness_and_design_fit():
    baseline = calculate_consensus([item("support") for _ in range(5)])
    lower_confidence = calculate_consensus(
        [item("support", confidence=0.6) for _ in range(5)]
    )
    lower_directness = calculate_consensus(
        [item("support", directness=0.6) for _ in range(5)]
    )
    lower_fit = calculate_consensus(
        [item("support", study_type="unknown") for _ in range(5)]
    )

    assert baseline.evidence_certainty > lower_confidence.evidence_certainty
    assert baseline.evidence_certainty > lower_directness.evidence_certainty
    assert baseline.evidence_certainty > lower_fit.evidence_certainty
