from consensus import EvidenceItem, calculate_consensus


def item(stance, *, quality="high", independence=1.0, eligible=True):
    return EvidenceItem(
        paper_id=stance,
        title=stance,
        eligible=eligible,
        exclusion_reason="",
        study_type="experimental",
        stance=stance,
        claim="claim",
        excerpt="evidence",
        quality=quality,
        directness=1.0,
        confidence=1.0,
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

    assert result.effective_independent_studies == 1.25
    assert result.score == 16


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
