"""第 4/5/6 批功能测试：Meter Snapshot、设计适配度、Evidence Review、SciFact 标签。

设计依据：
- Consensus 官方《证据分级指南》明确指出证据等级**不是**普适的质量排序，
  "最合适的研究设计取决于所问的问题"——因此权重必须按问题域区分。
- Consensus 官方 Snapshot 含四项指标（Recency/Methods/Journals/Citations）。
  本项目无 SJR 数据源，因此只用前三项 + 期刊名列表，绝不伪造影响因子。
- 开放问题的论文之间不是投票关系，因此 Evidence Review **不得产出百分比**。
"""
from __future__ import annotations

import pytest

from consensus import (
    DOMAIN_WEIGHTS,
    REVIEW_STUDY_TYPES,
    SCIFACT_LABELS,
    EvidenceItem,
    build_evidence_review,
    build_snapshot,
    calculate_consensus,
    domain_weights,
    scifact_label,
    study_weight_for,
)


def item(pid="p1", stance="support", study_type="bench_test", *, quality="high",
         eligible=True, limitations=(), population="GaAs HBT",
         intervention="annealing", outcome="gain", direction="increase"):
    return EvidenceItem(
        paper_id=pid, title=f"Paper {pid}", eligible=eligible, exclusion_reason="",
        study_type=study_type, stance=stance, claim=f"claim {pid}", excerpt="excerpt",
        quality=quality, directness=1.0, confidence=1.0, independence_factor=1.0,
        limitations=tuple(limitations), population=population,
        intervention=intervention, outcome=outcome, effect_direction=direction,
    )


# ============ 第 5 批：设计适配度 ============
def test_domain_weights_cover_all_domains():
    assert set(DOMAIN_WEIGHTS) == {"clinical", "materials_device", "general"}


@pytest.mark.parametrize("alias,expected", [
    ("biomedicine", "clinical"),
    ("medicine", "clinical"),
    ("cs_physics", "materials_device"),
    ("physics", "materials_device"),
    ("materials", "materials_device"),
    ("general", "general"),
    ("unknown_domain", "general"),
])
def test_domain_weights_aliases(alias, expected):
    assert domain_weights(alias) is domain_weights(expected)


def test_rct_is_preferred_in_clinical_but_not_in_materials():
    """这是官方指南的核心：设计适配度依问题而定。"""
    assert study_weight_for("rct", "clinical") == 1.0
    assert study_weight_for("rct", "cs_physics") < 0.3


def test_bench_test_is_preferred_in_materials_but_not_in_clinical():
    assert study_weight_for("bench_test", "cs_physics") == 1.0
    assert study_weight_for("bench_test", "clinical") < 0.4


def test_reviews_never_count_in_any_domain():
    """综述/Meta 分析不得参与独立计票（避免与其纳入的原始研究重复计数）。"""
    for study_type in REVIEW_STUDY_TYPES:
        for domain in DOMAIN_WEIGHTS:
            assert study_weight_for(study_type, domain) == 0.0, (
                f"{study_type} 在 {domain} 域下权重应为 0"
            )


def test_unknown_study_type_has_low_weight():
    assert study_weight_for("totally_made_up", "general") < 0.5


def test_domain_changes_consensus_score():
    """同一批证据在不同问题域下应得到不同分数（证明域权重真的生效）。"""
    evidence = [
        item("p1", "support", "rct"),
        item("p2", "support", "bench_test"),
        item("p3", "oppose", "in_vitro"),
    ]
    clinical = calculate_consensus(evidence, domain="clinical")
    materials = calculate_consensus(evidence, domain="cs_physics")

    assert clinical.weighted_support != materials.weighted_support


def test_calculate_consensus_defaults_to_general_domain():
    evidence = [item("p1")]
    assert calculate_consensus(evidence).score == calculate_consensus(evidence, domain="general").score


# ============ 第 4 批：Meter Snapshot ============
def _snapshot_fixture():
    evidence = [
        item("p1", "support", "bench_test"),
        item("p2", "support", "experimental"),
        item("p3", "oppose", "simulation"),
        item("p4", "uncertain", "theory"),
    ]
    papers = [
        {"id": "p1", "year": 2020, "citation_count": 50, "venue": "JAP"},
        {"id": "p2", "year": 2022, "citation_count": 30, "venue": "IEEE TED"},
        {"id": "p3", "year": 2019, "citation_count": 10, "venue": "APL"},
        {"id": "p4", "year": 2021, "citation_count": 2, "venue": "JAP"},
    ]
    return evidence, papers


def test_snapshot_groups_by_stance():
    evidence, papers = _snapshot_fixture()
    snap = build_snapshot(evidence, papers)

    assert snap["by_stance"]["support"]["count"] == 2
    assert snap["by_stance"]["oppose"]["count"] == 1
    assert snap["by_stance"]["uncertain"]["count"] == 1
    assert snap["by_stance"]["mixed"]["count"] == 0


def test_snapshot_computes_recency():
    evidence, papers = _snapshot_fixture()
    snap = build_snapshot(evidence, papers)

    assert snap["by_stance"]["support"]["avg_year"] == 2021.0
    # overall 只统计**方向性**证据（support/oppose/mixed），
    # 因此 uncertain 的 p4(2021) 不计入：(2020+2022+2019)/3 = 2020.33，
    # 结果按 1 位小数取整（展示用）
    assert snap["overall"]["count"] == 3
    assert snap["overall"]["avg_year"] == pytest.approx(2020.3, abs=0.05)


def test_snapshot_overall_excludes_uncertain_evidence():
    """overall 是方向性汇总，不能把 uncertain 混进去当方向证据。"""
    evidence, papers = _snapshot_fixture()
    snap = build_snapshot(evidence, papers)

    assert snap["total_directional"] == 3
    assert snap["by_stance"]["uncertain"]["count"] == 1
    assert snap["overall"]["count"] == snap["total_directional"]


def test_snapshot_sums_citations():
    evidence, papers = _snapshot_fixture()
    snap = build_snapshot(evidence, papers)

    assert snap["by_stance"]["support"]["citations_total"] == 80
    # overall 仅方向性：50 + 30 + 10 = 90（p4 的 2 次引用属不确定证据）
    assert snap["overall"]["citations_total"] == 90


def test_snapshot_counts_methods_and_designs():
    evidence, papers = _snapshot_fixture()
    snap = build_snapshot(evidence, papers)

    designs = snap["by_stance"]["support"]["designs"]
    assert designs == {"bench_test": 1, "experimental": 1}


def test_snapshot_strong_design_count_for_clinical():
    evidence = [item("p1", "support", "rct"), item("p2", "support", "meta_analysis")]
    papers = [{"id": "p1", "year": 2020, "citation_count": 1, "venue": "A"},
              {"id": "p2", "year": 2021, "citation_count": 1, "venue": "B"}]
    snap = build_snapshot(evidence, papers, domain="clinical")

    assert snap["by_stance"]["support"]["strong_design_count"] == 2


def test_snapshot_collects_venues_without_faking_impact_factor():
    """不得伪造 SJR；只列出去重期刊名，并用 notes 明确说明。"""
    evidence, papers = _snapshot_fixture()
    snap = build_snapshot(evidence, papers)

    assert snap["overall"]["venues"] == ["APL", "IEEE TED", "JAP"]
    assert any("SJR" in note for note in snap["notes"])


def test_snapshot_handles_missing_metadata():
    """缺少年份/引用量时不得崩溃，也不得编造数字。"""
    evidence = [item("missing", "support", "bench_test")]
    snap = build_snapshot(evidence, [])

    s = snap["by_stance"]["support"]
    assert s["avg_year"] is None
    assert s["citations_total"] == 0
    assert s["venues"] == []


def test_snapshot_excludes_ineligible_records_from_stance_summaries():
    evidence = [item("excluded", "support", eligible=False), item("included", "oppose")]
    snapshot = build_snapshot(evidence, [])

    assert snapshot["by_stance"]["support"]["count"] == 0
    assert snapshot["by_stance"]["oppose"]["count"] == 1


@pytest.mark.parametrize(("label", "expected"), [
    ("systematic review of randomized controlled trials", "systematic_review"),
    ("meta-analysis of RCTs", "meta_analysis"),
    ("in vitro experiment", "in_vitro"),
    ("randomized controlled trial", "rct"),
])
def test_descriptive_study_type_labels_match_specific_design(label, expected):
    from consensus import _normalize_study_type

    assert _normalize_study_type(label) == expected


def test_snapshot_is_deterministic():
    evidence, papers = _snapshot_fixture()
    assert build_snapshot(evidence, papers) == build_snapshot(evidence, papers)


# ============ 第 6 批：Evidence Review ============
def test_evidence_review_never_produces_percentages():
    """开放问题的核心约束：论文之间不是投票关系，禁止产出百分比。"""
    evidence = [item("p1"), item("p2", stance="oppose")]
    review = build_evidence_review(evidence)

    assert review["mode"] == "evidence_review"
    assert review["question_type"] == "open_ended"
    assert "score" not in review
    assert "directional_agreement" not in review


def test_evidence_review_builds_table_with_citations():
    evidence = [item("p1"), item("p2")]
    papers = [{"id": "p1", "year": 2020, "venue": "JAP"},
              {"id": "p2", "year": 2021, "venue": "APL"}]
    review = build_evidence_review(evidence, papers)

    table = review["evidence_table"]
    assert len(table) == 2
    assert table[0]["paper_id"] == "p1"
    assert table[0]["venue"] == "JAP"
    # 每条记录都必须带可追溯字段
    for row in table:
        assert row["claim"]
        assert row["excerpt"]
        assert row["scifact_label"]


def test_claim_evidence_graph_does_not_promote_title_to_a_claim():
    evidence = [item("p1", stance="support")]
    evidence[0] = EvidenceItem(
        **{**evidence[0].__dict__, "claim": "not reported"}
    )

    review = build_evidence_review(evidence)

    assert review["claims"] == []


def test_evidence_review_excludes_ineligible_from_table():
    evidence = [item("p1"), item("p2", eligible=False)]
    review = build_evidence_review(evidence)

    assert [r["paper_id"] for r in review["evidence_table"]] == ["p1"]
    assert review["excluded_count"] == 1


def test_evidence_review_groups_by_design_not_by_stance():
    """方法族聚类应按研究设计，而不是按支持/反对投票。"""
    evidence = [
        item("p1", "support", "bench_test"),
        item("p2", "oppose", "bench_test"),
        item("p3", "support", "simulation"),
    ]
    review = build_evidence_review(evidence)

    groups = {g["study_type"]: g["count"] for g in review["design_groups"]}
    assert groups == {"bench_test": 2, "simulation": 1}


def test_evidence_review_reports_design_fit():
    evidence = [item("p1", study_type="bench_test")]
    review = build_evidence_review(evidence, domain="cs_physics")

    assert review["design_groups"][0]["design_fit"] == 1.0


def test_evidence_review_detects_conflicting_directions():
    evidence = [
        item("p1", "support", outcome="gain", direction="increase"),
        item("p2", "oppose", outcome="gain", direction="decrease"),
    ]
    review = build_evidence_review(evidence)

    conflicts = [d for d in review["disagreements"] if d["is_conflicting"]]
    assert len(conflicts) == 1
    assert conflicts[0]["outcome"] == "gain"


def test_evidence_review_no_conflict_when_directions_agree():
    evidence = [
        item("p1", "support", outcome="gain", direction="increase"),
        item("p2", "support", outcome="gain", direction="increase"),
    ]
    review = build_evidence_review(evidence)

    assert not [d for d in review["disagreements"] if d["is_conflicting"]]


def test_evidence_review_flags_single_design_as_gap():
    evidence = [item("p1", study_type="bench_test"), item("p2", study_type="bench_test")]
    review = build_evidence_review(evidence)

    assert any("单一研究设计" in g for g in review["research_gaps"])


def test_evidence_review_summarizes_limitations():
    evidence = [
        item("p1", limitations=("abstract_only",)),
        item("p2", limitations=("abstract_only", "small_sample")),
    ]
    review = build_evidence_review(evidence)

    # 键经过归一化（连字符/下划线 → 空格，统一小写）
    assert review["limitations_summary"]["abstract only"] == 2
    assert any("abstract only" in g for g in review["research_gaps"])


def test_limitation_normalization_merges_case_and_separator_variants():
    """实测缺陷：模型同时返回 'abstract_only' 与 'Abstract only'，
    若不归一化会把同一含义拆成两条，稀释频次并让研究空白结论失真。"""
    evidence = [
        item("p1", limitations=("abstract_only",)),
        item("p2", limitations=("Abstract only",)),
        item("p3", limitations=("ABstract-Only",)),
    ]
    review = build_evidence_review(evidence)

    summary = review["limitations_summary"]
    assert summary == {"abstract only": 3}, f"归一化失败：{summary}"


def test_empty_limitation_is_ignored():
    evidence = [item("p1", limitations=("", "   "))]
    review = build_evidence_review(evidence)

    assert review["limitations_summary"] == {}


def test_evidence_review_states_no_percentage_in_notes():
    review = build_evidence_review([item("p1")])
    assert any("不计算共识百分比" in n for n in review["notes"])


# ============ SciFact 标签 ============
def test_scifact_labels_mapping():
    assert scifact_label("support") == "SUPPORT"
    assert scifact_label("oppose") == "CONTRADICT"
    assert scifact_label("uncertain") == "NOT_ENOUGH_INFO"
    assert scifact_label("not_applicable") == "NOT_ENOUGH_INFO"


def test_scifact_unknown_stance_is_not_enough_info():
    assert scifact_label("nonsense") == "NOT_ENOUGH_INFO"
    assert scifact_label("") == "NOT_ENOUGH_INFO"


def test_scifact_labels_cover_all_stance_values():
    from consensus import STANCE_VALUES
    assert set(STANCE_VALUES) <= set(SCIFACT_LABELS)
