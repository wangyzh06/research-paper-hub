"""方案 A：精度优先检索测试。

背景（实测）：本领域窄问题的相关文献客观稀少，且通用方法词
（annealing / thermal / mechanism）能命中整个半导体领域。
实测一次召回的 200 篇中仅 2–8 篇提到 HBT，即「大而杂」——
Meter 与证据表因此失去意义。

方案 A 改为「小而精」，并用**研究对象词**（HBT / GaAs）而非通用方法词
判定切题。核心实现是 ``build_subject_terms`` 与 ``precision_filter``。
"""
from __future__ import annotations

import pytest

from consensus import (
    GENERIC_TERMS,
    QuestionSpec,
    build_concept_terms,
    build_subject_terms,
    precision_filter,
)


MIN_ABSTRACT = 80  # precision_filter 默认要求的最小摘要长度


def _paper(title, abstract="", year=2020, cites=0):
    """构造测试论文。

    precision_filter 默认要求摘要 >= 80 字符（无摘要无法做证据抽取），
    因此这里统一把摘要补齐到阈值以上，让测试聚焦于**相关性逻辑**，
    而不是意外地去测长度门槛。需要测试长度门槛时请用 _raw_paper。
    """
    text = abstract or "This paper studies the relevant topic in detail."
    if len(text) < MIN_ABSTRACT:
        text = (text + " " + "Additional methodological detail is reported. " * 3).strip()
    return {
        "title": title,
        "abstract": text,
        "year": year,
        "citation_count": cites,
    }


def _raw_paper(title, abstract):
    """不做长度补齐，用于测试摘要门槛本身。"""
    return {"title": title, "abstract": abstract, "year": 2020, "citation_count": 0}


def _hbt_spec():
    return QuestionSpec(
        question="q", question_type="yes_no",
        population="InGaP/GaAs HBT",
        intervention="thermal annealing",
        outcome="burn-in effect",
        neutral_query="InGaP GaAs HBT annealing burn-in",
        subject_terms=("GaAs", "HBT", "InGaP"),
    )


# === 研究对象词抽取 ===
def test_subject_terms_include_device_acronyms():
    subjects = build_subject_terms(_hbt_spec())

    assert "hbt" in subjects
    assert "gaas" in subjects
    assert "ingap" in subjects


def test_subject_terms_exclude_generic_method_words():
    """annealing / thermal 这类通用词绝不能作为切题判据。"""
    subjects = build_subject_terms(_hbt_spec())

    assert "annealing" not in subjects
    assert "thermal" not in subjects
    assert "mechanism" not in subjects


def test_chinese_question_does_not_pollute_subject_terms():
    """中文问题不得被词法切分成 "in效应" 之类的噪声词。

    这是真实发生过的 bug：未做 CJK 防护时，"保证退火效果的情况下，尽量减弱
    burn in效应" 会产出 'in效应'、'preservation' 等垃圾判据，导致过滤失效。
    """
    spec = _hbt_spec()
    subjects = build_subject_terms(spec, "保证退火效果的情况下，尽量减弱burn in效应")

    for term in subjects:
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in term), (
            f"中文片段混入研究对象词: {term!r}"
        )
    # 真正的英文研究对象仍在
    assert "hbt" in subjects


def test_generic_terms_contains_expected_words():
    for word in ("annealing", "thermal", "mechanism", "semiconductor", "review"):
        assert word in GENERIC_TERMS, f"{word} 应被列为通用词"


def test_concept_terms_exclude_generic_but_keep_specific():
    spec = _hbt_spec()
    concepts = build_concept_terms(spec, dimension_queries=["thermal annealing HBT reliability"])

    assert "annealing" not in concepts, "通用方法词不应进入概念集"
    assert "hbt" in concepts


# === 精度过滤 ===
def test_precision_filter_drops_offtopic_papers():
    papers = [
        _paper("InGaP/GaAs HBT burn-in study",
               "We study HBT burn-in with GaAs base layers and measure current gain."),
        _paper("Perovskite solar cell annealing",
               "Annealing improves perovskite film morphology and power conversion efficiency."),
        _paper("Quantum annealing for optimization",
               "Quantum annealing solves travelling salesman problems on a D-Wave device."),
    ]
    kept, stats = precision_filter(
        papers, ("annealing", "hbt", "gaas"), spec=_hbt_spec(),
        subject_terms=("hbt", "gaas"),
    )

    titles = [p["title"] for p in kept]
    assert titles == ["InGaP/GaAs HBT burn-in study"]
    assert stats["precision_kept"] == 1
    assert stats["subject_filtered"] == 2


def test_precision_filter_can_be_disabled_for_subjects():
    papers = [
        _paper("HBT study", "HBT burn-in with annealing treatment on GaAs wafers."),
        _paper("Perovskite annealing", "Annealing of perovskite films improves efficiency."),
    ]
    kept, _ = precision_filter(
        papers, ("annealing",), spec=_hbt_spec(),
        subject_terms=("hbt",), require_subject=False,
    )

    assert len(kept) == 2


def test_precision_filter_without_terms_keeps_everything():
    """无判据时不得误杀全部结果。"""
    papers = [_paper("A"), _paper("B")]
    kept, stats = precision_filter(papers, (), spec=None, subject_terms=())

    assert len(kept) == 2
    assert stats["precision_dropped"] == 0


def test_precision_filter_requires_abstract():
    papers = [
        _raw_paper("HBT study without abstract", ""),
        _paper("HBT study 2", "GaAs HBT annealing and burn-in behaviour is characterised here."),
    ]
    kept, _ = precision_filter(
        papers, ("annealing",), spec=_hbt_spec(), subject_terms=("hbt",),
    )

    assert [p["title"] for p in kept] == ["HBT study 2"]


def test_precision_filter_is_deterministic():
    papers = [
        _paper("GaAs HBT burn-in", "HBT devices on GaAs substrates show burn-in during annealing."),
        _paper("Perovskite work", "Unrelated perovskite photovoltaic research is presented."),
    ]
    first, _ = precision_filter(papers, ("hbt", "gaas"), spec=_hbt_spec(), subject_terms=("hbt", "gaas"))
    second, _ = precision_filter(papers, ("hbt", "gaas"), spec=_hbt_spec(), subject_terms=("hbt", "gaas"))

    assert [p["title"] for p in first] == [p["title"] for p in second]


def test_precision_filter_ranks_by_subject_hits():
    papers = [
        _paper("One hit", "HBT devices are studied here with detailed measurements."),
        _paper("Two hits", "GaAs HBT devices are studied here with detailed measurements."),
    ]
    kept, _ = precision_filter(
        papers, ("hbt",), spec=_hbt_spec(), subject_terms=("hbt", "gaas"),
    )

    assert kept[0]["title"] == "Two hits", "研究对象命中更多的论文应排在前"
    assert kept[0]["_subject_hits"] == 2


def test_precision_filter_records_diagnostics():
    papers = [
        _paper("HBT", "HBT annealing study with detailed measurements reported here."),
        _paper("Other", "Something else entirely with no matching subject identifiers."),
    ]
    kept, stats = precision_filter(
        papers, ("annealing",), spec=_hbt_spec(), subject_terms=("hbt",),
    )

    assert stats["precision_input"] == 2
    assert stats["precision_kept"] == 1
    assert stats["precision_dropped"] == 1
    # 诊断字段供 UI 展示漏斗
    assert kept[0]["_concept_hits"] >= 0
    assert kept[0]["_subject_hits"] >= 1


# === 与粗排的精度过滤协同 ===
def test_rerank_min_concept_hits_filters_offtopic():
    from consensus import rerank_consensus_papers

    papers = [
        _paper("HBT annealing burn-in",
               "We study GaAs HBT burn-in behaviour during annealing treatment."),
        _paper("Marine biology study",
               "A study about marine biology and coral reef ecosystems worldwide."),
    ]
    kept, stats = rerank_consensus_papers(
        papers, spec=_hbt_spec(), target_count=10,
        concept_terms=("hbt", "gaas", "annealing"), min_concept_hits=1,
    )

    titles = [p["title"] for p in kept]
    assert "Marine biology study" not in titles
    assert stats["dropped_offtopic"] >= 1
