"""检索策略测试：7 维度查询生成 + 短检索式约束。

背景（实测）：OpenAlex / Europe PMC / S2 对超过 ~12 个词的长自然语言句子
召回为 **0**。Consensus 官方示例中的 25 词长句直接照搬会检索不到任何文献。
因此必须把长问题拆成 7 个战略维度的**短技术检索式**。
"""
from __future__ import annotations

import pytest

from consensus import (
    SEARCH_DIMENSIONS,
    QuestionSpec,
    heuristic_dimension_queries,
)

MAX_WORDS = 8  # 实测：召回量随词数急剧下降（3词→数百篇，6词→数十篇，>12词→0）
IDEAL_WORDS = 6  # 目标上限


# === 维度定义完整性 ===
def test_exactly_seven_search_dimensions():
    """对标 Consensus Deep 的 7 组战略查询。"""
    assert len(SEARCH_DIMENSIONS) == 7


def test_dimension_keys_are_unique_and_complete():
    keys = [d["key"] for d in SEARCH_DIMENSIONS]
    assert len(set(keys)) == len(keys)
    assert set(keys) == {
        "mechanism", "variants", "synonyms", "limitations",
        "alternatives", "generalization", "citation",
    }


def test_every_dimension_declares_goal_and_hint():
    for dim in SEARCH_DIMENSIONS:
        assert dim.get("goal", "").strip(), f"{dim['key']} 缺少 goal"
        assert dim.get("hint", "").strip(), f"{dim['key']} 缺少 hint"


# === 关键约束：检索式必须短 ===
def test_heuristic_queries_respect_word_limit():
    spec = QuestionSpec.fallback("Does annealing improve HBT gain?", "yes_no")
    dims = heuristic_dimension_queries("Does annealing improve HBT gain?", spec)

    for key, queries in dims.items():
        assert queries, f"{key} 未产出任何检索式"
        for q in queries:
            assert len(q.split()) <= MAX_WORDS, (
                f"{key} 的检索式过长（{len(q.split())} 词），召回量会急剧下降: {q!r}"
            )


def test_heuristic_queries_prefer_short_phrases():
    """绝大多数检索式应在 6 词以内——这是高召回的实测前提。"""
    spec = QuestionSpec(
        question="q", question_type="yes_no",
        population="GaAs HBT", intervention="in situ annealing",
    )
    dims = heuristic_dimension_queries("q", spec)
    all_q = [q for qs in dims.values() for q in qs]

    short = [q for q in all_q if len(q.split()) <= IDEAL_WORDS]
    assert len(short) / len(all_q) >= 0.8, (
        f"仅 {len(short)}/{len(all_q)} 条检索式在 {IDEAL_WORDS} 词内，召回广度不足"
    )


def test_heuristic_queries_cover_all_dimensions():
    spec = QuestionSpec.fallback("topic", "open_ended")
    dims = heuristic_dimension_queries("topic", spec)

    assert set(dims.keys()) == {d["key"] for d in SEARCH_DIMENSIONS}


def test_long_natural_language_question_is_compressed():
    """长自然语言问题不得原样变成检索式。"""
    long_q = (
        "in situ annealing methods during GaAs/GaInP HBT base growth and "
        "their effects on gain, interface quality, and electrical parameters"
    )
    spec = QuestionSpec.fallback(long_q, "open_ended")
    dims = heuristic_dimension_queries(long_q, spec)

    for key, queries in dims.items():
        for q in queries:
            assert q.split() != long_q.split(), f"{key} 原样使用了长句"
            assert len(q.split()) <= MAX_WORDS


def test_heuristic_queries_strip_quotes_and_boolean_operators():
    dims = heuristic_dimension_queries('"quoted" AND topic OR other', None)
    joined = " ".join(q for qs in dims.values() for q in qs)

    assert '"' not in joined
    assert "'" not in joined
    # 布尔运算符会破坏各引擎的检索语义
    assert " AND " not in f" {joined} "
    assert " OR " not in f" {joined} "


def test_heuristic_is_deterministic():
    spec = QuestionSpec.fallback("annealing HBT", "yes_no")
    first = heuristic_dimension_queries("annealing HBT", spec)
    second = heuristic_dimension_queries("annealing HBT", spec)

    assert first == second


def test_heuristic_uses_pico_when_available():
    """有 PICO 时应优先用 intervention + population 构造检索式。"""
    spec = QuestionSpec(
        question="q", question_type="yes_no",
        population="GaAs HBT", intervention="in situ annealing",
        outcome="current gain",
    )
    dims = heuristic_dimension_queries("q", spec)
    mechanism = " ".join(dims["mechanism"])

    assert "annealing" in mechanism.lower()


def test_no_duplicate_queries_within_dimension():
    spec = QuestionSpec.fallback("annealing", "yes_no")
    dims = heuristic_dimension_queries("annealing", spec)

    for key, queries in dims.items():
        assert len(queries) == len(set(queries)), f"{key} 存在重复检索式"


def test_dimension_retrieval_strategy_covers_counter_evidence():
    """维度设计必须包含反驳性检索（trade-offs / null results），
    否则会系统性放大确认偏误。"""
    keys = {d["key"] for d in SEARCH_DIMENSIONS}
    assert "limitations" in keys

    limitations = next(d for d in SEARCH_DIMENSIONS if d["key"] == "limitations")
    text = (limitations["goal"] + " " + limitations["hint"]).lower()
    assert "trade-off" in text or "null" in text, (
        "limitations 维度必须明确要求检索 trade-offs / null results"
    )
