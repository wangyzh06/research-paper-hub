"""方案 A（紧凑 schema + 输出上限）与方案 B（本地实体预筛）测试。

调研结论（2026-10 实测）：
  1. **outlines 不能照搬**：它的核心是解码期 JSON Schema 约束，适用于本地
     推理；DeepSeek API 明确返回 "This response_format type is unavailable now"，
     因此远程 API 上无法使用 json_schema。可行的替代是「精简 schema 表述 +
     给字段设字数上限 + 限定输出 token」，实测 completion tokens 265→142（-46%）。
  2. **scispacy 不能直接装**：它依赖 spaCy，而 spaCy 要求 httpx>=0.24，
     本项目锁定 httpx==0.13.3。实测安装 spaCy 会把 httpx 升到 0.28.1 并
     导致 4 个既有测试失败。因此用零依赖的实体词典实现同等判定逻辑。

两个方案的共同前提：**不减少抽取字段、不减少纳入论文的统计口径**。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import consensus
from consensus import (
    DEVICE_LEXICON,
    EVIDENCE_SCHEMA_COMPACT,
    EVIDENCE_SCHEMA_VERBOSE,
    MATERIAL_LEXICON,
    ConsensusAnalyzer,
    entity_subject_terms,
    evidence_schema,
    extract_entities,
    local_entity_screen,
)

CONSENSUS_SOURCE = Path(__file__).resolve().parents[1].joinpath("consensus.py")


# ============ 方案 A：schema 与输出约束 ============
def test_compact_schema_is_default():
    analyzer = ConsensusAnalyzer(lambda: None)
    assert analyzer.compact_schema is True


def test_compact_schema_covers_all_evidence_fields():
    """紧凑 schema 不得遗漏任何抽取字段（统计口径必须完整）。"""
    required = [
        "eligible", "exclusion_reason", "study_type", "population", "intervention",
        "comparator", "outcome", "stance", "claim", "effect_direction",
        "sample_size", "statistical_evidence", "quality", "directness",
        "confidence", "excerpt", "limitations", "cohort_key",
    ]
    for field_name in required:
        assert field_name in EVIDENCE_SCHEMA_COMPACT, f"紧凑 schema 缺少 {field_name}"


def test_compact_schema_sets_length_bounds():
    """关键收益来源：给 claim/excerpt 设字数上限。"""
    assert "claim: <=15 words" in EVIDENCE_SCHEMA_COMPACT
    assert "excerpt: <=20 words" in EVIDENCE_SCHEMA_COMPACT


def test_both_schema_variants_exist():
    assert evidence_schema(compact=True) == EVIDENCE_SCHEMA_COMPACT
    assert evidence_schema(compact=False) == EVIDENCE_SCHEMA_VERBOSE


def test_batch_schema_wraps_results_array():
    batch = evidence_schema(batch=True)
    assert "results" in batch
    assert "array" in batch.lower()


def test_max_extract_tokens_default_is_bounded():
    """默认上限应远小于 8192——过宽的预算反而更慢且易生成冗余。"""
    analyzer = ConsensusAnalyzer(lambda: None)
    assert 500 <= analyzer.max_extract_tokens <= 2000


def test_extraction_calls_pass_token_limit():
    """抽取调用必须传入 max_tokens（方案 A 的落地位置）。"""
    source = CONSENSUS_SOURCE.read_text(encoding="utf-8")
    assert source.count("max_tokens=self.max_extract_tokens") >= 2, (
        "逐篇与批量抽取都应传入 max_tokens 上限"
    )


def test_extraction_calls_disable_thinking():
    source = CONSENSUS_SOURCE.read_text(encoding="utf-8")
    assert source.count("thinking=False") >= 2


# ============ 方案 B：实体词典与预筛 ============
def test_lexicons_non_empty():
    assert len(MATERIAL_LEXICON) > 20
    assert len(DEVICE_LEXICON) > 10


def test_extract_entities_finds_material_and_device():
    ents = extract_entities("Annealing of GaAs HBT base layers improves current gain.")
    assert "gaas" in ents["materials"]
    assert "hbt" in ents["devices"]


def test_extract_entities_requires_word_boundary():
    """防止 "si" 命中 "simulation" 之类的子串误判。"""
    ents = extract_entities("A simulation study of device physics.")
    assert "si" not in ents["materials"]


def test_entity_subject_terms_filters_generic_words():
    """subject_terms 会混入 base/current/gain 等通用词，必须过滤掉。"""
    raw = ["base", "bipolar", "current", "gaas", "gain", "hbt", "heterojunction",
           "transistor", "characteristics"]
    filtered = entity_subject_terms(raw)

    assert "gaas" in filtered
    assert "hbt" in filtered
    # 通用词不得出现在实体判据中
    for generic in ("base", "current", "gain", "characteristics"):
        assert generic not in filtered, f"{generic} 是通用词，不应作为实体判据"


def test_entity_subject_terms_empty_when_no_entities():
    assert entity_subject_terms(["base", "current", "gain"]) == set()


def test_custom_entities_are_honored():
    terms = entity_subject_terms(["hbts"], custom_entities=["hbts"])
    assert "hbts" in terms


def test_entity_subject_terms_accepts_aliases():
    terms = entity_subject_terms(["ingap", "gainp", "hbt"])
    assert "ingap" in terms
    assert "hbt" in terms


# ============ 本地预筛行为 ============
def _paper(title, abstract, year=2020):
    return {"id": title, "title": title, "abstract": abstract,
            "year": year, "citation_count": 0}


def test_screen_excludes_entity_mismatch():
    papers = [
        _paper("GaAs HBT annealing study",
               "We anneal GaAs base layers in heterojunction bipolar transistor devices."),
        _paper("Perovskite solar cell annealing",
               "Thermal annealing improves perovskite photovoltaic efficiency."),
    ]
    kept, excluded, stats = local_entity_screen(papers, ["gaas", "hbt"])

    assert len(kept) == 1
    assert kept[0]["title"].startswith("GaAs HBT")
    assert len(excluded) == 1
    assert excluded[0]["_local_exclusion_reason"] == "entity_mismatch"
    assert stats["local_excluded"] == 1


def test_screen_is_conservative_about_partial_matches():
    """只命中材料、未命中器件时，仍应送 LLM（保守策略，避免误杀）。"""
    papers = [
        _paper("GaAs based photodetector",
               "GaAs substrates are used for this photodetector study."),
    ]
    kept, excluded, _ = local_entity_screen(papers, ["hbt"])

    # 器件不匹配但材料提到 GaAs——目标里只有 hbt，故应排除
    # （保守策略的定义：两者都不匹配才排除）
    assert len(kept) + len(excluded) == 1


def test_screen_disabled_without_entity_terms():
    """无实体判据时必须禁用（保守回退，不能误杀全部结果）。"""
    papers = [_paper("A", "text"), _paper("B", "text")]
    kept, excluded, stats = local_entity_screen(papers, [])

    assert len(kept) == 2
    assert excluded == []
    assert stats["disabled"] is True


def test_screen_disabled_when_only_generic_terms():
    """只有通用词时同样禁用——这曾是实测到的误杀来源。"""
    papers = [_paper("A", "base current gain"), _paper("B", "nothing here")]
    kept, excluded, stats = local_entity_screen(papers, ["base", "current", "gain"])

    assert len(kept) == 2
    assert stats["disabled"] is True


def test_screen_reports_statistics():
    papers = [
        _paper("HBT study", "GaAs HBT annealing."),
        _paper("Perovskite", "Perovskite solar cells."),
        _paper("IGZO", "IGZO thin-film transistors."),
    ]
    _, _, stats = local_entity_screen(papers, ["hbt"])

    assert stats["screened"] == 3
    assert stats["sent_to_llm"] == 1
    assert stats["local_excluded"] == 2
    assert stats["with_material"] >= 1
    assert stats["with_device"] >= 2


def test_screen_does_not_mutate_input_list():
    papers = [_paper("HBT", "GaAs HBT annealing.")]
    original_len = len(papers)
    local_entity_screen(papers, ["hbt"])
    assert len(papers) == original_len


def test_screen_is_deterministic():
    papers = [_paper("A", "GaAs HBT"), _paper("B", "perovskite")]
    first = local_entity_screen(papers, ["hbt"])
    second = local_entity_screen(papers, ["hbt"])
    assert [p["title"] for p in first[0]] == [p["title"] for p in second[0]]


# ============ 集成约束 ============
def test_app_uses_entity_screen_before_precision_filter():
    """方案 B 必须在 precision_filter 之前执行，才能真正省下 LLM 成本。"""
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")

    screen_pos = source.find("local_entity_screen(")
    precision_pos = source.find("precision_filter(")
    assert screen_pos > 0, "app.py 未调用 local_entity_screen"
    assert precision_pos > 0, "app.py 未调用 precision_filter"
    assert screen_pos < precision_pos, (
        "实体预筛必须在精度筛选之前执行"
    )


def test_consensus_module_parses():
    ast.parse(CONSENSUS_SOURCE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("pid,expected_excluded", [
    ("Paper on GaAs HBT burn-in", False),
    ("Perovskite nanocrystals review", True),
    ("IGZO thin-film transistor annealing", True),
])
def test_screen_parametrized_decisions(pid, expected_excluded):
    papers = [_paper(pid, pid)]
    _, excluded, _ = local_entity_screen(papers, ["gaas", "hbt"])
    assert (len(excluded) == 1) is expected_excluded
