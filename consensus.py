"""Auditable evidence records and deterministic consensus scoring."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Sequence


STANCE_VALUES = {
    "support",
    "oppose",
    "mixed",
    "uncertain",
    "not_applicable",
}

# SciFact 风格的三值立场标签（学术上更通用的证据关系词汇）。
# SciFact 是 FEVER/证据检索领域的标准数据集，其 SUPPORT / CONTRADICT /
# NOT_ENOUGH_INFO 标注体系被广泛引用，比自造词更有出处。
SCIFACT_LABELS = {
    "support": "SUPPORT",
    "oppose": "CONTRADICT",
    "mixed": "PARTIAL",
    "uncertain": "NOT_ENOUGH_INFO",
    "not_applicable": "NOT_ENOUGH_INFO",
}


def scifact_label(stance: str) -> str:
    """把内部立场映射为 SciFact 风格标签。"""
    return SCIFACT_LABELS.get(str(stance or "").strip().casefold(), "NOT_ENOUGH_INFO")
# 研究设计细分权重（P2）：高级别设计权重更高；综述类不参与独立计票
STUDY_WEIGHTS = {
    "rct": 1.0,
    "non_randomized_experiment": 0.9,
    "experimental": 0.9,
    "cohort": 0.8,
    "case_control": 0.7,
    "cross_sectional": 0.6,
    "observational": 0.7,
    "in_vitro": 0.5,
    "bench_test": 0.6,
    "simulation": 0.6,
    "theory": 0.5,
    "unknown": 0.4,
    "review": 0.0,
    "systematic_review": 0.0,
    "meta_analysis": 0.0,
    "commentary": 0.0,
}
QUALITY_WEIGHTS = {"high": 1.0, "medium": 0.75, "low": 0.4, "unknown": 0.5}

# === 设计适配度（P2）：按问题域给权重，而非全领域统一 ===
# 依据 Consensus 官方《证据分级指南》：
#   "The hierarchy should NOT be interpreted as a universal ranking of research
#    quality. The most appropriate study design depends on the question being asked."
# 因此临床问题奖励 RCT/队列，而材料/器件问题应由台架实验、体外实验与仿真主导；
# 在半导体器件研究里给 RCT 满分、给 bench_test 低分是完全错误的。
DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    # 临床/生物医学：经典证据金字塔
    "clinical": {
        "rct": 1.0,
        "non_randomized_experiment": 0.85,
        "experimental": 0.8,
        "cohort": 0.8,
        "case_control": 0.7,
        "cross_sectional": 0.6,
        "observational": 0.65,
        "in_vitro": 0.35,
        "bench_test": 0.3,
        "simulation": 0.3,
        "theory": 0.3,
        "unknown": 0.4,
    },
    # 材料/器件/物理：实验与测量优先，临床设计几乎不适用
    "materials_device": {
        "bench_test": 1.0,
        "experimental": 1.0,
        "non_randomized_experiment": 0.9,
        "in_vitro": 0.85,
        "simulation": 0.7,
        "theory": 0.6,
        "observational": 0.5,
        "cross_sectional": 0.4,
        "cohort": 0.3,
        "case_control": 0.25,
        "rct": 0.2,
        "unknown": 0.4,
    },
    # 通用/跨学科：温和的折中，不偏袒任一范式
    "general": {
        "rct": 0.9,
        "non_randomized_experiment": 0.85,
        "experimental": 0.9,
        "bench_test": 0.85,
        "cohort": 0.7,
        "case_control": 0.65,
        "cross_sectional": 0.6,
        "observational": 0.6,
        "in_vitro": 0.6,
        "simulation": 0.6,
        "theory": 0.5,
        "unknown": 0.4,
    },
}

# 综述类在任何域都不参与独立计票（避免与其纳入的原始研究重复计数）
REVIEW_STUDY_TYPES = {"review", "systematic_review", "meta_analysis", "commentary"}


def domain_weights(domain: str = "general") -> dict[str, float]:
    """按问题域返回设计类型权重表。"""
    key = (domain or "general").strip().casefold()
    if key in {"biomedicine", "medicine", "biology", "clinical"}:
        return DOMAIN_WEIGHTS["clinical"]
    if key in {"cs_physics", "computer_science", "physics", "materials",
               "materials_device", "engineering"}:
        return DOMAIN_WEIGHTS["materials_device"]
    return DOMAIN_WEIGHTS["general"]


def study_weight_for(study_type: str, domain: str = "general") -> float:
    """取某研究设计在给定问题域下的权重；综述类恒为 0。

    注意函数名刻意避开 ``study_weight``，以免与 calculate_consensus 内部的
    局部变量同名造成遮蔽。
    """
    st = (study_type or "unknown").strip().casefold()
    if st in REVIEW_STUDY_TYPES:
        return 0.0
    table = domain_weights(domain)
    if st in table:
        return table[st]
    return STUDY_WEIGHTS.get(st, table.get("unknown", 0.4))

# 展示 Consensus Meter 所需的最少方向性/独立研究数（P1）。
# 低于此值通常意味着**检索策略**未能召回足够证据，应扩充检索而非直接给结论。
MIN_DIRECTIONAL_STUDIES = 5
MIN_EVIDENCE_CONFIDENCE = 0.5
MIN_EVIDENCE_DIRECTNESS = 0.5
MAX_SUBQUESTION_PAPERS = 5

# 设计类型显示名（中文）
STUDY_LABELS = {
    "rct": "随机对照试验",
    "non_randomized_experiment": "非随机实验",
    "experimental": "实验研究",
    "cohort": "队列研究",
    "case_control": "病例对照",
    "cross_sectional": "横断面研究",
    "observational": "观察性研究",
    "in_vitro": "体外实验",
    "bench_test": "台架实验",
    "simulation": "仿真模拟",
    "theory": "理论研究",
    "unknown": "未知设计",
    "review": "综述",
    "systematic_review": "系统综述",
    "meta_analysis": "Meta 分析",
    "commentary": "评论",
}


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


def _normalize_study_type(val: str) -> str:
    s = str(val or "").strip().casefold()
    if s in STUDY_WEIGHTS:
        return s
    if "non_randomized" in s or "non-randomized" in s:
        return "non_randomized_experiment"
    # Specific synthesis designs must be recognized before the primary study
    # designs mentioned in their titles (e.g. a review of randomized trials).
    if "systematic_review" in s or "systematic review" in s:
        return "systematic_review"
    if "meta_analysis" in s or "meta-analysis" in s or "meta analysis" in s:
        return "meta_analysis"
    if "review" in s or "survey" in s or "overview" in s:
        return "review"
    if "comment" in s or "editorial" in s:
        return "commentary"
    if "in_vitro" in s or "in-vitro" in s or "in vitro" in s:
        return "in_vitro"
    if "bench" in s:
        return "bench_test"
    if "case_control" in s or "case-control" in s:
        return "case_control"
    if "cross_sectional" in s or "cross-sectional" in s:
        return "cross_sectional"
    if "cohort" in s:
        return "cohort"
    if "observation" in s:
        return "observational"
    if "simulation" in s or "model" in s or "numerical" in s:
        return "simulation"
    if "theory" in s or "theoretical" in s:
        return "theory"
    if "rct" in s or "randomized" in s or "randomised" in s:
        return "rct"
    if "experiment" in s or "trial" in s or "lab" in s or "empirical" in s:
        return "experimental"
    return "unknown"


def _normalize_stance(val: str) -> str:
    s = str(val or "").strip().casefold()
    if s in STANCE_VALUES:
        return s
    if "support" in s or "positive" in s or "yes" in s or "pro" in s or "agree" in s:
        return "support"
    if "oppose" in s or "negative" in s or "no" in s or "against" in s or "refute" in s:
        return "oppose"
    if "mix" in s or "neutral" in s or "inconsistent" in s or "conflict" in s:
        return "mixed"
    if "uncertain" in s or "unclear" in s or "inconclusive" in s:
        return "uncertain"
    return "not_applicable"


def _normalize_quality(val: str) -> str:
    s = str(val or "").strip().casefold()
    if s in QUALITY_WEIGHTS:
        return s
    if "high" in s:
        return "high"
    if "med" in s or "moderate" in s:
        return "medium"
    if "low" in s:
        return "low"
    return "unknown"


def _clamp(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _is_meter_qualified(
    item: EvidenceItem,
    domain: str = "general",
    min_confidence: float = MIN_EVIDENCE_CONFIDENCE,
    min_directness: float = MIN_EVIDENCE_DIRECTNESS,
) -> bool:
    return (
        item.eligible
        and item.stance in {"support", "oppose", "mixed"}
        and study_weight_for(item.study_type, domain) > 0
        and _clamp(item.confidence) >= min_confidence
        and _clamp(item.directness) >= min_directness
        and _clamp(item.independence_factor) > 0
    )


def qualified_stance_counts(
    evidence: Sequence[EvidenceItem],
    domain: str = "general",
    min_confidence: float = MIN_EVIDENCE_CONFIDENCE,
    min_directness: float = MIN_EVIDENCE_DIRECTNESS,
) -> dict[str, float]:
    """Return Meter stance totals weighted by independent-study contribution."""
    counts = {"support": 0.0, "oppose": 0.0, "mixed": 0.0}
    for item in evidence:
        if not _is_meter_qualified(item, domain, min_confidence, min_directness):
            continue
        counts[item.stance] += _clamp(item.independence_factor)
    counts["effective_independent_studies"] = sum(counts.values())
    return counts


def apply_min_study_gate(score: ConsensusScore,
                         items: Sequence[EvidenceItem],
                         minimum: int = MIN_DIRECTIONAL_STUDIES,
                         domain: str = "general",
                         min_confidence: float = MIN_EVIDENCE_CONFIDENCE,
                         min_directness: float = MIN_EVIDENCE_DIRECTNESS) -> ConsensusScore:
    """标记方向性证据是否充足（P1），但**不销毁**已算出的统计量。

    用户决策：“少于 5 篇一般是搜索策略出了问题。”
    因此这里只负责把不足标成 ``insufficient_evidence`` 供上层扩展检索，
    分数、立场分布与逐篇证据全部保留，避免出现“只有文献列表、没有任何分析”
    的白屏式失败。真正的数值门控交给 ``is_meter_displayable``。
    """
    qualified_n = qualified_stance_counts(
        items, domain, min_confidence, min_directness
    )["effective_independent_studies"]
    if score.score is not None and qualified_n < minimum:
        return replace(score, direction="insufficient_evidence")
    return score


def is_meter_displayable(score: ConsensusScore, minimum: int = 5) -> bool:
    """是否达到展示 Consensus Meter 数字的样本量门槛（P1）。"""
    return (
        score.score is not None
        and score.direction != "insufficient_evidence"
        and score.effective_independent_studies >= minimum
    )


def needs_search_expansion(score: ConsensusScore, minimum: int = 5) -> bool:
    """方向性证据不足时，应扩充检索而不是向用户报“无结果”。"""
    return (
        score.direction == "insufficient_evidence"
        or score.score is None
        or score.effective_independent_studies < minimum
    )


def _normalize_limitation(value: str) -> str:
    """归一化局限描述，避免同一含义被大小写/连字符拆成多个条目。

    实测问题：模型会同时返回 ``abstract_only`` 与 ``Abstract only``，
    若不归一化，频次统计会被稀释，研究空白结论失真。
    """
    import re as _re
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    # 统一分隔符，便于 "abstract only" 与 "abstract_only" 合并
    text = _re.sub(r"[\s_\-]+", " ", text)
    return text.strip()


def build_evidence_review(evidence: Sequence[EvidenceItem],
                          papers: Sequence[dict] | None = None,
                          domain: str = "general") -> dict:
    """开放问题的结构化证据综述（模式二：Evidence Review）。

    与 Meter 的根本区别：开放问题（"比较不同工艺的优劣"）的论文之间**不是投票
    关系**，强行算"共识百分比"是伪造数据。因此这里输出证据表 + 主题归纳 +
    分歧与空白，而**绝不产生任何百分比**。

    每条结论都挂到具体证据记录（claim-evidence 关联），保证可追溯。
    """
    meta: dict[str, dict] = {}
    for paper in (papers or ()):
        key = str(paper.get("id") or paper.get("doi") or paper.get("title") or "")
        if key:
            meta[key] = paper

    included = [item for item in evidence if item.eligible]
    excluded = [item for item in evidence if not item.eligible]

    # --- 1) 证据表：逐篇方法/对象/结果/局限 ---
    rows: list[dict] = []
    for item in included:
        raw = meta.get(item.paper_id) or {}
        rows.append({
            "paper_id": item.paper_id,
            "title": item.title,
            "year": raw.get("year"),
            "venue": raw.get("venue") or "",
            "study_type": item.study_type,
            "study_label": STUDY_LABELS.get(item.study_type, item.study_type),
            # SciFact 风格立场标签（便于与学术数据集对接与复核）
            "scifact_label": scifact_label(item.stance),
            "population": item.population,
            "intervention": item.intervention,
            "comparator": item.comparator,
            "outcome": item.outcome,
            "claim": item.claim,
            "effect_direction": item.effect_direction,
            "sample_size": item.sample_size,
            "statistical_evidence": item.statistical_evidence,
            "quality": item.quality,
            "limitations": list(item.limitations),
            "excerpt": item.excerpt,
        })

    # --- 2) 方法族归纳：按研究设计聚类（而非按"支持/反对"投票）---
    by_design: dict[str, list[EvidenceItem]] = {}
    for item in included:
        st = (item.study_type or "unknown").strip().casefold()
        by_design.setdefault(st, []).append(item)

    design_groups = [
        {
            "study_type": name,
            "study_label": STUDY_LABELS.get(name, name),
            "count": len(items),
            # 设计适配度：该设计在此问题域的权重（越高越适合回答本问题）
            "design_fit": round(study_weight_for(name, domain), 2),
            "paper_ids": [i.paper_id for i in items],
        }
        for name, items in sorted(by_design.items(), key=lambda kv: -len(kv[1]))
    ]

    # --- 3) 主要分歧：同一 outcome 上 effect_direction 不一致的论文 ---
    by_outcome: dict[str, dict[str, list[str]]] = {}
    for item in included:
        outcome = (item.outcome or "not reported").strip().lower()
        direction = (item.effect_direction or "not reported").strip().lower()
        if outcome == "not reported" or direction == "not reported":
            continue
        by_outcome.setdefault(outcome, {}).setdefault(direction, []).append(item.paper_id)
    disagreements = [
        {
            "outcome": outcome,
            "directions": {d: ids for d, ids in sorted(dirs.items())},
            "is_conflicting": len([d for d in dirs if d not in {"none", "no change"}]) > 1,
        }
        for outcome, dirs in by_outcome.items()
        if len(dirs) > 1
    ]

    # --- 4) 研究空白：局限高频项 + 单一设计主导 ---
    limitation_counts: dict[str, int] = {}
    for item in included:
        for lim in item.limitations:
            key = _normalize_limitation(lim)
            if key:
                limitation_counts[key] = limitation_counts.get(key, 0) + 1
    gaps: list[str] = []
    if limitation_counts:
        top = sorted(limitation_counts.items(), key=lambda kv: -kv[1])[:5]
        gaps.extend(f"普遍局限「{k}」（{v} 篇）" for k, v in top)
    if len(design_groups) == 1 and included:
        only = design_groups[0]["study_label"]
        gaps.append(f"全部证据均来自单一研究设计（{only}），缺少方法学交叉验证")
    if len(included) < MIN_DIRECTIONAL_STUDIES and included:
        gaps.append(f"仅 {len(included)} 篇合格证据，覆盖度不足")

    claims_by_text: dict[str, dict] = {}
    for item in included:
        claim_text = (item.claim or "").strip()
        if not claim_text or claim_text.casefold() == "not reported":
            continue
        key = claim_text.casefold()
        group = claims_by_text.setdefault(
            key,
            {
                "claim": claim_text,
                "supporting_papers": [],
                "opposing_papers": [],
                "limiting_papers": [],
            },
        )
        if item.stance == "support" and item.paper_id not in group["supporting_papers"]:
            group["supporting_papers"].append(item.paper_id)
        elif item.stance == "oppose" and item.paper_id not in group["opposing_papers"]:
            group["opposing_papers"].append(item.paper_id)
        elif item.paper_id not in group["limiting_papers"]:
            group["limiting_papers"].append(item.paper_id)
        if item.limitations and item.paper_id not in group["limiting_papers"]:
            group["limiting_papers"].append(item.paper_id)

    return {
        "mode": "evidence_review",
        "question_type": "open_ended",
        "evidence_table": rows,
        "claims": list(claims_by_text.values()),
        "design_groups": design_groups,

        "disagreements": disagreements,
        "research_gaps": gaps,
        "limitations_summary": dict(sorted(limitation_counts.items(), key=lambda kv: -kv[1])),
        "excluded_count": len(excluded),
        "included_count": len(included),
        "domain": domain,
        # 明确声明：开放问题不产出共识百分比
        "notes": ["开放问题不计算共识百分比：论文之间不是投票关系"],
    }


def build_snapshot(evidence: Sequence[EvidenceItem],
                   papers: Sequence[dict] | None = None,
                   domain: str = "general") -> dict:
    """Consensus Meter Snapshot：按立场分组的质量概览。

    对标 Consensus 官方 Snapshot 的四项指标（Recency / Methods / Journals /
    Citations）。本项目**不接入 SJR**（OpenAlex 无该字段），因此 Journals 一项
    以 `venues` 列出去重期刊名代替，绝不伪造影响因子。

    返回结构便于 UI 直接渲染，且完全由结构化证据确定性计算，模型无法篡改。
    """
    # paper_id → 元数据（年份/引用量），用于补齐 EvidenceItem 未含的字段
    meta: dict[str, dict] = {}
    for paper in (papers or ()):
        key = str(paper.get("id") or paper.get("doi") or paper.get("title") or "")
        if key:
            meta[key] = paper

    groups: dict[str, list[EvidenceItem]] = {
        "support": [], "oppose": [], "mixed": [], "uncertain": [],
    }
    for item in evidence:
        if not item.eligible:
            continue
        stance = item.stance if item.stance in groups else "uncertain"
        groups[stance].append(item)

    total_directional = sum(len(groups[k]) for k in ("support", "oppose", "mixed"))

    def summarize(items: Sequence[EvidenceItem]) -> dict:
        years: list[int] = []
        citations = 0
        designs: dict[str, int] = {}
        venues: set[str] = set()
        for item in items:
            raw = meta.get(item.paper_id) or {}
            year = raw.get("year")
            if isinstance(year, int) and 1900 < year < 2100:
                years.append(year)
            try:
                citations += int(raw.get("citation_count") or 0)
            except (TypeError, ValueError):
                pass
            st = (item.study_type or "unknown").strip().casefold()
            designs[st] = designs.get(st, 0) + 1
            venue = str(raw.get("venue") or "").strip()
            if venue:
                venues.add(venue)
        # 高质量设计计数（Methods 指标：RCT / 系统综述 / Meta 分析）
        strong = sum(
            count for name, count in designs.items()
            if name in {"rct", "systematic_review", "meta_analysis"}
        )
        return {
            "count": len(items),
            "avg_year": round(sum(years) / len(years), 1) if years else None,
            "years": sorted(years),
            "citations_total": citations,
            "citations_avg": round(citations / len(items), 1) if items else 0.0,
            "designs": dict(sorted(designs.items(), key=lambda kv: -kv[1])),
            "strong_design_count": strong,
            "venues": sorted(venues),
        }

    by_stance = {key: summarize(items) for key, items in groups.items()}

    # 跨全部方向性证据的汇总（供顶部概览）
    directional_items = [i for k in ("support", "oppose", "mixed") for i in groups[k]]
    overall = summarize(directional_items)

    return {
        "by_stance": by_stance,
        "overall": overall,
        "domain": domain,
        "total_directional": total_directional,
        # 明确声明未使用的指标，避免用户误以为有 SJR
        "notes": ["期刊指标未接入 SJR（数据源不提供），仅列出去重后期刊名"],
    }


def _raw_counts(items: Sequence[EvidenceItem]) -> dict[str, int]:
    counts = {stance: 0 for stance in sorted(STANCE_VALUES)}
    counts["excluded"] = 0
    for item in items:
        stance = item.stance if item.stance in STANCE_VALUES else "uncertain"
        counts[stance] += 1
        if not item.eligible:
            counts["excluded"] += 1
    return counts


def calculate_consensus(items: Sequence[EvidenceItem],
                        domain: str = "general",
                        min_confidence: float = MIN_EVIDENCE_CONFIDENCE,
                        min_directness: float = MIN_EVIDENCE_DIRECTNESS) -> ConsensusScore:
    """Calculate a reproducible score; uncertain evidence is never treated as 50%.

    ``domain`` 决定研究设计的权重表（设计适配度）：临床问题奖励 RCT，
    材料/器件问题奖励台架实验与实测，避免用统一权重硬套所有领域。
    """
    raw_counts = _raw_counts(items)
    support = 0.0
    oppose = 0.0
    mixed_weight = 0.0
    certainty_numerator = 0.0
    certainty_denominator = 0.0
    effective_studies = 0.0
    max_design_weight = max(domain_weights(domain).values(), default=1.0) or 1.0

    for item in items:
        if not _is_meter_qualified(item, domain, min_confidence, min_directness):
            continue
        study_weight = study_weight_for(item.study_type, domain)
        quality_weight = QUALITY_WEIGHTS.get(item.quality, QUALITY_WEIGHTS["unknown"])
        directness = _clamp(item.directness)
        confidence = _clamp(item.confidence)
        independence = _clamp(item.independence_factor)
        base_weight = study_weight * directness * confidence * independence
        weight = base_weight * quality_weight

        # Keep certainty sensitive to design fit, directness, confidence and
        # quality; do not divide by those same factors and cancel them out.
        normalized_design_fit = min(1.0, study_weight / max_design_weight)
        certainty_numerator += (
            independence * normalized_design_fit * directness * confidence * quality_weight
        )
        certainty_denominator += independence
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
    certainty = (
        certainty_numerator / certainty_denominator
        if certainty_denominator else 0.0
    )
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
class QuestionSpec:
    """PICO 结构化的研究问题（P1）。"""
    question: str
    question_type: str  # "yes_no" | "open_ended"
    population: str = "not reported"
    intervention: str = "not reported"
    comparator: str = "not reported"
    outcome: str = "not reported"
    direction: str = "unknown"  # improves/reduces/increases/associates/unknown
    domain: str = "general"     # "cs_physics" | "biomedicine" | "general"
    neutral_query: str = ""     # 高精度中立学术检索式
    search_queries: tuple[str, ...] = ()  # 多级学术检索变体列表 (Broad -> Specific)
    # 具体研究对象标识（材料体系/器件缩写），用于切题判定（方案 A）
    subject_terms: tuple[str, ...] = ()

    @classmethod
    def fallback(cls, question: str, question_type: str,
                 domain: str = "general", neutral_query: str = "",
                 search_queries: Sequence[str] | None = None,
                 subject_terms: Sequence[str] | None = None) -> "QuestionSpec":
        clean_q = " ".join(question.strip().split())
        base_q = neutral_query or clean_q
        queries = tuple(search_queries) if search_queries else (base_q,)
        return cls(
            question=question,
            question_type=question_type,
            domain=domain,
            neutral_query=base_q,
            search_queries=queries,
            subject_terms=tuple(subject_terms) if subject_terms else (),
        )


_YES_NO_MARKERS = (
    "是否", "能不能", "能否", "会不会", "有无", "有没有",
    "does ", "is ", "are ", "can ", "could ", "should ", "was ", "were ",
    "has ", "have ", "do ",
)


def heuristic_question_type(question: str) -> str:
    """启发式判断是否为二元（是/否）问题。"""
    normalized = question.strip().lower()
    for marker in _YES_NO_MARKERS:
        if normalized.startswith(marker):
            return "yes_no"
    return "open_ended"


def heuristic_domain(text: str) -> str:
    """启发式识别研究领域：cs_physics | biomedicine | general。"""
    t = text.lower()
    bio_keywords = (
        "clinical", "patient", "therapy", "disease", "drug", "treatment", "medicine",
        "cancer", "gene", "protein", "cell", "infection", "vaccine", "surgery",
        "health", "depression", "diet", "fasting", "metabolic", "diabetes",
        "临床", "患者", "疾病", "药物", "治疗", "癌症", "基因", "蛋白质", "细胞",
        "感染", "疫苗", "手术", "抑郁", "禁食", "饮食", "代谢", "糖尿病", "医学"
    )
    cs_phys_keywords = (
        "algorithm", "model", "neural", "bert", "gpt", "transformer", "network",
        "classification", "semiconductor", "gaas", "gainp", "ingap", "hbt", "fet",
        "laser", "vcsel", "mocvd", "anneal", "annealing", "physics", "quantum",
        "computer", "software", "code", "latency", "throughput", "optics",
        "算法", "神经网络", "深度学习", "分类", "半导体", "变容管", "激光器", "退火",
        "外延", "异质结", "晶体管", "量子", "物理", "计算机", "光学", "硅光"
    )
    if any(k in t for k in cs_phys_keywords):
        return "cs_physics"
    if any(k in t for k in bio_keywords):
        return "biomedicine"
    return "general"


_ZH_EN_ACADEMIC_MAP = [
    ("退火", "annealing"),
    ("增益", "gain"),
    ("异质结双极晶体管", "heterojunction bipolar transistor"),
    ("异质结", "heterojunction"),
    ("晶体管", "transistor"),
    ("外延", "epitaxial growth"),
    ("具体方法", "methods"),
    ("方法", "methods"),
    ("影响", "effects"),
    ("参数", "parameters"),
    ("机理", "mechanism"),
    ("机制", "mechanism"),
    ("缺陷", "defects"),
    ("界面", "interface"),
    ("钝化", "passivation"),
    ("掺杂", "doping"),
    ("寿命", "lifetime"),
    ("少子", "minority carrier"),
    ("载流子", "carrier"),
    ("激光器", "laser"),
    ("变容管", "varactor"),
    ("变容二极管", "varactor diode"),
    ("算法", "algorithm"),
    ("分类", "classification"),
    ("深度学习", "deep learning"),
    ("神经网络", "neural network"),
    ("临床", "clinical"),
    ("患者", "patient"),
    ("治疗", "treatment"),
    ("药物", "drug"),
    ("疾病", "disease"),
    ("癌症", "cancer"),
    ("效果", "effects"),
]


def build_neutral_query_fallback(question: str) -> str:
    """剔除倾向性词构建中立学术检索短语，并在中文或中英混合时启发式英文化。"""
    import re
    q = " ".join(question.strip().split())
    # 移除疑问助词与典型二元前缀
    for marker in _YES_NO_MARKERS:
        if q.lower().startswith(marker):
            q = q[len(marker):].strip()
            break
    # 移除常见肯定/否定/比较倾向动词
    cleaned = re.sub(
        r"\b(outperform|outperforms|improve|improves|better|worse|effective|cause|causes|increase|increases|reduce|reduces|versus|vs)\b",
        " ", q, flags=re.IGNORECASE
    )
    cleaned = re.sub(r"[？?。！!,，、：:（）\(\)]", " ", cleaned)

    # 若包含中文字符，利用学术词汇映射字典替换中文术语
    if re.search(r'[\u4e00-\u9fff]', cleaned):
        translated = cleaned
        for zh, en in _ZH_EN_ACADEMIC_MAP:
            translated = translated.replace(zh, f" {en} ")
        # 移除其余残留的中文虚词
        translated = re.sub(r'[\u4e00-\u9fff]', ' ', translated)
        tokens = [t.strip("\"' ") for t in translated.split() if len(t.strip("\"' ")) > 1]
        if tokens:
            return " ".join(tokens)

    tokens = [term.strip("\"' ") for term in cleaned.split() if len(term.strip("\"' ")) > 1]
    return " ".join(tokens) if tokens else question


# 检索维度定义：直接对标 Consensus Deep 模式的 7 组战略查询
# （源自 example/三个例子的搜索过程.txt 中 30 steps = 7 组 × 3 查询的实际行为）
SEARCH_DIMENSIONS: tuple[dict[str, str], ...] = (
    {
        "key": "mechanism",
        "goal": "Identify core theories and mechanisms linking the key concepts.",
        "hint": "focus on underlying physical/chemical/biological mechanisms and theory",
    },
    {
        "key": "variants",
        "goal": "Explore specific process/method variations (conditions, windows, parameter ranges).",
        "hint": "enumerate concrete process variants, temperatures, durations, in-situ vs ex-situ",
    },
    {
        "key": "synonyms",
        "goal": "Rephrase using synonyms and technical variants.",
        "hint": "use alternative terminology, abbreviations and equivalent technical terms",
    },
    {
        "key": "limitations",
        "goal": "Target literature reporting trade-offs, null results, or limitations.",
        "hint": "seek negative results, trade-offs, failure modes, adverse effects and limitations",
    },
    {
        "key": "alternatives",
        "goal": "Expand to related alternative methods addressing the same goal.",
        "hint": "adjacent techniques that achieve a similar outcome",
    },
    {
        "key": "generalization",
        "goal": "Contrast findings with other device types or domains for generalizability.",
        "hint": "compare with other material systems, device families or application domains",
    },
    {
        "key": "citation",
        "goal": "Discover additional papers by following citation connections.",
        "hint": "foundational and highly-cited works, seminal references",
    },
)


def _sanitize_query_phrase(text: str, max_words: int = 6) -> str:
    """清洗检索式：去除引号/布尔运算符/标点，并压缩到 max_words。

    布尔运算符（AND/OR/NOT）在 OpenAlex / Europe PMC / S2 的 search 参数中
    会被当作普通词参与匹配，反而污染检索结果，因此统一剔除。

    默认 6 词：实测召回量随词数急剧下降（3 词可召回数百至数万篇，
    6 词仅数十篇，>12 词为 0），短检索式才能获得足够召回广度。
    """
    import re as _re
    value = str(text or "")
    value = value.replace('"', " ").replace("'", " ")
    # 移除布尔运算符（独立词形式）
    value = _re.sub(r"(?i)\b(and|or|not)\b", " ", value)
    # 去掉会干扰检索的标点，保留字母数字、空格、连字符与斜杠（如 GaInP/GaAs、burn-in）
    value = _re.sub(r"[^\w\s/\-\.]", " ", value, flags=_re.UNICODE)
    value = " ".join(value.split())
    words = value.split()
    if len(words) > max_words:
        value = " ".join(words[:max_words])
    return value.strip()


def heuristic_dimension_queries(question: str, spec: "QuestionSpec | None" = None,
                                max_per_dimension: int = 2) -> dict[str, tuple[str, ...]]:
    """启发式维度检索式（LLM 不可用时的兜底）。

    关键约束（实测得出）：OpenAlex / Europe PMC / S2 对**超过 ~10 个词的长句**
    几乎全部返回 0 结果。因此这里只产出 4-8 个技术词的短检索式。
    """
    population = ""
    intervention = ""
    outcome = ""
    if spec is not None:
        population = (spec.population or "").strip()
        intervention = (spec.intervention or "").strip()
        outcome = (spec.outcome or "").strip()
    population = population if population and population != "not reported" else ""
    intervention = intervention if intervention and intervention != "not reported" else ""
    outcome = outcome if outcome and outcome != "not reported" else ""

    base_terms = " ".join(t for t in (intervention, population) if t).strip()
    if not base_terms:
        base_terms = build_neutral_query_fallback(question)
    base_terms = _sanitize_query_phrase(base_terms, max_words=5)
    if not base_terms:
        base_terms = _sanitize_query_phrase(question, max_words=5)

    population = _sanitize_query_phrase(population, max_words=3)
    intervention = _sanitize_query_phrase(intervention, max_words=3)
    outcome = _sanitize_query_phrase(outcome, max_words=3)

    out: dict[str, tuple[str, ...]] = {}
    specs = {
        "mechanism": [f"{base_terms} mechanism", base_terms],
        "variants": [f"{base_terms} conditions", f"{base_terms} parameters"],
        "synonyms": [f"{population} {outcome}".strip() if population else base_terms],
        "limitations": [f"{base_terms} limitations", f"{base_terms} degradation"],
        "alternatives": [f"{intervention or base_terms} alternative"],
        "generalization": [f"{intervention or base_terms} comparison"],
        "citation": [base_terms],
    }
    for key, queries in specs.items():
        cleaned: list[str] = []
        for q in queries:
            q = _sanitize_query_phrase(q, max_words=6)
            if len(q.split()) >= 2 and q not in cleaned:
                cleaned.append(q)
        out[key] = tuple(cleaned[:max_per_dimension])
    return out


async def build_dimension_queries(question: str, llm_factory,
                                  spec: "QuestionSpec | None" = None,
                                  max_per_dimension: int = 2) -> dict[str, tuple[str, ...]]:
    """按 Consensus Deep 的 7 个战略维度生成检索式。

    每个维度产出 1-3 条**短技术检索式**（4-10 词）。长自然语言句子在
    OpenAlex / Europe PMC / S2 上实测召回为 0，必须避免。
    """
    import re as _re
    fallback = heuristic_dimension_queries(question, spec, max_per_dimension)
    client = llm_factory()
    try:
        dim_lines = "\n".join(
            f'- {d["key"]}: {d["goal"]} ({d["hint"]})' for d in SEARCH_DIMENSIONS
        )
        pico = ""
        if spec is not None:
            pico = (
                f"\nKnown PICO:\n"
                f"- population: {spec.population}\n"
                f"- intervention: {spec.intervention}\n"
                f"- comparator: {spec.comparator}\n"
                f"- outcome: {spec.outcome}\n"
            )
        schema = (
            "Return one JSON object whose keys are exactly these dimension names: "
            + ", ".join(d["key"] for d in SEARCH_DIMENSIONS)
            + ". Each value is an array of 1-2 English academic search phrases. "
            "CRITICAL LENGTH RULE: each phrase MUST be only 3-6 words. "
            "Short phrases recall far more literature than long ones: a 3-word phrase can "
            "return hundreds of papers while a 6-word phrase returns only a few dozen, and "
            "sentences longer than 10 words return ZERO results. "
            "Keep the essential technical nouns and drop filler words like "
            "'methods', 'effects', 'during', 'their'. "
            "Do NOT use quotes, Boolean operators, or punctuation."
        )
        prompt = (
            "You are the query planner of an academic search engine (modeled on Consensus.app Deep review).\n"
            "Generate targeted literature search phrases across these strategic dimensions:\n"
            f"{dim_lines}\n"
            f"{pico}\n"
            f"Research topic: {question}\n\n"
            "Translate to precise academic English if the topic is not English. Output strict JSON only."
        )
        raw = await client.complete_strict(
            [
                {"role": "system", "content": "You are an academic search query strategist. " + schema},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return fallback

        result: dict[str, tuple[str, ...]] = {}
        for dim in SEARCH_DIMENSIONS:
            key = dim["key"]
            raw_value = payload.get(key)
            candidates: list[str] = []
            if isinstance(raw_value, list):
                candidates = [str(v) for v in raw_value]
            elif isinstance(raw_value, str):
                candidates = [raw_value]

            cleaned: list[str] = []
            for item in candidates:
                q = " ".join(str(item).replace('"', "").replace("'", "").split())
                # 硬性长度约束：实测召回量随词数急剧下降
                #  3 词 → 数百至数万篇；6 词 → 数十篇；>12 词 → 0 篇
                # 因此超过 8 词的检索式强制截断到 6 个核心词。
                words = q.split()
                if len(words) > 8:
                    q = " ".join(words[:6])
                if len(q.split()) >= 2 and q not in cleaned:
                    cleaned.append(q)
            # 逐维度回退：缺失或清洗后为空时用启发式补齐
            if not cleaned:
                cleaned = list(fallback.get(key, ()))
            result[key] = tuple(cleaned[:max_per_dimension])
        return result
    except Exception:
        return fallback
    finally:
        close = getattr(client, "aclose", None)
        if close:
            try:
                await close()
            except Exception:
                pass


async def build_question_spec(question: str, llm_factory) -> QuestionSpec:
    """用 LLM 深度解构 PICO、学科领域、学术检索长句与短语变体（对标 Consensus 搜索策略）；失败退回启发式。"""
    import re
    heuristic_type = heuristic_question_type(question)
    h_domain = heuristic_domain(question)
    h_query = build_neutral_query_fallback(question)
    h_queries = (h_query,)

    client = llm_factory()
    try:
        schema = (
            "Return one JSON object with keys: "
            "question_type (yes_no|open_ended), "
            "domain (cs_physics|biomedicine|general), "
            "population (Target material/device/population in English, e.g. 'GaAs HBT'), "
            "intervention (Technique/method in English, e.g. 'in situ thermal annealing'), "
            "comparator ('not reported' or baseline), "
            "outcome (Target metrics/effects in English, e.g. 'current gain'), "
            "direction (improves|reduces|increases|associates|unknown), "
            "subject_terms (Array of 2-5 CONCRETE subject identifiers in English: "
            "material systems, device acronyms, chemistry, population. "
            "e.g. ['GaAs', 'HBT', 'InGaP', 'heterojunction bipolar transistor']. "
            "CRITICAL: these must be specific named entities, NOT generic categories. "
            "NEVER output vague words like 'semiconductor', 'materials', 'devices', "
            "'parameters', 'gain', 'properties' — those match the entire literature. "
            "If the topic does not name a specific material/device, infer the most "
            "likely concrete one from context and say so in neutral_queries.), "
            "neutral_queries (Array of 2-3 high-precision English academic search phrases/sentences, "
            "strictly neutral, NO quotes around words, e.g. ['in situ annealing GaAs HBT', "
            "'in situ annealing methods GaAs heterojunction bipolar transistors current gain', "
            "'effects of in situ thermal annealing during GaAs HBT growth on gain']), "
            "primary_query (Concise 3-5 word clean English academic search query for API, e.g. 'GaAs HBT in situ annealing gain')."
        )
        prompt = (
            f"You are an expert academic search query strategist modeled after Consensus.app.\n"
            f"Deconstruct the user's research topic for scholarly literature retrieval.\n"
            f"Even if the topic is in Chinese or mixed Chinese-English, translate all search queries into precise, idiomatic academic English.\n"
            f"Do NOT wrap query terms in quotes. Avoid Boolean operators (AND/OR). Output strict JSON only.\n"
            f"IMPORTANT: subject_terms drive topical filtering downstream. Generic category\n"
            f"words make every paper look relevant, so always prefer the most specific\n"
            f"concrete identifier the context implies (device acronym, material formula,\n"
            f"chemical system) over an abstract category.\n\n"
            f"Research Topic: {question}"
        )
        raw = await client.complete_strict(
            [
                {"role": "system", "content": "You are a scholarly search engine planner. " + schema},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
        q_type = str(payload.get("question_type", heuristic_type)).strip().casefold()
        if q_type not in {"yes_no", "open_ended"}:
            q_type = heuristic_type

        domain = str(payload.get("domain", h_domain)).strip().casefold()
        if domain not in {"cs_physics", "biomedicine", "general"}:
            domain = h_domain

        # 提取主检索词与变体列表
        primary_q = str(payload.get("primary_query") or "").strip()
        # 清理多余的双引号
        primary_q = primary_q.replace('"', '').replace("'", '').strip()

        nq_raw = payload.get("neutral_queries")
        queries_list: list[str] = []
        if isinstance(nq_raw, list):
            for q_item in nq_raw:
                cleaned_item = str(q_item).replace('"', '').replace("'", '').strip()
                if cleaned_item and len(cleaned_item) > 2 and cleaned_item not in queries_list:
                    queries_list.append(cleaned_item)
        elif isinstance(nq_raw, str) and nq_raw.strip():
            cleaned_item = nq_raw.replace('"', '').replace("'", '').strip()
            if cleaned_item:
                queries_list.append(cleaned_item)

        # 确保 primary_q 包含在 queries_list 的第一位
        if primary_q and primary_q not in queries_list:
            queries_list.insert(0, primary_q)
        elif primary_q and queries_list:
            # 将 primary_q 置顶
            queries_list.remove(primary_q)
            queries_list.insert(0, primary_q)

        # 兜底：若 LLM 没有产出有效英文检索词，使用启发式
        if not queries_list:
            queries_list = [h_query]
        final_neutral_query = queries_list[0] if queries_list else h_query

        # 具体研究对象标识（方案 A 切题判定的依据）
        subject_list: list[str] = []
        st_raw = payload.get("subject_terms")
        if isinstance(st_raw, list):
            for item in st_raw:
                cleaned = str(item).replace('"', '').replace("'", '').strip()
                if cleaned and cleaned.lower() not in {s.lower() for s in subject_list}:
                    subject_list.append(cleaned)
        elif isinstance(st_raw, str) and st_raw.strip():
            subject_list.append(st_raw.replace('"', '').replace("'", '').strip())

        return QuestionSpec(
            question=question,
            question_type=q_type,
            population=str(payload.get("population") or "not reported"),
            intervention=str(payload.get("intervention") or "not reported"),
            comparator=str(payload.get("comparator") or "not reported"),
            outcome=str(payload.get("outcome") or "not reported"),
            direction=str(payload.get("direction") or "unknown"),
            domain=domain,
            neutral_query=final_neutral_query,
            search_queries=tuple(queries_list),
            subject_terms=tuple(subject_list),
        )
    except Exception:
        return QuestionSpec.fallback(
            question, heuristic_type, domain=h_domain,
            neutral_query=h_query, search_queries=h_queries
        )
    finally:
        close = getattr(client, "aclose", None)
        if close:
            try:
                await close()
            except Exception:
                pass


def rerank_consensus_papers(
    papers: Sequence[dict],
    spec: QuestionSpec | None = None,
    target_count: int = 20,
    min_abstract_len: int = 80,
    concept_terms: Sequence[str] | None = None,
    min_concept_hits: int = 1,
) -> tuple[list[dict], dict[str, int]]:
    """在本地内存中对候选文献进行两阶段质量与 PICO 交叉精排。

    ``min_concept_hits``：低于该概念命中数的论文直接剔除（精度优先）。
    默认 1 表示「至少要沾一个主题概念」，可有效滤掉完全不相关的文献；
    设为 0 则关闭该过滤（退回旧行为）。

    返回: (精排后的候选论文列表, 漏斗统计字典)
    """
    total_raw = len(papers)
    # 阶段 1：摘要完整性过滤
    valid_papers = [
        dict(p) for p in papers
        if (p.get("abstract") or "").strip() and len((p.get("abstract") or "").strip()) >= min_abstract_len
    ]
    with_abstract_count = len(valid_papers)
    if not valid_papers:
        # 降级：若全部无长摘要，返回非空摘要或原论文
        fallback = [dict(p) for p in papers if (p.get("abstract") or "").strip()] or [dict(p) for p in papers]
        return fallback[:target_count], {
            "retrieved": total_raw,
            "with_abstract": len(fallback),
            "final_selected": len(fallback[:target_count]),
        }

    # 阶段 2：PICO 交叉与实证信号打分
    empirical_keywords = (
        "experiment", "experimental", "trial", "rct", "cohort", "measured", "measurement",
        "benchmark", "dataset", "evaluation", "sample", "participants", "statistically",
        "p <", "p=", "results show", "we propose", "in vitro", "in vivo", "simulation"
    )
    negative_keywords = ("erratum", "corrigendum", "author correction", "retraction", "editorial")

    # 构建匹配项
    pico_terms: set[str] = set()
    if spec:
        for val in (spec.intervention, spec.comparator, spec.population, spec.outcome):
            if val and val != "not reported":
                for token in val.lower().split():
                    t = token.strip("\"',.() ")
                    if len(t) > 2:
                        pico_terms.add(t)

    # 概念词覆盖率：区分“字面命中”与“主题相关”的关键信号。
    # 例如 "quantum annealing" 命中 annealing，但缺少 HBT/device 等概念词。
    concept_set: set[str] = set()
    for term in (concept_terms or ()):
        t = str(term).strip().lower()
        if len(t) > 2:
            concept_set.add(t)
    if not concept_set:
        concept_set = set(pico_terms)

    scored: list[tuple[float, dict]] = []
    dropped_offtopic = 0
    for paper in valid_papers:
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()
        combined = f"{title} {abstract}"

        # 初始分：优先使用融合分；缺失时用引用数归一化（高引论文获得基础加权）
        fusion_score = paper.get("_score") or paper.get("_fusion_score")
        if fusion_score is not None:
            score = float(fusion_score)
        else:
            score = min(2.0, float(paper.get("citation_count") or 0) / 100.0)

        # 实证加权
        empirical_hits = sum(1 for kw in empirical_keywords if kw in combined)
        score += min(2.0, empirical_hits * 0.3)

        # 垃圾词惩罚
        if any(neg in title for neg in negative_keywords):
            score -= 5.0

        # PICO 交叉加权
        if pico_terms:
            overlap = sum(1 for term in pico_terms if term in combined)
            score += (overlap / len(pico_terms)) * 2.5
            # 若干预与对照同时出现，额外强加权
            if spec and spec.intervention != "not reported" and spec.comparator != "not reported":
                int_hit = any(w in combined for w in spec.intervention.lower().split() if len(w) > 2)
                comp_hit = any(w in combined for w in spec.comparator.lower().split() if len(w) > 2)
                if int_hit and comp_hit:
                    score += 2.0

        # 概念覆盖率：主题词命中越多越相关（线性归一到 0~3 分）
        concept_hits = 0
        if concept_set:
            concept_hits = sum(1 for term in concept_set if term in combined)
            coverage = concept_hits / len(concept_set)
            score += coverage * 3.0
            # 完全不沾任何概念词 → 强惩罚（典型跑题论文）
            if concept_hits == 0:
                score -= 3.0

        # 精度优先：主题概念命中数不足的直接剔除（方案 A）
        if concept_set and concept_hits < min_concept_hits:
            dropped_offtopic += 1
            continue

        scored.append((score, paper))

    # 按最终分数从高到低排序
    scored.sort(key=lambda pair: pair[0], reverse=True)
    selected = [paper for _, paper in scored[:target_count]]

    funnel_stats = {
        "retrieved": total_raw,
        "with_abstract": with_abstract_count,
        "dropped_offtopic": dropped_offtopic,
        "final_selected": len(selected),
    }
    return selected, funnel_stats


def precision_filter(
    papers: Sequence[dict],
    concept_terms: Sequence[str],
    spec: "QuestionSpec | None" = None,
    min_coverage: float = 0.25,
    min_hits: int = 2,
    min_abstract_len: int = 80,
    subject_terms: Sequence[str] | None = None,
    require_subject: bool = True,
) -> tuple[list[dict], dict[str, int]]:
    """精度优先筛选：只保留真正切题的论文（方案 A）。

    动机（实测）：本领域窄问题的相关文献客观稀少，且通用方法词
    （annealing / thermal / mechanism）能命中整个半导体领域。
    强行凑满 200 篇只会掺入大量跑题论文（实测 200 篇中仅 2 篇提到 HBT），
    使 Meter 与证据表失去意义。

    因此改为「小而精」，并**用研究对象词（HBT / GaAs）而非通用方法词判定切题**：
    宁可只纳入 10 篇全部相关的论文，也不纳 200 篇里 2 篇相关的。

    判定规则：
      1. 若提供 subject_terms，则**必须**命中至少一个研究对象词（require_subject）
      2. 且概念覆盖率 >= min_coverage 或 概念命中数 >= min_hits
    """
    total = len(papers)
    concepts = []
    seen = set()
    for term in (concept_terms or ()):
        t = str(term).strip().lower()
        if len(t) > 2 and t not in seen:
            seen.add(t)
            concepts.append(t)

    subjects = []
    seen_subj = set()
    for term in (subject_terms or ()):
        t = str(term).strip().lower()
        if len(t) > 1 and t not in seen_subj:
            seen_subj.add(t)
            subjects.append(t)

    # 既无概念词也无研究对象词时无法判定，直接返回（避免误杀全部结果）
    if not concepts and not subjects:
        kept = [dict(p) for p in papers]
        return kept, {
            "precision_input": total,
            "precision_kept": len(kept),
            "precision_dropped": 0,
            "relevance_threshold": 0,
            "subject_filtered": 0,
        }

    kept: list[dict] = []
    subject_filtered = 0
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if len(abstract) < min_abstract_len:
            continue
        text = f"{paper.get('title') or ''} {abstract}".lower()

        # 1) 研究对象必须是切题的
        subject_hits = sum(1 for term in subjects if term in text) if subjects else 0
        if require_subject and subjects and subject_hits == 0:
            subject_filtered += 1
            continue

        # 2) 概念覆盖
        hits = sum(1 for term in concepts if term in text) if concepts else 0
        coverage = hits / len(concepts) if concepts else 0.0
        if concepts and not (coverage >= min_coverage or hits >= min_hits):
            continue

        paper = dict(paper)
        paper["_concept_hits"] = hits
        paper["_concept_coverage"] = round(coverage, 4)
        paper["_subject_hits"] = subject_hits
        kept.append(paper)

    # 按研究对象命中数优先，其次概念覆盖、最后引用量
    kept.sort(
        key=lambda p: (
            p.get("_subject_hits", 0),
            p.get("_concept_hits", 0),
            p.get("citation_count") or 0,
        ),
        reverse=True,
    )
    return kept, {
        "precision_input": total,
        "precision_kept": len(kept),
        "precision_dropped": total - len(kept),
        "relevance_threshold": round(min_coverage, 3),
        "subject_filtered": subject_filtered,
    }


# 领域通用词：单靠它们无法判定切题（"annealing" 能命中整个半导体界）
GENERIC_TERMS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "not", "reported", "effect", "effects", "method", "methods", "study",
    "studies", "using", "used", "use", "via", "during", "their", "its",
    "into", "than", "then", "also", "such", "has", "have", "had", "can",
    "may", "might", "will", "would", "should", "could", "between", "among",
    "other", "others", "more", "most", "less", "least", "very", "much",
    "many", "some", "any", "all", "each", "both", "after", "before",
    "under", "over", "about", "based", "however", "therefore", "thus",
    # 通用方法/属性词：跨领域高频，不能作为切题判据
    "annealing", "anneal", "thermal", "temperature", "heat", "treatment",
    "process", "processing", "method", "technique", "mechanism", "mechanisms",
    "effect", "effects", "performance", "property", "properties", "behavior",
    "reliability", "degradation", "improvement", "improving", "enhancement",
    "optimization", "optimizing", "conditions", "parameters", "review",
    "comparison", "analysis", "investigation", "semiconductor",
    "semiconductors", "device", "devices", "film", "films", "thin",
    "material", "materials", "high", "low", "novel", "new", "recent",
    "advanced", "advances", "application", "applications", "based",
    "induced", "enhanced", "reduced", "reducing", "maintaining", "tradeoff",
    "trade", "limitations", "while", "using", "toward", "towards",
}


def _is_acronym(token: str) -> bool:
    """判断是否为领域缩写（如 HBT、RCT、MOSFET、LED）。"""
    return len(token) >= 2 and token.isalpha() and token.isupper()


def build_concept_terms(spec: "QuestionSpec | None",
                        dimension_queries: Sequence[str] | None = None,
                        extra: Sequence[str] | None = None) -> set[str]:
    """汇总问题相关的核心概念词，用于词面相关性判断。

    仅用 PICO 不够：维度检索式里含大量同义技术词（hydrogen passivation、
    defect annealing 等），把它们纳入可显著提升主题判别力。
    """
    terms: set[str] = set()

    def add_text(value: str) -> None:
        raw = str(value or "")
        for token in raw.replace("/", " ").replace("-", " ").split():
            stripped = token.strip("\"',.()[]{}:;!?*")
            low = stripped.lower()
            # 缩写（大写形式）即使较短也保留：HBT、RCT、LED 等是强切题信号
            if _is_acronym(stripped) and len(stripped) <= 6:
                terms.add(low)
                continue
            if len(low) > 2 and low not in GENERIC_TERMS and not low.isdigit():
                terms.add(low)

    if spec is not None:
        for value in (spec.population, spec.intervention, spec.comparator,
                      spec.outcome, spec.neutral_query):
            add_text(value)
        for value in getattr(spec, "search_queries", ()) or ():
            add_text(value)
    for value in (dimension_queries or ()):
        add_text(value)
    for value in (extra or ()):
        add_text(value)
    return terms


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")


# ============================================================
# 方案 B：本地实体级预筛（只把「难判」的交给 LLM）
# ============================================================
# 动机（实测 2026-10）：
#   召回池中大量论文在**材料/器件**层面就与问题无关（如问 GaAs HBT 却召回
#   钙钛矿太阳能电池、IGZO TFT、SiC MOSFET）。这类论文无需任何 LLM 调用
#   即可安全排除——实测 3/3 全部正确，零误判。
#   而「material 与 device 都对上」或「部分对得上」的论文仍送 LLM，
#   因此**不改变最终纳入集合的统计口径**，只是省掉无意义的抽取。
#
# 关于 scispacy：其 NER 能力确实适合此项，但它依赖 spaCy，而 spaCy 要求
# httpx>=0.24，本项目锁定 httpx==0.13.3 —— 安装会破坏现有 HTTP 客户端
# （实测导致 4 个测试失败）。因此这里用一个**零依赖的轻量实体词典**实现
# 同样的判定逻辑，可通过 ``custom_entities`` 扩展领域词表。

# 材料体系（半导体/光电领域常见）
MATERIAL_LEXICON: frozenset[str] = frozenset({
    "gaas", "ingap", "gainp", "gaainp", "gap", "inp", "ingaas", "gainas",
    "algaas", "algan", "gan", "inn", "gann", "sic", "4h-sic", "6h-sic",
    "si", "silicon", "ge", "germanium", "sige", "sige", "sigesn",
    "perovskite", "igzo", "inzno", "zno", "ito", "ito",
    "mos2", "ws2", "graphene", "cnt", "carbon", "diamond",
    "in2o3", "sno2", "hfo2", "al2o3", "sio2", "si3n4",
    "inas", "gasb", "insb", "cdte", "cigs", "mote2", "wse2",
})

# 器件类型
DEVICE_LEXICON: frozenset[str] = frozenset({
    "hbt", "bipolar transistor", "heterojunction bipolar transistor",
    "hemt", "hfET", "hemt", "mesfet", "mosfet", "jfet", "fet", "finfet",
    "tft", "thin-film transistor", "led", "oled", "laser", "vcsel",
    "photodetector", "photodiode", "solar cell", "photovoltaic",
    "varactor", "diode", "thyristor", "igbt", "memristor", "transistor",
    "semiconductor", "device", "integrated circuit", "cmos",
})


def extract_entities(text: str,
                     custom_entities: Sequence[str] | None = None) -> dict[str, set[str]]:
    """从文本中抽取材料 / 器件实体（零依赖词典匹配）。

    匹配规则：
      - 归一化：小写、去标点、连字符转空格、合并空白；
      - 用**词边界**正则匹配，同时容忍复数（transistor / transistors）；
      - 多词实体（如 "thin-film transistor"）在归一化后变为空格分隔形式，
        需整体匹配。

    这比裸子串匹配严谨：避免 "si" 命中 "simulation" 之类的误判
    （实测该问题会导致器件统计严重失真）。
    """
    import re as _re
    raw = str(text or "").lower()
    # 连字符转空格后再去标点，使 "thin-film" 与 "thin film" 等价
    normalized = raw.replace("-", " ").replace("/", " ")
    normalized = _re.sub(r"[^a-z0-9\.\s]", " ", normalized)
    # 去掉句点（保留作词分隔），合并空白
    normalized = _re.sub(r"\.", " ", normalized)
    normalized = " " + _re.sub(r"\s+", " ", normalized).strip() + " "

    def hits(lexicon) -> set[str]:
        found = set()
        for term in lexicon:
            t = str(term).lower().replace("-", " ").replace("/", " ").strip()
            if not t:
                continue
            # 词边界 + 可选复数 s
            pattern = r"(?<![a-z0-9])" + _re.escape(t) + r"s?(?![a-z0-9])"
            if _re.search(pattern, normalized):
                found.add(str(term).lower())
        return found

    materials = hits(MATERIAL_LEXICON)
    devices = hits(DEVICE_LEXICON)
    if custom_entities:
        materials |= hits({str(t).lower() for t in custom_entities})
    return {"materials": materials, "devices": devices}


def entity_subject_terms(subject_terms: Sequence[str],
                         custom_entities: Sequence[str] | None = None) -> set[str]:
    """从 subject_terms 中过滤出**真正的实体词**，供本地预筛使用。

    实测问题：``build_subject_terms`` 会混入 ``base`` / ``current`` / ``gain`` /
    ``characteristics`` 这类通用词，若直接用于实体匹配会造成大面积假阳性
    （实测 45 篇被误判为"仍含目标词"）。

    这里只保留落在材料/器件词典里的词，或调用方显式声明的扩展实体。
    若过滤后为空，则返回空集 → 本地预筛自动禁用（保守回退）。
    """
    material_set = {m.lower() for m in MATERIAL_LEXICON}
    device_set = {d.lower() for d in DEVICE_LEXICON}
    custom = {str(c).strip().lower() for c in (custom_entities or ()) if str(c).strip()}

    kept = set()
    for term in subject_terms or ():
        t = str(term).strip().lower()
        if not t:
            continue
        if t in material_set or t in device_set or t in custom:
            kept.add(t)
    return kept


def local_entity_screen(papers: Sequence[dict],
                        subject_terms: Sequence[str],
                        custom_entities: Sequence[str] | None = None
                        ) -> tuple[list[dict], list[dict], dict]:
    """本地实体级预筛：把明显不相关的论文直接排除，其余送 LLM。

    判定规则（**保守**，只排除高置信无关项）：
      - 先用 ``entity_subject_terms`` 把 subject_terms 收敛为真正的实体词；
      - 论文中**既无**目标材料、**也无**目标器件 → 本地排除；
      - 其余（含只命中其一、或无法识别实体）→ 送 LLM 判定。

    保守策略保证：本地排除的一定是无关论文，不会误杀真正相关的文献，
    因此**最终纳入集合的统计口径不受影响**。

    返回：(保留送 LLM 的论文, 本地排除的论文, 统计信息)
    """
    targets = sorted(entity_subject_terms(subject_terms, custom_entities))
    if not targets:
        return list(papers), [], {"screened": len(papers), "local_excluded": 0,
                                  "sent_to_llm": len(papers), "disabled": True}

    target_set = set(targets)
    kept: list[dict] = []
    excluded: list[dict] = []
    entity_stats = {"with_material": 0, "with_device": 0}

    for paper in papers:
        text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
        ents = extract_entities(text, custom_entities)
        materials, devices = ents["materials"], ents["devices"]
        # 目标词可能同时出现在两本词典里，按字典取并集匹配
        mat_hit = bool(materials & target_set)
        dev_hit = bool(devices & target_set)
        # 也允许目标词直接以普通词形式命中（如 "ingap" 不在词典但出现在 subject_terms）
        if not (mat_hit or dev_hit):
            flat = " " + re.sub(r"[^a-z0-9\s]", " ", text.lower()) + " "
            dev_hit = any(f" {t} " in flat for t in target_set if len(t) > 3)

        if materials:
            entity_stats["with_material"] += 1
        if devices:
            entity_stats["with_device"] += 1

        if not mat_hit and not dev_hit:
            record = dict(paper)
            record["_local_exclusion_reason"] = "entity_mismatch"
            excluded.append(record)
        else:
            kept.append(paper)

    return kept, excluded, {
        "screened": len(papers),
        "local_excluded": len(excluded),
        "sent_to_llm": len(kept),
        **entity_stats,
        "disabled": False,
    }


def build_subject_terms(spec: "QuestionSpec | None",
                        question: str = "") -> set[str]:
    """抽取**研究对象词**（population / 缩写 / 材料体系），用于切题判定。

    这是方案 A 的核心：``annealing`` 这类通用方法词无法区分 HBT 与钙钛矿，
    但 ``hbt`` / ``gaas`` / ``ingap`` 可以。因此切题判定必须依赖研究对象词。

    注意：**不能用原始问题做词法抽取**——中文问题（如"保证退火效果的情况下…"）
    会被切出 "in效应" 这类无意义片段，污染判据。因此只从 PICO 与
    英文检索式提取，中文一律跳过。
    """
    subjects: set[str] = set()

    def add_text(value: str) -> None:
        raw = str(value or "")
        if not raw or _has_cjk(raw):
            # 中文（或中英混排）不做词法切分，避免产出噪声词
            return
        for token in raw.replace("/", " ").replace("-", " ").split():
            stripped = token.strip("\"',.()[]{}:;!?*")
            low = stripped.lower()
            if _is_acronym(stripped) and len(stripped) <= 6:
                subjects.add(low)
            elif len(low) > 3 and low not in GENERIC_TERMS and not low.isdigit():
                subjects.add(low)

    if spec is not None:
        # 优先使用 LLM 显式声明的具体研究对象（prompt 已明确禁止泛化词）
        for value in getattr(spec, "subject_terms", ()) or ():
            add_text(value)
        # population / outcome / neutral_query 是英文（由 build_question_spec 翻译）
        add_text(spec.population)
        add_text(spec.outcome)
        add_text(spec.neutral_query)
        for value in getattr(spec, "search_queries", ()) or ():
            add_text(value)
    add_text(question)
    return subjects


async def rerank_with_llm(
    papers: Sequence[dict],
    question: str,
    llm_factory,
    spec: "QuestionSpec | None" = None,
    candidate_limit: int = 60,
    target_count: int = 20,
) -> tuple[list[dict], dict[str, int]]:
    """LLM 精排：复刻 Consensus 三步法的“高精度模型只施用于 top N”。

    本地粗排先选出 candidate_limit 篇，再由 LLM 逐篇判定与问题的真实相关性，
    过滤掉“字面命中但主题无关”的论文（如把 quantum annealing 误判为热退火）。

    失败时完整回退到本地粗排结果，绝不因精排失败而丢失论文。
    """
    total = len(papers)
    if not papers:
        return [], {"retrieved": 0, "prefiltered": 0, "llm_reranked": 0, "final_selected": 0}

    candidates = list(papers)[:max(1, candidate_limit)]
    prefiltered = len(candidates)
    if len(candidates) <= target_count:
        return candidates, {
            "retrieved": total,
            "prefiltered": prefiltered,
            "llm_reranked": 0,
            "final_selected": len(candidates),
        }

    # 构造紧凑清单：编号 + 年份 + 标题 + 摘要片段（控制 token 体积）
    lines = []
    for idx, paper in enumerate(candidates, start=1):
        title = " ".join(str(paper.get("title") or "").split())[:180]
        abstract = " ".join(str(paper.get("abstract") or "").split())[:420]
        lines.append(f"[{idx}] ({paper.get('year') or 'n.d.'}) {title}\n    {abstract}")

    client = llm_factory()
    try:
        schema = (
            "Return one JSON object with a single key 'ranked', whose value is an array of "
            "objects {\"id\": <number>, \"relevance\": <0-10 integer>}. "
            "Score how directly each paper addresses the research question. "
            "10 = directly studies the exact intervention and outcome; "
            "5 = related method or adjacent population; "
            "0-2 = same words but a different topic (e.g. 'quantum annealing', "
            "unrelated materials, unrelated device types). "
            "Include EVERY index exactly once. Output strict JSON only."
        )
        prompt = (
            f"Research question: {question}\n\n"
            f"Rate each candidate paper's relevance to this question.\n"
            f"Be strict: papers that merely share vocabulary but study a different "
            f"phenomenon, material system, or device must receive a low score.\n\n"
            + "\n".join(lines)
        )
        raw = await client.complete_strict(
            [
                {"role": "system", "content": "You are a rigorous academic relevance assessor. " + schema},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
        rows = payload.get("ranked") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ValueError("rerank payload missing 'ranked' array")

        scores: dict[int, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("id"))
                rel = int(round(float(row.get("relevance"))))
            except (TypeError, ValueError):
                continue
            if 1 <= idx <= len(candidates):
                scores[idx] = max(0, min(10, rel))

        if not scores:
            raise ValueError("rerank produced no usable scores")

        # 未被打分的候选给中性分，避免因模型遗漏而丢失有价值论文
        ordered = sorted(
            range(1, len(candidates) + 1),
            key=lambda i: (-scores.get(i, 3), i),
        )
        selected = [candidates[i - 1] for i in ordered[:target_count]]
        return selected, {
            "retrieved": total,
            "prefiltered": prefiltered,
            "llm_reranked": len(scores),
            "final_selected": len(selected),
        }
    except Exception:
        # 精排失败：回退本地粗排顺序（绝不丢弃论文）
        selected = candidates[:target_count]
        return selected, {
            "retrieved": total,
            "prefiltered": prefiltered,
            "llm_reranked": 0,
            "final_selected": len(selected),
        }
    finally:
        close = getattr(client, "aclose", None)
        if close:
            try:
                await close()
            except Exception:
                pass


_COMPLEX_QUESTION_MARKERS = (
    "compare",
    "comparison",
    "versus",
    "vs ",
    "trade-off",
    "tradeoff",
    " and ",
    " as well as ",
    "along with",
    "also",
    "multiple",
    "different",
    "overview",
    "review",
    "比较",
    "对比",
    "优劣",
    "不同",
    "多种",
    "以及",
    "并且",
    "与",
    "和",
    "机制",
    "研究现状",
    "总结",
)


def is_complex_question(question: str,
                        spec: "QuestionSpec | None" = None) -> bool:
    """Detect broad/multi-part questions that benefit from sub-questions."""
    text = (question or "").strip()
    if not text:
        return False
    lowered = text.casefold()
    if any(sep in text for sep in ("；", ";", "、")):
        return True
    if text.count("?") + text.count("？") > 1:
        return True
    if any(marker in lowered for marker in _COMPLEX_QUESTION_MARKERS):
        return True
    if spec is not None and spec.question_type == "open_ended" and len(text.split()) >= 8:
        return True
    return False


async def decompose_question(question: str, llm_factory,
                             max_subquestions: int = 8) -> list[str]:
    """把复杂问题拆成 3-8 个子问题（P3）；简单问题返回自身。"""
    cap = max(3, min(8, int(max_subquestions or 6)))
    client = llm_factory()
    try:
        raw = await client.complete_strict(
            [
                {"role": "system", "content": (
                    f"Split the research question into 3-{cap} concrete sub-questions. "
                    "Return a JSON object with a single key 'subquestions' containing "
                    "an array of strings. If the question is simple, return an array "
                    "with the original question only."
                )},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        payload = json.loads(raw)
        candidates = []
        if isinstance(payload, list):
            candidates.extend(str(v) for v in payload if str(v).strip())
        elif isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    candidates.extend(str(v) for v in value if str(v).strip())
                elif isinstance(value, str) and value.strip():
                    candidates.append(value)
        cleaned: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            item = candidate.strip()
            key = item.casefold()
            if item and key not in seen:
                seen.add(key)
                cleaned.append(item)
        if not cleaned:
            return [question]
        if len(cleaned) < 3:
            return [question]
        return cleaned[:cap]
    except Exception:
        return [question]
    finally:
        close = getattr(client, "aclose", None)
        if close:
            try:
                await close()
            except Exception:
                pass


@dataclass(frozen=True)
class ConsensusAnalysis:
    evidence: tuple[EvidenceItem, ...]
    score: ConsensusScore
    narrative: str
    question_spec: QuestionSpec = None  # type: ignore[assignment]
    stop_reason: str = ""
    # Meter Snapshot 质量概览（Recency / Methods / Citations / Venues）
    snapshot: dict = field(default_factory=dict)
    # 开放问题的结构化证据综述（含证据表/方法族/分歧/空白）
    evidence_review: dict = field(default_factory=dict)
    # 引用与证据核验结果（DOI/URL/摘录可追溯性 + 质量门控）
    verification: dict = field(default_factory=dict)
    subquestions: tuple[str, ...] = ()
    subquestion_reviews: tuple[dict, ...] = ()


    def __post_init__(self):
        if self.question_spec is None:
            object.__setattr__(
                self,
                "question_spec",
                QuestionSpec.fallback("", "open_ended"),
            )


# ============================================================
# 抽取用的 JSON Schema（紧凑表述）
# ============================================================
# 背景（实测 2026-10）：
#   1. DeepSeek API **不支持** response_format={"type":"json_schema"}（返回
#      "This response_format type is unavailable now"），因此 outlines 那类
#      **解码期约束**无法在远程 API 上生效，不能照搬。
#   2. 但**给每个字段加明确字数上限**能显著缩短输出：
#        实测同一摘要 3 轮均值：
#          冗长句描述        : 265 completion tokens（claim 124c, excerpt 176c）
#          逐行 key + 字数上限: 142 completion tokens（claim  60c, excerpt  37c）  ← -46%
#          极简描述          : 171 completion tokens
#      18 个字段全部保留，抽取口径不变。
#   3. 因此本方案的收益来自「约束值的长度」而非「削减字段」——这正是用户
#      要求避免的副作用（不减少抽取量）。
EVIDENCE_SCHEMA_COMPACT = (
    "Extract evidence as JSON. One line per key; keep every value SHORT.\n"
    "eligible: bool\n"
    "exclusion_reason: <=6 words or \"\"\n"
    "study_type: rct|non_randomized_experiment|experimental|cohort|case_control|"
    "cross_sectional|observational|in_vitro|bench_test|simulation|theory|review|"
    "systematic_review|meta_analysis|commentary|unknown\n"
    "population: <=5 words\n"
    "intervention: <=5 words\n"
    "comparator: <=3 words or \"not reported\"\n"
    "outcome: <=4 words\n"
    "stance: support|oppose|mixed|uncertain|not_applicable\n"
    "claim: <=15 words\n"
    "effect_direction: <=3 words\n"
    "sample_size: <=4 words\n"
    "statistical_evidence: <=6 words\n"
    "quality: high|medium|low|unknown\n"
    "directness: float 0..1\n"
    "confidence: float 0..1\n"
    "excerpt: <=20 words, verbatim from the abstract\n"
    "limitations: array of <=2 short tags\n"
    "cohort_key: <=3 words or \"\"\n"
    "Return JSON only, no prose."
)

# 旧的冗长版本保留，供需要更丰富叙述时切换（token 约多 46%）
EVIDENCE_SCHEMA_VERBOSE = (
    "Return one JSON object with keys: eligible (boolean), exclusion_reason, "
    "study_type (rct|non_randomized_experiment|experimental|cohort|case_control|"
    "cross_sectional|observational|in_vitro|bench_test|simulation|theory|review|"
    "systematic_review|meta_analysis|commentary|unknown), population, "
    "intervention, comparator, outcome, stance "
    "(support|oppose|mixed|uncertain|not_applicable), claim, effect_direction, "
    "sample_size, statistical_evidence, quality (high|medium|low|unknown), "
    "directness (0..1), confidence (0..1), excerpt, limitations (array), "
    "and cohort_key. Use 'not reported' for facts absent from the abstract. "
    "The excerpt must be copied exactly from the supplied abstract."
)

# batch 抽取用（要求按序返回数组）
EVIDENCE_SCHEMA_BATCH = (
    "Return a JSON object with key 'results': an array of objects, one per paper, "
    "in the same order as the input. Each object uses the keys below:\n"
    + EVIDENCE_SCHEMA_COMPACT
)


def evidence_schema(compact: bool = True, batch: bool = False) -> str:
    """返回抽取用的 schema 文本。"""
    if batch:
        return EVIDENCE_SCHEMA_BATCH
    return EVIDENCE_SCHEMA_COMPACT if compact else EVIDENCE_SCHEMA_VERBOSE


class ConsensusAnalyzer:
    """Use an LLM for bounded evidence extraction, never for score calculation."""

    def __init__(self, llm_factory, max_concurrency: int = 4,
                 compact_schema: bool = True, max_extract_tokens: int = 1200):
        self.llm_factory = llm_factory
        self.max_concurrency = max(1, max_concurrency)
        # 紧凑 schema：实测输出 token -46%（265→142），18 个字段不变
        self.compact_schema = compact_schema
        # 单篇抽取的输出 token 上限。实测 3 篇/组的批量抽取约需 400-500 tokens，
        # 1200 留 2-3 倍富余可避免截断重试；设 8192 反而最慢（17.2s vs 13.8s），
        # 因为模型更容易在宽松预算下生成冗余内容。
        self.max_extract_tokens = max_extract_tokens
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

        study_type = _normalize_study_type(self._text(payload, "study_type", "unknown"))
        stance = _normalize_stance(self._text(payload, "stance", "uncertain"))
        quality = _normalize_quality(self._text(payload, "quality", "unknown"))

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
        claim = self._text(payload, "claim", "") if excerpt else ""

        return EvidenceItem(
            paper_id=str(paper.get("id") or paper.get("doi") or paper.get("title") or ""),
            title=paper.get("title") or "Untitled",
            eligible=eligible,
            exclusion_reason=self._text(payload, "exclusion_reason", ""),
            study_type=study_type,
            stance=stance,
            claim=claim,
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

        schema = evidence_schema(compact=self.compact_schema)
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
                        # 逐篇抽取同样关闭隐藏推理（实测快 2.35 倍，判定质量不变）
                        thinking=False,
                        # 输出上限：精简 schema 实测仅需 ~142 tokens，700 足够
                        # 容纳全部 18 个字段并留有富余，同时可尽早截断跑偏输出
                        max_tokens=self.max_extract_tokens,
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

    async def extract_batch(self, question: str,
                            papers: Sequence[dict]) -> list[EvidenceItem]:
        """批量提取多篇论文证据（每组 3~5 篇），失败或超限时平滑回退到逐篇抽取。"""
        if not papers:
            return []
        if len(papers) == 1 or self._stop_reason:
            return [await self.extract_one(question, papers[0])]

        valid_entries = []
        for p in papers:
            abstract = (p.get("abstract") or "").strip()
            if not abstract:
                valid_entries.append((p, self._unavailable(p, "no abstract available")))
            else:
                valid_entries.append((p, None))

        pending_papers = [p for p, item in valid_entries if item is None]
        if not pending_papers:
            return [item for _, item in valid_entries]

        # 构造批量抽取 Prompt
        papers_text_list = []
        for idx, p in enumerate(pending_papers, start=1):
            papers_text_list.append(
                f"--- Paper [{idx}] ---\n"
                f"Title: {p.get('title', '')}\n"
                f"Abstract: {p.get('abstract', '')}"
            )
        combined_text = "\n\n".join(papers_text_list)

        schema = evidence_schema(compact=self.compact_schema, batch=True)

        messages = [
            {"role": "system",
             "content": "Extract evidence conservatively from each supplied abstract. " + schema},
            {
                "role": "user",
                "content": f"Research question: {question}\n\n{combined_text}",
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
            return [await self.extract_one(question, p) for p in papers]

        try:
            raw = await client.complete_strict(
                messages,
                temperature=0,
                response_format={"type": "json_object"},
                # 批量抽取是结构化分类任务，关闭隐藏推理可提速约 2.35 倍且不影响结果
                thinking=False,
                # 批量时按篇数线性放宽上限（每篇 ~250 tokens 富余量）
                max_tokens=self.max_extract_tokens * max(1, len(pending_papers)),
            )
            data = json.loads(str(raw))
            results_arr = data.get("results")
            if not isinstance(results_arr, list) and isinstance(data, list):
                results_arr = data
            if isinstance(results_arr, list) and len(results_arr) == len(pending_papers):
                parsed_items = []
                for p, p_raw in zip(pending_papers, results_arr):
                    try:
                        parsed_items.append(self._parse_evidence(json.dumps(p_raw), p))
                    except Exception:
                        parsed_items.append(self._unavailable(p, "batch item parse error"))
                # 组装完整返回
                idx_parsed = 0
                final_results = []
                for p, pre_item in valid_entries:
                    if pre_item is not None:
                        final_results.append(pre_item)
                    else:
                        final_results.append(parsed_items[idx_parsed])
                        idx_parsed += 1
                return final_results
        except Exception:
            pass
        finally:
            await self._close_client(client)

        # 降级方案：若批量解析失败，逐篇提取（单测模拟或非标准大模型保证稳定）
        single_results = []
        for p in papers:
            single_results.append(await self.extract_one(question, p))
        return single_results

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
        if score.direction == "insufficient_evidence":
            # 统计量有效但样本量不足：必须明确标注，避免把不可靠数字当作定论
            return (
                f"当前方向性证据不足（有效独立研究 "
                f"{score.effective_independent_studies:.1f} 篇，低于 "
                f"{MIN_DIRECTIONAL_STUDIES} 篇门槛），暂不给出共识结论。"
                f"现有证据方向一致率为 {score.directional_agreement:.1%}，"
                "仅供参考；建议扩充检索范围后重新分析。"
            )
        return (
            f"主要证据方向为 {score.direction}；方向一致率为 "
            f"{score.directional_agreement:.1%}，证据确定性为 "
            f"{score.evidence_certainty:.1%}。共识分为 {score.score}/100。"
        )

    @staticmethod
    def format_apa_references(papers: Sequence[dict]) -> list[str]:
        """将论文元数据转换为符合 APA 7th 规范的标准学术参考文献列表字符串。"""
        formatted = []
        for p in papers:
            # 作者解析
            authors = p.get("authors") or []
            author_str = ""
            if isinstance(authors, list) and authors:
                names = []
                for a in authors:
                    name = a.get("name", "") if isinstance(a, dict) else str(a)
                    name = name.strip()
                    if not name:
                        continue
                    parts = name.split()
                    if len(parts) > 1:
                        # 姓氏, 名字首字母.
                        surname = parts[-1]
                        initials = " ".join(f"{part[0]}." for part in parts[:-1] if part)
                        names.append(f"{surname}, {initials}")
                    else:
                        names.append(name)
                if len(names) == 1:
                    author_str = names[0]
                elif len(names) == 2:
                    author_str = f"{names[0]}, & {names[1]}"
                elif len(names) > 2:
                    if len(names) > 6:
                        author_str = ", ".join(names[:6]) + f", ... & {names[-1]}"
                    else:
                        author_str = ", ".join(names[:-1]) + f", & {names[-1]}"
            if not author_str:
                author_str = "Unknown Author"

            year = str(p.get("year") or "n.d.")
            title = (p.get("title") or "Untitled").strip().rstrip(".")
            venue = (p.get("venue") or "").strip()
            doi = (p.get("doi") or "").strip()
            url = (p.get("url") or "").strip()

            ref = f"{author_str} ({year}). {title}."
            if venue:
                ref += f" *{venue}*."
            if doi:
                clean_doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                ref += f" https://doi.org/{clean_doi}"
            elif url and url.startswith("http"):
                ref += f" {url}"
            formatted.append(ref)
        return formatted

    @staticmethod
    def _deterministic_narrative(evidence: Sequence[EvidenceItem],
                                 score: ConsensusScore,
                                 question: str,
                                 mode: str = "deep",
                                 question_type: str = "open_ended") -> str:
        """Build a citation-grounded fallback using only extracted evidence."""
        lines = [f"## {question}：证据摘要"]
        included = [
            (index, item)
            for index, item in enumerate(
                (item for item in evidence if item.eligible), start=1
            )
        ]
        if not included:
            lines.append("当前没有可纳入的结构化证据，无法形成研究结论。")
            return "\n\n".join(lines)

        lines.extend([
            "以下条目仅整理纳入论文中可追溯的结构化摘要信息，不外推未提供的实验事实。",
            "",
            "### 纳入证据",
        ])
        for index, item in included:
            claim = (item.claim or "").strip()
            excerpt = (item.excerpt or "").strip()
            entry = (
                f"- [{index}] **{item.title}** "
                f"（{STUDY_LABELS.get(item.study_type, item.study_type)}；立场：{item.stance}）"
            )
            if claim and claim != "not reported" and excerpt:
                entry += f"：{claim}"
            if excerpt:
                entry += f" 摘录：\"{excerpt}\""
            lines.append(entry)

        lines.append("")
        if question_type == "yes_no":
            if score.direction == "insufficient_evidence" or score.score is None:
                lines.append(
                    f"合格的有效独立研究不足 {MIN_DIRECTIONAL_STUDIES} 篇，"
                    "当前不提供共识分数；建议扩充检索。"
                )
            else:
                lines.append(
                    f"结构化证据的主要方向为 {score.direction}；"
                    f"方向一致率 {score.directional_agreement:.1%}，"
                    f"证据确定性 {score.evidence_certainty:.1%}，"
                    f"共识分 {score.score}/100。"
                )
        else:
            lines.append("该问题为开放式问题，论文之间不视为投票关系，因此不计算共识百分比。")
        return "\n\n".join(lines)

    async def analyze(self, question: str,
                      papers: Sequence[dict],
                      question_spec: QuestionSpec | None = None,
                      search_stats: dict | None = None,
                      use_batch: bool = False,
                      mode: str = "deep",
                      subquestions: Sequence[str] | None = None,
                      subquestion_papers: dict[str, Sequence[dict]] | None = None,
                      subquestion_search_stats: dict[str, dict] | None = None) -> ConsensusAnalysis:
        self._stop_reason = None
        spec = question_spec or QuestionSpec.fallback(
            question, heuristic_question_type(question)
        )
        if subquestions is not None:
            seen_subquestions: set[str] = set()
            cleaned_subquestions = []
            for value in subquestions:
                text = str(value).strip()
                key = text.casefold()
                if text and key not in seen_subquestions:
                    seen_subquestions.add(key)
                    cleaned_subquestions.append(text)
            resolved_subquestions = tuple(cleaned_subquestions[:8])
        elif is_complex_question(question, spec):
            resolved_subquestions = tuple(
                await decompose_question(question, self.llm_factory)
            )
        else:
            resolved_subquestions = ()
        extracted: list[EvidenceItem] = []


        # 当明确开启 use_batch 时采用分组批量抽取；默认按流控逐篇处理以保证向后兼容与精准熔断
        if use_batch and len(papers) > 1:
            # chunk_size 决定单次 LLM 调用的 prompt 体积与并发度。
            # 实测 5 篇/组 + 30 篇 = 6 轮，每组 prompt 含 5 段完整摘要，
            # 在推理型模型上单轮可达 30-60s，是 analyze 阶段的主要耗时来源。
            # 3 篇/组可获得 30 篇 → 10 轮并行，单轮 prompt 更小、整体更快且更稳。
            chunk_size = 3
            chunks = [papers[i:i + chunk_size] for i in range(0, len(papers), chunk_size)]
            batch_tasks = [self.extract_batch(question, chunk) for chunk in chunks]
            chunk_results = await asyncio.gather(*batch_tasks)
            for sublist in chunk_results:
                extracted.extend(sublist)
        else:
            semaphore = asyncio.Semaphore(self.max_concurrency)

            async def bounded(paper):
                async with semaphore:
                    if self._stop_reason:
                        return self._unavailable(paper, self._stop_reason)
                    return await self.extract_one(question, paper)

            extracted = list(await asyncio.gather(*(bounded(paper) for paper in papers)))

        evidence = self._apply_duplicate_cohort_weights(extracted)
        # 设计适配度：按问题域选用权重表（临床奖励 RCT，材料/器件奖励实测台架）。
        # 注意此处必须用参数 question_spec——局部变量 spec 在下文才绑定。
        score = calculate_consensus(
            evidence,
            domain=getattr(question_spec, "domain", "general") if question_spec else "general",
        )

        # 至少 5 篇合格方向性证据才显示 Meter（P1）
        score = apply_min_study_gate(
            score,
            evidence,
            domain=getattr(question_spec, "domain", "general") if question_spec else "general",
        )

        subquestion_reviews: list[dict] = []
        if subquestion_papers is not None:
            subquestion_semaphore = asyncio.Semaphore(self.max_concurrency)

            async def _extract_subquestion_paper(subquestion: str, paper: dict):
                async with subquestion_semaphore:
                    return await self.extract_one(subquestion, paper)

            async def _analyze_subquestion(subquestion: str) -> dict:
                subpapers = list(subquestion_papers.get(subquestion, ()))
                bounded_papers = subpapers[:MAX_SUBQUESTION_PAPERS]
                search_stats_for_question = dict(
                    (subquestion_search_stats or {}).get(subquestion, {})
                )
                subquestion_domain = str(
                    search_stats_for_question.get("domain")
                    or getattr(spec, "domain", "general")
                )
                sub_evidence = await asyncio.gather(*(
                    _extract_subquestion_paper(subquestion, paper)
                    for paper in bounded_papers
                ))
                weighted_sub_evidence = self._apply_duplicate_cohort_weights(sub_evidence)
                domain = subquestion_domain
                qualified_counts = qualified_stance_counts(
                    weighted_sub_evidence, domain=domain
                )
                sub_review = build_evidence_review(
                    weighted_sub_evidence,
                    bounded_papers,
                    domain=domain,
                )
                return {
                    "question": subquestion,
                    "screened_paper_ids": [
                        str(p.get("id") or p.get("doi") or p.get("title") or "")
                        for p in bounded_papers
                    ],
                    "search_stats": search_stats_for_question,
                    "paper_ids": [
                        item.paper_id for item in weighted_sub_evidence if item.eligible
                    ],
                    "effective_independent_studies": qualified_counts[
                        "effective_independent_studies"
                    ],
                    "evidence_review": sub_review,
                }

            subquestion_reviews = list(await asyncio.gather(*(
                _analyze_subquestion(subquestion)
                for subquestion in resolved_subquestions
            )))

        fixed_values = (
            asdict(score)
            if spec.question_type == "yes_no"
            else {"raw_counts": score.raw_counts}
        )
        stats = search_stats or {}
        meth_stats = {
            "检索总量_retrieved": stats.get("retrieved", len(papers)),
            "深度分析量_deep_analyzed": len(papers),
            "最终纳入量_included": sum(1 for item in evidence if item.eligible),
            "排除量_excluded": sum(1 for item in evidence if not item.eligible),
        }
        included_evidence = [item for item in evidence if item.eligible]
        evidence_summary = [
            {
                "reference_number": index,
                "title": item.title,
                "study_type": item.study_type,
                "stance": item.stance,
                "claim": item.claim,
                "quality": item.quality,
                "limitations": item.limitations,
            }
            for index, item in enumerate(included_evidence, start=1)
        ]
        reference_list = "\n".join(
            f"[{index}] {item.title} ({item.study_type})"
            for index, item in enumerate(included_evidence, start=1)
        )

        # 双轨 Prompt 机制：Pro 模式（实用工艺对比表）与 Deep 模式（系统深度综述）
        if mode == "pro":
            system_instruction = (
                "你是 Consensus 专业学术证据综述引擎（Pro 模式）。基于给定的结构化证据，"
                "生成一份直击工程/实验技术与参数影响对比的专业学术报告（对标 consensus.app Pro 报告风格）。\n\n"
                "【严格禁令 - 必须遵循】：\n"
                "- 绝对禁止出现任何 AI 自述、检索过程描述（如“我为您搜索了”、“根据检索结果”、“经过对XX篇论文分析”等）。"
                "检索统计已由系统外部图表展示，正文必须直接、客观陈述学术与实验事实！\n"
                "- 引用论文必须统一用 [编号]（如 [1] 或 [1][3]），系统会自动渲染为 Author-Year 徽章。\n\n"
                "严格按以下结构输出：\n\n"
                "## 核心背景与技术概要\n"
                "100-200字精炼阐明该工艺/技术在实际器件或场景中的核心作用、关键物理/化学过程，直接引用 [编号]。\n\n"
                "## 具体工艺方式与参数影响对照\n"
                "输出清晰的对比表格（对照 consensus.app 格式）：\n"
                "| 现象/参数 | 退火/工艺影响概要 | 引用 |\n"
                "|---|---|---|\n"
                "| 参数A (如增益/缺陷/速率) | 具体工艺下的正负向变化与数据范围 | [1][2] |\n"
                "| 参数B | 具体工艺下的正负向变化与数据范围 | [3] |\n\n"
                "## 物理机制与工艺折中\n"
                "深入分析 2-3 个核心折中点（如去氢与少子寿命的平衡、温度-时间窗口选择、单步 vs 多步工艺的优劣），引用编号。\n\n"
                "## 结论与工艺建议\n"
                "给出针对具体应用/工业实践的明确折中建议与推荐工艺窗口，引用编号。\n\n"
                "## 核心参考文献\n"
                "按 [编号] 标题 格式列出。"
            )

        else:
            system_instruction = (
                "你是 Consensus 学术证据综述引擎（Deep 模式）。基于给定的结构化证据，"
                "生成一份 Consensus Deep Review 风格的中文深度学术综述报告（对标 consensus.app 官方深度研报）。\n\n"
                "【严格禁令 - 必须遵循】：\n"
                "- 绝对禁止出现任何 AI 自述或检索过程描述（如“我为你搜索了”、“根据系统分析”等）。"
                "文献数量与漏斗完全由前端独立图表单独呈现，正文必须直接以客观学术视角论述事实！\n"
                "- 引用论文统一标注为 [编号]（如 [1] 或 [1][3]）。\n\n"
                "严格按以下 Markdown 结构输出：\n\n"
                "## 1. 引言\n"
                "150-250字背景与现状，说明研究问题的核心意义、应用场景与科学难点。引用论文用 [编号]。\n\n"
                "## 2. 方法\n"
                "仅写 1 段简明方法概述（说明本综述基于多数据源去重与实证标准严格筛选），不要自述对话过程。\n\n"
                "## 3. 结果\n"
                "按 3.1 / 3.2 / 3.3 等子主题递进组织（2-4 个具体子主题，例如不同工艺效果、机理与性能折中、结构优化等）。"
                "每个子主题写一段深度的证据综合，必须紧密结合实验细节并引用论文编号 [n]。\n\n"
                "## 4. 讨论\n"
                "先写 2-3 段综合讨论（分析机理、共识与争议分歧点，引用编号）。\n"
                "然后输出核心结论证据强度汇总表（对照 consensus.app 格式）：\n"
                "| Claim | Evidence Strength | Reasoning | Papers |\n"
                "|---|---|---|---|\n"
                "| 主要结论声明 | Strong/Moderate/Weak | 核心判断与实验依据 | [1][3] |\n\n"
                "## 5. 结论\n"
                "综合结论段落（150-250字，总结最佳实践与折中方案，引用编号）。\n\n"
                "### Research Gaps\n"
                "仅在纳入证据中存在可复核计数时才输出分布表；每个数字必须能从纳入证据逐条复算。"
                "证据不可比或未报告时明确写明，不得填造数值或模板示例。\n\n"
                "### Open Research Questions\n"
                "列出未来值得深入探索的开放科学问题及其重要性说明（对照 consensus.app 格式）：\n"
                "| Question | Why |\n"
                "|---|---|\n"
                "| 关键待解决问题1 | 重要性说明与潜在应用价值 |\n"
                "| 关键待解决问题2 | 重要性说明与潜在应用价值 |\n\n"
                "## 核心参考文献\n"
                "按 [编号] 标题 格式列出。"
            )

        if spec.question_type != "yes_no":
            system_instruction = (
                "你是学术 Evidence Review 引擎。针对开放式问题，只按给定的结构化证据与子问题独立证据集"
                "进行定性综合。每个结论必须引用纳入文献编号 [n]，并区分结果、方法、分歧和局限。"
                "不得输出共识分、方向一致率、支持/反对百分比，或把论文数量解释为投票比例；"
                "不得引用排除文献，不得臆造实验数据、研究空白计数、因果关系或建议。"
                "证据不足时明确说未报告/无法判断。"
            )
        else:
            system_instruction += (
                "\n\n【证据边界】：所有事实与数字必须能在纳入证据及其摘录中追溯；"
                "不得引用排除文献，不得复制模板示例数字或补造研究空白计数。"
                "无法由纳入证据支持的细节应明确标为未报告。"
            )

        prompt = (
            system_instruction
            + "\n\n统计值（固定）：\n"
            + json.dumps(fixed_values, ensure_ascii=False, sort_keys=True)
            + "\n筛选统计：\n"
            + json.dumps(meth_stats, ensure_ascii=False, sort_keys=True)
            + "\n问题解析（PICO）：\n"
            + json.dumps(asdict(spec), ensure_ascii=False, sort_keys=True)
            + "\n\n纳入论文（编号 → 标题/类型/立场/质量）：\n"
            + json.dumps(evidence_summary, ensure_ascii=False)
            + "\n\n参考文献列表：\n"
            + reference_list
        )
        if subquestion_reviews:
            prompt += (
                "\n\n子问题独立检索与筛选结果（每个子问题使用独立候选集；不得跨组借用证据）：\n"
                + json.dumps(subquestion_reviews, ensure_ascii=False, default=str)
            )
        if needs_search_expansion(score):
            # 证据不足时，禁止模型把不可靠的统计值当结论陈述
            prompt += (
                "\n\n【重要约束】：direction 为 insufficient_evidence，表示"
                "有效独立研究方向性证据低于门槛，统计量不可作为结论依据。"
                "请在综述中明确指出证据不足与研究空白，不得给出确定性结论，"
                "也不要引用或复述共识分数。"
            )
        # 基础叙述：高保真双模确定性学术综述（对标 Consensus 官网原版报告，绝不自述检索过程）
        narrative = self._deterministic_narrative(
            evidence,
            score,
            question,
            mode=mode,
            question_type=spec.question_type,
        )
        # 无论抽取是否失败，都尝试生成 LLM Deep Review 综述
        try:
            client = self.llm_factory()
            try:
                generated = await client.complete_strict(
                    [
                        {"role": "system", "content": "Synthesize objective academic evidence without meta-talk or filler."},
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
        except Exception:
            pass
        evidence_review = build_evidence_review(
            evidence, papers, domain=getattr(spec, "domain", "general")
        )
        evidence_review["subquestions"] = subquestion_reviews
        return ConsensusAnalysis(
            evidence, score, narrative, spec, self._stop_reason or "",
            build_snapshot(evidence, papers, domain=getattr(spec, "domain", "general")),
            evidence_review,
            build_verification(evidence, papers, score),
            resolved_subquestions,
            tuple(subquestion_reviews),
        )


# ============================================================
# 引用与证据核验（借鉴 PaperPilot 的 verification 设计）
# ============================================================
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.I)

# 核验状态语义
VERIFY_VERIFIED = "verified"
VERIFY_WARNINGS = "plausible_with_warnings"
VERIFY_UNVERIFIED = "unverified"
VERIFY_EXCLUDED = "excluded"


def verify_evidence(
    evidence: Sequence["EvidenceItem"],
    papers: Sequence[dict] | None = None,
) -> dict:
    """核验每条证据记录的可追溯性，防止报告出现"编造的引用"。

    核验项：
      1. **DOI 格式**——是否符合 ``10.xxxx/...`` 规范（可抓出模型幻觉出的 DOI）
      2. **URL 元数据是否存在**（不发起网络请求，不验证链接可访问性）
      3. **原文摘录**——excerpt 必须能在摘要中定位（防止编造证据）
      4. **摘要来源**——标记 evidence_basis（abstract / metadata only）
      5. **方向性证据有效性**——stance 是否可判定

    返回逐条记录 + 汇总指标，供 UI 展示与质量门控使用。
    """
    meta: dict[str, dict] = {}
    for paper in (papers or ()):
        key = str(paper.get("id") or paper.get("doi") or paper.get("title") or "")
        if key:
            meta[key] = paper

    records: list[dict] = []
    for item in evidence:
        raw = meta.get(item.paper_id) or {}
        issues: list[str] = []

        # 1) DOI 格式核验
        doi = str(raw.get("doi") or "").strip()
        if doi:
            doi_status = "format_valid" if DOI_PATTERN.match(doi) else "format_invalid"
            if doi_status == "format_invalid":
                issues.append("invalid_doi_format")
        else:
            doi_status = "not_available"

        # 2) URL 元数据存在性；此处不验证网络可达性
        url = str(raw.get("url") or "").strip()
        url_status = "present" if url else "missing"
        url_validation = "not_performed" if url else "not_available"
        if not url:
            issues.append("missing_url")

        # 3) 摘录可定位性：excerpt 必须出现在摘要中
        abstract = str(raw.get("abstract") or "")
        excerpt = str(item.excerpt or "").strip()
        if excerpt:
            if abstract and excerpt[:60] in abstract:
                excerpt_status = "traceable"
            elif "untraceable_excerpt" in (item.limitations or ()):
                excerpt_status = "untraceable"
                issues.append("untraceable_excerpt")
            else:
                excerpt_status = "unchecked"
        else:
            excerpt_status = "absent"

        # 4) 证据来源级别
        if abstract and len(abstract) >= 80:
            evidence_basis = "abstract"
        elif abstract:
            evidence_basis = "short_abstract"
            issues.append("short_abstract")
        else:
            evidence_basis = "metadata_only"
            issues.append("no_abstract")

        # 5) 立场可判定性
        if item.stance in {"not_applicable", ""} or item.stance not in STANCE_VALUES:
            issues.append("undetermined_stance")

        if not item.eligible:
            status = VERIFY_EXCLUDED
        elif issues:
            status = VERIFY_WARNINGS
        else:
            status = VERIFY_VERIFIED

        records.append({
            "paper_id": item.paper_id,
            "title": item.title,
            "status": status,
            "doi_status": doi_status,
            "url_status": url_status,
            "url_validation": url_validation,
            "excerpt_status": excerpt_status,
            "evidence_basis": evidence_basis,
            "scifact_label": scifact_label(item.stance),
            "issues": issues,
        })

    total = len(records)
    verified = sum(1 for r in records if r["status"] == VERIFY_VERIFIED)
    warned = sum(1 for r in records if r["status"] == VERIFY_WARNINGS)
    unverified = sum(1 for r in records if r["status"] == VERIFY_UNVERIFIED)
    excluded = sum(1 for r in records if r["status"] == VERIFY_EXCLUDED)
    with_excerpt = sum(1 for r in records if r["excerpt_status"] == "traceable")

    # 质量门控：这些情况必须在报告里明确警告
    warnings: list[str] = []
    if total:
        if any(r["doi_status"] == "format_invalid" for r in records):
            warnings.append("存在 DOI 格式非法的引用，可能为模型编造")
        if verified / total < 0.5:
            warnings.append("超过一半证据未通过完整核验，结论可靠性受限")
        if with_excerpt / total < 0.5:
            warnings.append("多数证据缺少可定位的原文摘录")
        if any(r["evidence_basis"] == "metadata_only" for r in records):
            warnings.append("部分证据仅有元数据、无摘要依据")

    return {
        "records": records,
        "total": total,
        "verified": verified,
        "plausible_with_warnings": warned,
        "unverified": unverified,
        "excluded": excluded,
        "traceable_excerpts": with_excerpt,
        "verification_warnings": warnings,
    }


def build_verification(evidence: Sequence["EvidenceItem"],
                       papers: Sequence[dict] | None = None,
                       score: "ConsensusScore | None" = None) -> dict:
    """核验 + 质量门控的组合入口（供 analyze 与 UI 使用）。"""
    result = verify_evidence(evidence, papers)
    result["gate"] = build_quality_gate(result, score=score)
    return result


def build_quality_gate(verification: dict,
                       score: "ConsensusScore | None" = None,
                       min_papers: int = MIN_DIRECTIONAL_STUDIES) -> dict:
    """综合质量门控：判断本次分析是否足以支撑结论。

    借鉴 PaperPilot 的 quality gate：不只给结论，还要给出
    「为什么这个结论可能不靠谱」以及可操作的改进建议。
    """
    issues: list[str] = []
    recommendations: list[str] = []

    total = int(verification.get("total") or 0)
    verified = int(verification.get("verified") or 0)
    excluded = int(verification.get("excluded") or 0)

    if total == 0:
        issues.append("no_evidence")
        recommendations.append("扩充检索：放宽关键词或改用更常见的材料/器件名称")
    else:
        if verified < min_papers:
            issues.append("too_few_verified_papers")
            recommendations.append(
                f"仅 {verified} 篇通过核验（建议 >= {min_papers} 篇）；"
                "建议扩充检索范围或放宽纳入标准"
            )
        if verified / total < 0.5:
            issues.append("low_verification_rate")
            recommendations.append("不足一半证据通过完整核验，暂不应依赖报告结论")
        if total and excluded / total > 0.5:
            issues.append("high_exclusion_rate")
            recommendations.append("过半候选被排除，建议校准问题表述或检索式")
        if any("DOI 格式非法" in w for w in (verification.get("verification_warnings") or [])):
            issues.append("invalid_doi_present")
            recommendations.append("剔除 DOI 格式非法的引用，避免传播错误文献")

    if score is not None and score.direction == "insufficient_evidence":
        issues.append("insufficient_evidence")
        recommendations.append("证据不足，不应给出确定性结论")

    return {
        "issues": issues,
        "recommendations": recommendations,
        "pass": not issues,
        "verification": {
            "total": total,
            "verified": verified,
            "excluded": excluded,
        },
    }
