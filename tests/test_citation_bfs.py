"""BFS 引文扩展 + 前向引文排序 + 跨源一致性信号测试。

设计依据（对比 findpapers —— 4 年迭代的成熟实现）：
  1. 引文网络是**指数爆炸**的（1 → 50 → 2500 → …），因此必须有两个独立上限：
     - max_papers_per_level：每层**入库**上限
     - max_expansion_per_level：每层**参与下一轮扩展**上限（更严格）
     我们此前只有"单轮浅扩展"，这是与成熟实现的主要差距。
  2. 前向引文必须按引用量降序取：当有数千篇引用者而只要 200 篇时，
     排序决定截断质量（保留最有影响力的，而非任意子集）。
  3. 跨源命中数（findpapers 的 found_in）是天然的文献重要性信号，
     我们此前采集了 sources 却没用于排序。
"""
from __future__ import annotations

import pytest

from engine import SearchEngine, SourceBatch


def _paper(title, *, abstract="a" * 150, cites=0, doi="", pmid="", oaid="",
           year=2020, source="test", sources=None):
    p = {"title": title, "abstract": abstract, "year": year,
         "citation_count": cites, "doi": doi, "id": title,
         "source": source, "authors": [],
         "sources": list(sources) if sources is not None else [source]}
    if pmid:
        p["pmid"] = pmid
    if oaid:
        p["openalex_id"] = oaid
    return p


class FakeS2:
    """按 DOI 返回预置引文邻居。"""

    def __init__(self, graph):
        self._graph = graph          # {doi: {"citations": [...], "references": [...]}}
        self.calls = []

    async def get_citation_neighbors(self, paper_id, limit=50, direction="citations"):
        self.calls.append((paper_id, direction))
        doi = paper_id.replace("DOI:", "")
        return list(self._graph.get(doi, {}).get(direction, []))


class FakeOpenAlex:
    def __init__(self, citing=None, referenced=None):
        self._citing = citing or {}
        self._referenced = referenced or {}
        self.calls = []

    async def get_citing_works(self, wid, limit=200, sort_by_citations=True):
        self.calls.append(("citing", wid, sort_by_citations))
        return SourceBatch("openalex", tuple(self._citing.get(wid, [])), 0, None)

    async def get_referenced_works(self, wid, limit=200):
        self.calls.append(("referenced", wid))
        return SourceBatch("openalex", tuple(self._referenced.get(wid, [])), 0, None)


def _engine(clients):
    return SearchEngine(clients=clients)


# ============ BFS 分层 ============
@pytest.mark.asyncio
async def test_bfs_depth_one_stops_after_first_level():
    """depth=1 只扩展一层：不应触达第二跳邻居。"""
    seed = _paper("Seed", doi="10.1/seed")
    level1 = _paper("L1", doi="10.1/l1")
    level2 = _paper("L2", doi="10.1/l2")
    s2 = FakeS2({
        "10.1/seed": {"citations": [level1]},
        "10.1/l1": {"citations": [level2]},
    })
    engine = _engine({"semantic_scholar": s2})

    result = await engine._expand_citation_graph(
        [seed], seed_count=1, max_neighbors=10, max_depth=1)

    titles = {p["title"] for p in result}
    assert titles == {"L1"}, "depth=1 不应返回第二跳结果"


@pytest.mark.asyncio
async def test_bfs_depth_two_reaches_second_hop():
    """depth=2 应能挖到两跳之外的文献（findpapers BFS 的核心价值）。"""
    seed = _paper("Seed", doi="10.1/seed")
    level1 = _paper("L1", doi="10.1/l1")
    level2 = _paper("L2", doi="10.1/l2")
    s2 = FakeS2({
        "10.1/seed": {"citations": [level1]},
        "10.1/l1": {"citations": [level2]},
    })
    engine = _engine({"semantic_scholar": s2})

    result = await engine._expand_citation_graph(
        [seed], seed_count=1, max_neighbors=10, max_depth=2)

    titles = {p["title"] for p in result}
    assert titles == {"L1", "L2"}, "depth=2 应能触及第二跳"


@pytest.mark.asyncio
async def test_expansion_cap_prevents_exponential_blowup():
    """扩展上限必须显著小于入库上限，否则引文网络指数爆炸。"""
    seed = _paper("Seed", doi="10.1/seed")
    # 第一层给 20 篇，每篇又各带 20 篇 —— 若不设限，第二层会达 400 篇
    level1 = [_paper(f"L1-{i}", doi=f"10.1/l1-{i}") for i in range(20)]
    graph = {"10.1/seed": {"citations": level1}}
    for i in range(20):
        graph[f"10.1/l1-{i}"] = {
            "citations": [_paper(f"L2-{i}-{j}", doi=f"10.1/l2-{i}-{j}") for j in range(20)]
        }
    s2 = FakeS2(graph)
    engine = _engine({"semantic_scholar": s2})

    result = await engine._expand_citation_graph(
        [seed], seed_count=1, max_neighbors=50,
        max_depth=2, max_papers_per_level=60, max_expansion_per_level=2,
    )

    # 第一层入库 20 + 第二层仅扩展 2 个父节点 × 20 = 40，总量远小于 400
    assert len(result) <= 60, f"扩展未受控：{len(result)} 篇（预期 <= 60）"


@pytest.mark.asyncio
async def test_caps_rank_by_citation_count():
    """上限截断应按引用量优先，保留最有影响力的文献。"""
    seed = _paper("Seed", doi="10.1/seed")
    low = _paper("LowCited", doi="10.1/low", cites=1)
    high = _paper("HighCited", doi="10.1/high", cites=9999)
    s2 = FakeS2({"10.1/seed": {"citations": [low, high]}})
    engine = _engine({"semantic_scholar": s2})

    result = await engine._expand_citation_graph(
        [seed], seed_count=1, max_neighbors=10,
        max_depth=1, max_papers_per_level=1)

    assert [p["title"] for p in result] == ["HighCited"]


@pytest.mark.asyncio
async def test_visited_set_prevents_revisiting_across_levels():
    """visited 必须跨层共享，避免循环与重复抓取。"""
    seed = _paper("Seed", doi="10.1/seed")
    a = _paper("A", doi="10.1/a")
    b = _paper("B", doi="10.1/b")
    s2 = FakeS2({
        "10.1/seed": {"citations": [a]},
        "10.1/a": {"citations": [b, seed]},   # 指回种子
        "10.1/b": {"citations": [a]},         # 指回 A
    })
    engine = _engine({"semantic_scholar": s2})

    result = await engine._expand_citation_graph(
        [seed], seed_count=1, max_neighbors=10, max_depth=3)

    titles = [p["title"] for p in result]
    assert len(titles) == len(set(titles)), "出现重复论文"
    assert "Seed" not in titles, "种子被重复计入"


@pytest.mark.asyncio
async def test_empty_frontier_stops_early():
    seed = _paper("Seed", doi="10.1/seed")
    s2 = FakeS2({"10.1/seed": {"citations": []}})
    engine = _engine({"semantic_scholar": s2})

    result = await engine._expand_citation_graph(
        [seed], seed_count=1, max_depth=5)

    assert result == []


@pytest.mark.asyncio
async def test_no_abstract_papers_are_excluded():
    """无摘要的文献无法进入证据抽取，不应计入扩展结果。"""
    seed = _paper("Seed", doi="10.1/seed")
    s2 = FakeS2({"10.1/seed": {"citations": [
        _paper("NoAbstract", doi="10.1/x", abstract=""),
        _paper("HasAbstract", doi="10.1/y"),
    ]}})
    engine = _engine({"semantic_scholar": s2})

    result = await engine._expand_citation_graph(
        [seed], seed_count=1, max_neighbors=10, max_depth=1)

    assert [p["title"] for p in result] == ["HasAbstract"]


@pytest.mark.asyncio
async def test_source_failure_does_not_raise():
    class Exploding:
        async def get_citation_neighbors(self, *a, **k):
            raise RuntimeError("boom")

    seed = _paper("Seed", doi="10.1/seed")
    engine = _engine({"semantic_scholar": Exploding()})

    result = await engine._expand_citation_graph(
        [seed], seed_count=1, max_depth=2)

    assert result == []


@pytest.mark.asyncio
async def test_depth_zero_returns_nothing():
    seed = _paper("Seed", doi="10.1/seed")
    s2 = FakeS2({"10.1/seed": {"citations": [_paper("X", doi="10.1/x")]}})
    engine = _engine({"semantic_scholar": s2})

    assert await engine._expand_citation_graph([seed], max_depth=0) == []


# ============ 前向引文排序 ============
@pytest.mark.asyncio
async def test_forward_citations_request_citation_sort():
    """前向引文必须按引用量降序，保证截断时保留最有影响力的引用者。"""
    oa = FakeOpenAlex()
    await oa.get_citing_works("W123", limit=10)

    assert oa.calls[0] == ("citing", "W123", True)


def test_citing_works_has_sort_parameter():
    import inspect
    from engine import OpenAlexClient
    sig = inspect.signature(OpenAlexClient.get_citing_works)
    assert "sort_by_citations" in sig.parameters
    assert sig.parameters["sort_by_citations"].default is True


# ============ 跨源一致性信号 ============
def test_multi_source_agreement_boosts_ranking():
    """多源命中的论文应比单源命中排名更高（findpapers 的 found_in 信号）。"""
    from engine import QueryPlan

    engine = SearchEngine(clients={})
    plan = QueryPlan("annealing HBT", ("annealing HBT",))

    single = _paper("SingleSource", cites=0, source="openalex", sources=["openalex"])
    multi = _paper("MultiSource", cites=0, source="openalex",
                   sources=["openalex", "europepmc", "semantic_scholar"])

    occurrences = [
        ("openalex", 0, 1, single),
        ("openalex", 0, 2, multi),
    ]
    ranked = engine._merge_and_rank(plan, occurrences, sort="relevance")

    assert ranked[0]["title"] == "MultiSource", "多源命中未获得加权"
    assert ranked[0]["_source_agreement"] == 2
    assert ranked[1]["_source_agreement"] == 0


def test_single_source_gets_zero_agreement():
    from engine import QueryPlan

    engine = SearchEngine(clients={})
    plan = QueryPlan("q", ("q",))
    paper = _paper("Only", source="openalex", sources=["openalex"])

    ranked = engine._merge_and_rank(plan, [("openalex", 0, 1, paper)], sort="relevance")

    assert ranked[0]["_source_agreement"] == 0


def test_agreement_signal_is_capped():
    """跨源数封顶，避免源多的论文被过度加权。"""
    from engine import QueryPlan

    engine = SearchEngine(clients={})
    plan = QueryPlan("q", ("q",))
    paper = _paper("ManySources", source="a",
                   sources=["a", "b", "c", "d", "e", "f", "g"])

    ranked = engine._merge_and_rank(plan, [("a", 0, 1, paper)], sort="relevance")

    assert ranked[0]["_source_agreement"] == 3, "跨源信号应封顶为 3"


def test_merge_preserves_incoming_sources():
    """首次入库时必须保留调用方已标注的来源，否则跨源信号被清零。"""
    from engine import QueryPlan

    engine = SearchEngine(clients={})
    plan = QueryPlan("q", ("q",))
    paper = _paper("P", source="openalex",
                   sources=["openalex", "europepmc"])

    ranked = engine._merge_and_rank(plan, [("openalex", 0, 1, paper)], sort="relevance")

    assert set(ranked[0]["sources"]) == {"openalex", "europepmc"}
