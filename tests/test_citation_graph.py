"""引文网络扩展（Citation Graph / Snowballing）测试。

对标 Consensus Deep 的第一步：「Discover additional papers by following
citation connections between the initial results」。

成本动机（实测）：OpenAlex 关键词检索 10 积分/次（每日 1000 积分），
而引文图查询仅 1 积分/次、Europe PMC 引文接口完全免费。
"""
from __future__ import annotations

import pytest

from engine import (
    EuropePMCClient,
    SearchEngine,
    SourceBatch,
    SourceError,
    _normalize_openalex_work,
)


def _paper(title, *, doi="", pmid="", oaid="", abstract="a" * 120, cites=0, year=2020):
    p = {
        "title": title, "abstract": abstract, "year": year,
        "citation_count": cites, "doi": doi, "id": title,
        "source": "test", "authors": [],
    }
    if pmid:
        p["pmid"] = pmid
    if oaid:
        p["openalex_id"] = oaid
    return p


class FakeEPMC:
    """模拟 Europe PMC：免费引文接口 + 批量补全。"""

    def __init__(self, citations=None, references=None, hydrated=None):
        self._citations = citations or {}
        self._references = references or {}
        self._hydrated = hydrated or {}
        self.hydrate_calls = []

    async def get_citations(self, pmid, limit=50):
        return self._citations.get(pmid, [])

    async def get_references(self, pmid, limit=50):
        return self._references.get(pmid, [])

    async def hydrate(self, ids, limit=50):
        self.hydrate_calls.append(list(ids))
        papers = [self._hydrated[i] for i in ids if i in self._hydrated]
        return SourceBatch("europepmc", tuple(papers), len(papers), None)


class FakeS2:
    def __init__(self, neighbours=None):
        self._n = neighbours or {}
        self.calls = []

    async def get_citation_neighbors(self, paper_id, limit=50, direction="citations"):
        self.calls.append((paper_id, direction))
        return self._n.get((paper_id, direction), [])


class FakeOpenAlex:
    def __init__(self, citing=None, referenced=None):
        self._citing = citing or {}
        self._referenced = referenced or {}
        self.calls = []

    async def get_citing_works(self, wid, limit=200):
        self.calls.append(("citing", wid))
        return SourceBatch("openalex", tuple(self._citing.get(wid, [])), 0, None)

    async def get_referenced_works(self, wid, limit=200):
        self.calls.append(("referenced", wid))
        return SourceBatch("openalex", tuple(self._referenced.get(wid, [])), 0, None)


def _engine(clients):
    return SearchEngine(clients=clients)


# === Europe PMC 路径 ===
@pytest.mark.asyncio
async def test_expands_via_europepmc_citations_and_references():
    seed = _paper("Seed", pmid="111", doi="10.1/seed")
    neighbour = _paper("Neighbour", abstract="x" * 200)
    epmc = FakeEPMC(
        citations={"111": [{"id": "222"}]},
        references={"111": [{"id": "333"}]},
        hydrated={"222": neighbour, "333": _paper("Ref", abstract="y" * 200)},
    )
    engine = _engine({"europepmc": epmc})

    result = await engine._expand_citation_graph([seed], seed_count=1, max_neighbors=10)

    titles = {p["title"] for p in result}
    assert titles == {"Neighbour", "Ref"}
    # 批量补全应被调用（而非逐条查询）
    assert epmc.hydrate_calls, "未使用批量补全 hydration"


@pytest.mark.asyncio
async def test_seed_itself_is_not_returned_as_new_paper():
    seed = _paper("Seed", pmid="111", doi="10.1/seed")
    epmc = FakeEPMC(
        citations={"111": [{"id": "SELF"}]},
        hydrated={"SELF": dict(seed)},
    )
    engine = _engine({"europepmc": epmc})

    result = await engine._expand_citation_graph([seed], seed_count=1, max_neighbors=10)

    assert result == [], "种子论文被当作新增结果重复计入"


@pytest.mark.asyncio
async def test_papers_without_abstract_are_dropped():
    """无摘要的论文无法进入证据抽取，不应计入扩展结果。"""
    seed = _paper("Seed", pmid="111")
    epmc = FakeEPMC(
        citations={"111": [{"id": "222"}, {"id": "333"}]},
        hydrated={"222": _paper("NoAbs", abstract=""), "333": _paper("HasAbs", abstract="z" * 150)},
    )
    engine = _engine({"europepmc": epmc})

    result = await engine._expand_citation_graph([seed], seed_count=1, max_neighbors=10)

    assert [p["title"] for p in result] == ["HasAbs"]


@pytest.mark.asyncio
async def test_duplicate_neighbours_are_deduplicated():
    seed = _paper("Seed", pmid="111")
    shared = _paper("Shared", doi="10.1/shared", abstract="q" * 150)
    epmc = FakeEPMC(
        citations={"111": [{"id": "222"}]},
        references={"111": [{"id": "333"}]},
        hydrated={"222": shared, "333": dict(shared)},
    )
    engine = _engine({"europepmc": epmc})

    result = await engine._expand_citation_graph([seed], seed_count=1, max_neighbors=10)

    assert len(result) == 1, "同一篇论文（同 DOI）被重复计入"


# === S2 路径（DOI，覆盖无 pmid 的文献）===
@pytest.mark.asyncio
async def test_expands_via_s2_when_no_pmid():
    """本领域多数半导体器件文献没有 pmid，必须能走 DOI 路径。"""
    seed = _paper("Seed", doi="10.1/seed", pmid="")
    s2 = FakeS2({("DOI:10.1/seed", "citations"): [_paper("Cited", doi="10.1/c", abstract="c" * 150)]})
    engine = _engine({"semantic_scholar": s2})

    result = await engine._expand_citation_graph([seed], seed_count=1, max_neighbors=10)

    assert [p["title"] for p in result] == ["Cited"]
    assert ("DOI:10.1/seed", "citations") in s2.calls


# === OpenAlex 路径 ===
@pytest.mark.asyncio
async def test_expands_via_openalex_graph():
    seed = _paper("Seed", oaid="W123")
    oa = FakeOpenAlex(
        citing={"W123": [_paper("Forward", abstract="f" * 150)]},
        referenced={"W123": [_paper("Backward", abstract="b" * 150)]},
    )
    engine = _engine({"openalex": oa})

    result = await engine._expand_citation_graph([seed], seed_count=1, max_neighbors=10)

    assert {p["title"] for p in result} == {"Forward", "Backward"}


@pytest.mark.asyncio
async def test_source_failure_does_not_raise():
    """任一引文源异常都不得中断整体扩展。"""
    class Exploding:
        async def get_citations(self, *a, **k):
            raise RuntimeError("boom")

        async def get_references(self, *a, **k):
            raise RuntimeError("boom")

        async def hydrate(self, *a, **k):
            raise RuntimeError("boom")

    seed = _paper("Seed", pmid="111", doi="10.1/seed")
    engine = _engine({"europepmc": Exploding()})

    result = await engine._expand_citation_graph([seed], seed_count=1, max_neighbors=10)

    assert result == []


@pytest.mark.asyncio
async def test_no_seeds_returns_empty():
    engine = _engine({"europepmc": FakeEPMC()})

    assert await engine._expand_citation_graph([], seed_count=5) == []


@pytest.mark.asyncio
async def test_seed_without_any_identifier_is_skipped():
    """没有任何标识符（pmid/doi/openalex_id）的论文无法作为种子。"""
    engine = _engine({"europepmc": FakeEPMC()})

    assert await engine._expand_citation_graph([_paper("NoIds")], seed_count=5) == []


# === OpenAlex 规范化 ===
def test_normalize_openalex_work_extracts_fields():
    raw = {
        "id": "https://openalex.org/W999",
        "title": "A study",
        "doi": "https://doi.org/10.1/X",
        "publication_year": 2019,
        "cited_by_count": 42,
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "primary_location": {"source": {"display_name": "J. Test"}},
        "authorships": [{"author": {"display_name": "A. Author"}}],
    }

    paper = _normalize_openalex_work(raw)

    assert paper["openalex_id"] == "W999"
    assert paper["doi"] == "10.1/x"
    assert paper["year"] == 2019
    assert paper["citation_count"] == 42
    assert paper["abstract"] == "Hello world"
    assert paper["venue"] == "J. Test"
    assert paper["authors"] == [{"name": "A. Author"}]


def test_normalize_openalex_work_handles_missing_abstract():
    paper = _normalize_openalex_work({"id": "https://openalex.org/W1", "title": "T"})

    assert paper["abstract"] == ""
    assert paper["openalex_id"] == "W1"


# === Europe PMC 引文存根结构 ===
def test_europepmc_citation_stub_shape_is_documented():
    """Europe PMC 引文接口返回的是无摘要存根，因此必须二次补全。

    这条测试锁定了这个事实，防止有人误以为可以直接使用存根。
    """
    import inspect

    source = inspect.getsource(EuropePMCClient.get_citations)
    assert "citationList" in source
    # 存根字段不含摘要，代码注释与 hydrate() 的存在即为证据
    assert "hydrate" in inspect.getsource(EuropePMCClient)
