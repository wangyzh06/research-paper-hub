import pytest

from engine import (
    LLMConfig,
    QueryPlan,
    QueryPlanner,
    SearchEngine,
    SourceBatch,
    SourceError,
)


class StaticPlanner:
    def __init__(self, variants=("query", "silicon photonic modulator")):
        self.variants = variants

    async def build(self, query):
        return QueryPlan(query, self.variants, len(self.variants) > 1, None, "topic")


class KindLLM:
    async def translate_query(self, query):
        return query

    async def expand_query(self, query):
        return []


class FakeSource:
    def __init__(self, source, pages=None, error=None):
        self.source = source
        self.pages = pages or {}
        self.error = error
        self.calls = []
        self.closed = False

    async def search(self, query, limit=20, page=1, offset=0):
        cursor = offset if self.source == "semantic_scholar" else page
        self.calls.append((query, cursor))
        if self.error:
            return SourceBatch(self.source, (), 0, self.error)
        papers = tuple(self.pages.get(cursor, ()))
        total = sum(len(values) for values in self.pages.values())
        return SourceBatch(self.source, papers, total, None)

    async def aclose(self):
        self.closed = True


def paper(source, ident, title, doi="", abstract=""):
    return {
        "id": ident,
        "source": source,
        "title": title,
        "abstract": abstract,
        "year": 2025,
        "citation_count": 0,
        "venue": "",
        "authors": [],
        "doi": doi,
        "url": "",
    }


@pytest.mark.asyncio
async def test_full_openalex_pool_does_not_starve_other_sources():
    clients = {
        "openalex": FakeSource(
            "openalex",
            {1: [paper("openalex", f"oa{i}", f"OA {i}") for i in range(20)]},
        ),
        "arxiv": FakeSource("arxiv", {1: [paper("arxiv", "ax", "arXiv match")]}),
        "crossref": FakeSource(
            "crossref", {1: [paper("crossref", "cr", "Crossref match")]}
        ),
        "semantic_scholar": FakeSource(
            "semantic_scholar",
            {0: [paper("semantic_scholar", "ss", "S2 match")]},
        ),
        "pubmed": FakeSource("pubmed", {1: [paper("pubmed", "pm", "PubMed match")]}),
    }
    engine = SearchEngine(
        LLMConfig.default(), clients=clients, query_planner=StaticPlanner(("query",))
    )
    response = await engine.search_with_diagnostics("query", limit=5)
    assert {p["source"] for p in response.papers} == set(clients)
    assert all(client.closed for client in clients.values())


@pytest.mark.asyncio
async def test_dedup_merges_doi_metadata_before_limit_is_applied():
    richer = paper(
        "crossref", "cr", "Exact paper", "10.1/x", abstract="rich abstract"
    )
    duplicate = paper(
        "openalex", "oa", "Exact paper", "https://doi.org/10.1/X"
    )
    clients = {
        "openalex": FakeSource("openalex", {1: [duplicate]}),
        "crossref": FakeSource("crossref", {1: [richer]}),
    }
    engine = SearchEngine(
        LLMConfig.default(), clients=clients, query_planner=StaticPlanner(("Exact paper",))
    )
    response = await engine.search_with_diagnostics("Exact paper", limit=1)
    assert len(response.papers) == 1
    assert response.papers[0]["abstract"] == "rich abstract"
    assert set(response.papers[0]["sources"]) == {"openalex", "crossref"}


@pytest.mark.asyncio
async def test_chinese_query_uses_english_variant_for_relevance_ranking():
    clients = {
        "openalex": FakeSource(
            "openalex",
            {
                1: [
                    paper("openalex", "bad", "Unrelated first"),
                    paper("openalex", "good", "Silicon photonic modulator"),
                ]
            },
        )
    }
    engine = SearchEngine(
        LLMConfig.default(), clients=clients, query_planner=StaticPlanner()
    )
    response = await engine.search_with_diagnostics("硅光调制器", limit=2)
    assert response.papers[0]["id"] == "good"


@pytest.mark.asyncio
async def test_provider_pagination_requests_a_new_page():
    source = FakeSource(
        "openalex",
        {
            1: [paper("openalex", "p1", "Paper 1")],
            2: [paper("openalex", "p2", "Paper 2")],
        },
    )
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": source},
        query_planner=StaticPlanner(("paper",)),
    )
    response = await engine.search_with_diagnostics("paper", limit=2)
    assert [p["id"] for p in response.papers] == ["p1", "p2"]
    assert source.calls == [("paper", 1), ("paper", 2)]


@pytest.mark.asyncio
async def test_partial_and_total_failures_remain_distinguishable():
    timeout = SourceError("timeout", None, True, "slow")
    partial_engine = SearchEngine(
        LLMConfig.default(),
        clients={
            "openalex": FakeSource("openalex", error=timeout),
            "arxiv": FakeSource("arxiv", {1: [paper("arxiv", "a", "Paper")]}),
        },
        query_planner=StaticPlanner(("paper",)),
    )
    total_engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": FakeSource("openalex", error=timeout)},
        query_planner=StaticPlanner(("paper",)),
    )
    partial = await partial_engine.search_with_diagnostics("paper", limit=2)
    total = await total_engine.search_with_diagnostics("paper", limit=2)
    assert partial.partial is True and partial.all_failed is False
    assert total.partial is False and total.all_failed is True


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("10.1000/xyz", "doi"),
        ("arXiv:2401.00001", "arxiv"),
        ("PMID:12345", "pmid"),
        ("W12345", "openalex"),
        ("this is a complete paper title with more than eight separate words", "title"),
    ],
)
@pytest.mark.asyncio
async def test_query_planner_labels_identifier_and_title_queries(query, kind):
    plan = await QueryPlanner(lambda: KindLLM()).build(query)
    assert plan.kind == kind
    assert plan.original in plan.variants
