import asyncio
import time

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


class TranslationLLM:
    async def translate_query(self, query):
        return "silicon photonic modulator"

    async def expand_query(self, query):
        return ["integrated optical modulator"]


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


class PartialPageSource:
    async def search(self, query, limit=20, page=1, offset=0):
        if page == 1:
            return SourceBatch(
                "openalex",
                (paper("openalex", "first", "First page result"),),
                2,
            )
        return SourceBatch(
            "openalex",
            (),
            2,
            SourceError("timeout", None, True, "second page timeout"),
        )

    async def aclose(self):
        pass


class ConcurrencyTrackingSource:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.call_times = []

    async def search(self, query, limit=20, page=1, offset=0):
        self.call_times.append(time.monotonic())
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return SourceBatch(
            "openalex",
            (paper("openalex", query, f"Result for {query}"),),
            1,
        )

    async def aclose(self):
        pass


class BudgetTrackingSource:
    def __init__(self):
        self.calls = []

    async def search(self, query, limit=20, page=1, offset=0):
        self.calls.append((query, limit, page))
        papers = tuple(
            paper("openalex", f"{query}-{page}-{index}", f"{query} result {index}")
            for index in range(limit)
        )
        return SourceBatch("openalex", papers, 1000)

    async def aclose(self):
        pass


class CrossrefBudgetSource:
    def __init__(self):
        self.calls = []

    async def search(self, query, limit=20, page=1, offset=0):
        self.calls.append((query, limit, page))
        papers = tuple(
            paper("crossref", f"{query}-{index}", f"{query} result {index}")
            for index in range(limit)
        )
        return SourceBatch("crossref", papers, 1000)

    async def aclose(self):
        pass


class RateLimitOnceSource:
    def __init__(self):
        self.attempts = 0

    async def search(self, query, limit=20, page=1, offset=0):
        self.attempts += 1
        if self.attempts == 1:
            return SourceBatch(
                "openalex",
                (),
                0,
                SourceError("rate_limit", 429, True, "slow down"),
            )
        return SourceBatch(
            "openalex",
            (paper("openalex", "recovered", "Recovered result"),),
            1,
        )

    async def aclose(self):
        pass


class AlwaysRateLimitedSource:
    def __init__(self):
        self.attempts = 0

    async def search(self, query, limit=20, page=1, offset=0):
        self.attempts += 1
        return SourceBatch(
            "openalex",
            (),
            0,
            SourceError("rate_limit", 429, True, "slow down"),
        )

    async def aclose(self):
        pass


class AlwaysNetworkFailingSource:
    def __init__(self):
        self.attempts = 0

    async def search(self, query, limit=20, page=1, offset=0):
        self.attempts += 1
        return SourceBatch(
            "openalex",
            (),
            0,
            SourceError("network", None, True, "connection reset"),
        )

    async def aclose(self):
        pass


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
async def test_second_page_failure_keeps_first_page_results_as_partial():
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": PartialPageSource()},
        query_planner=StaticPlanner(("paper",)),
    )

    response = await engine.search_with_diagnostics("paper", limit=2)

    assert [result["id"] for result in response.papers] == ["first"]
    assert response.partial is True
    assert response.all_failed is False


@pytest.mark.asyncio
async def test_query_variants_are_serialized_within_each_source():
    source = ConcurrencyTrackingSource()
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": source},
        query_planner=StaticPlanner(("q1", "q2", "q3", "q4")),
    )

    await engine.search_with_diagnostics("query", limit=200)

    assert source.max_active == 1


@pytest.mark.asyncio
async def test_query_variants_share_one_source_candidate_budget():
    source = BudgetTrackingSource()
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": source},
        query_planner=StaticPlanner(("q1", "q2", "q3", "q4")),
    )

    await engine.search_with_diagnostics("query", limit=200)

    assert len(source.calls) == 4
    assert sum(limit for _query, limit, _page in source.calls) == 200


@pytest.mark.asyncio
async def test_retryable_rate_limit_is_retried_once_before_reporting_failure():
    source = RateLimitOnceSource()
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": source},
        query_planner=StaticPlanner(("query",)),
        retry_delays=(0,),
    )

    response = await engine.search_with_diagnostics("query", limit=5)

    assert source.attempts == 2
    assert [result["id"] for result in response.papers] == ["recovered"]
    assert response.partial is False


@pytest.mark.asyncio
async def test_persistent_rate_limit_stops_remaining_variants_for_that_source():
    source = AlwaysRateLimitedSource()
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": source},
        query_planner=StaticPlanner(("q1", "q2", "q3", "q4")),
        retry_delays=(0,),
    )

    response = await engine.search_with_diagnostics("query", limit=200)

    assert source.attempts == 2
    assert response.all_failed is True
    assert response.sources[0].error.category == "rate_limit"


@pytest.mark.asyncio
async def test_persistent_network_failure_stops_remaining_source_variants():
    source = AlwaysNetworkFailingSource()
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": source},
        query_planner=StaticPlanner(("q1", "q2", "q3", "q4")),
        retry_delays=(0,),
    )

    response = await engine.search_with_diagnostics("query", limit=200)

    assert source.attempts == 2
    assert response.all_failed is True
    assert response.sources[0].error.category == "network"


@pytest.mark.asyncio
async def test_configured_source_interval_spaces_provider_requests():
    source = ConcurrencyTrackingSource()
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"openalex": source},
        query_planner=StaticPlanner(("q1", "q2", "q3")),
        retry_delays=(),
        source_intervals={"openalex": 0.03},
    )

    await engine.search_with_diagnostics("query", limit=200)

    gaps = [
        later - earlier
        for earlier, later in zip(source.call_times, source.call_times[1:])
    ]
    assert len(gaps) == 2
    assert all(gap >= 0.025 for gap in gaps)


@pytest.mark.asyncio
async def test_crossref_uses_small_responses_without_extra_page_per_variant():
    source = CrossrefBudgetSource()
    engine = SearchEngine(
        LLMConfig.default(),
        clients={"crossref": source},
        query_planner=StaticPlanner(("q1", "q2", "q3", "q4")),
        retry_delays=(),
        source_intervals={},
    )

    await engine.search_with_diagnostics("query", limit=200)

    assert len(source.calls) == 4
    assert {limit for _query, limit, _page in source.calls} == {20}


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


@pytest.mark.asyncio
async def test_offline_multisource_chinese_search_deduplicates_and_reports_partial():
    shared_openalex = paper(
        "openalex", "oa-shared", "Silicon photonic modulator", "10.1000/shared"
    )
    shared_crossref = paper(
        "crossref",
        "cr-shared",
        "Silicon photonic modulator",
        "https://doi.org/10.1000/SHARED",
        abstract="A richer abstract",
    )
    timeout = SourceError("timeout", None, True, "temporary timeout")
    clients = {
        "openalex": FakeSource("openalex", {1: [shared_openalex]}),
        "crossref": FakeSource("crossref", {1: [shared_crossref]}),
        "arxiv": FakeSource(
            "arxiv", {1: [paper("arxiv", "ax", "Integrated optical modulator")]}
        ),
        "pubmed": FakeSource(
            "pubmed", {1: [paper("pubmed", "pm", "Silicon device trial")]}
        ),
        "semantic_scholar": FakeSource("semantic_scholar", error=timeout),
    }
    engine = SearchEngine(
        LLMConfig.default(),
        clients=clients,
        query_planner=QueryPlanner(lambda: TranslationLLM()),
    )

    response = await engine.search_with_diagnostics("硅光调制器", limit=10)

    assert response.partial is True
    assert response.query_plan.translation_used is True
    shared = next(p for p in response.papers if p["doi"] == "10.1000/shared")
    assert shared["abstract"] == "A richer abstract"
    assert set(shared["sources"]) == {"openalex", "crossref"}
    assert {batch.source for batch in response.sources if not batch.ok} == {
        "semantic_scholar"
    }
