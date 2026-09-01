import httpx
import pytest

from engine import (
    ArxivClient,
    CrossrefClient,
    OpenAlexClient,
    PubMedClient,
    QueryPlan,
    SearchResponse,
    SemanticScholarClient,
    SourceBatch,
    SourceError,
    _flatten_abstract,
    normalize_doi,
    normalize_pubmed_summary,
    normalize_semantic_paper,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)


class SequenceHttpClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self):
        return None


class LegacyStatusError(Exception):
    def __init__(self, status_code):
        self.response = type("Response", (), {"status_code": status_code})()
        super().__init__(f"HTTP {status_code}")


def test_openalex_abstract_reconstruction_keeps_repeated_tokens():
    assert _flatten_abstract({"the": [0, 2], "cat": [1]}) == "the cat the"


def test_doi_url_and_bare_doi_normalize_to_the_same_identity():
    assert normalize_doi("https://doi.org/10.1000/XYZ") == "10.1000/xyz"
    assert normalize_doi("doi: 10.1000/xyz") == "10.1000/xyz"


def test_semantic_scholar_response_maps_external_ids_and_publication_venue():
    paper = normalize_semantic_paper(
        {
            "paperId": "s2",
            "title": "Paper",
            "authors": [{"name": "A"}],
            "citationCount": 9,
            "externalIds": {"DOI": "10.1/X"},
            "publicationVenue": {"name": "Nature"},
        }
    )
    assert paper["source"] == "semantic_scholar"
    assert paper["semantic_id"] == "s2"
    assert paper["doi"] == "10.1/x"
    assert paper["venue"] == "Nature"
    assert paper["citation_count"] == 9


def test_pubmed_summary_keeps_pmid_separate_from_doi_and_abstract():
    paper = normalize_pubmed_summary(
        "12345",
        {
            "title": "Clinical result",
            "authors": [{"name": "A Author"}],
            "pubdate": "2024 Jan",
            "source": "Journal",
            "elocationid": "doi: 10.2/ABC",
        },
    )
    assert paper["id"] == "pmid:12345"
    assert paper["pmid"] == "12345"
    assert paper["doi"] == "10.2/abc"
    assert paper["abstract"] == ""
    assert paper["year"] == 2024


def test_source_failure_is_not_an_empty_success():
    batch = SourceBatch(
        "openalex",
        (),
        0,
        SourceError("timeout", None, True, "timed out"),
    )
    assert batch.ok is False
    assert batch.error.category == "timeout"


@pytest.mark.asyncio
async def test_openalex_search_returns_canonical_source_batch():
    client = OpenAlexClient.__new__(OpenAlexClient)
    client.client = SequenceHttpClient(
        [
            FakeResponse(
                {
                    "results": [
                        {
                            "id": "https://openalex.org/W1",
                            "title": "OpenAlex paper",
                            "abstract_inverted_index": {"paper": [1], "A": [0]},
                            "publication_year": 2024,
                            "cited_by_count": 3,
                            "authorships": [],
                            "doi": "https://doi.org/10.3/X",
                        }
                    ],
                    "meta": {"count": 1},
                }
            )
        ]
    )
    batch = await client.search("paper", limit=10)
    assert isinstance(batch, SourceBatch)
    assert batch.source == "openalex"
    assert batch.papers[0]["doi"] == "10.3/x"
    assert batch.papers[0]["openalex_id"] == "W1"


@pytest.mark.asyncio
async def test_source_timeout_is_reported_as_retryable_error():
    client = OpenAlexClient.__new__(OpenAlexClient)
    client.client = SequenceHttpClient([httpx.ReadTimeout("slow")])
    batch = await client.search("paper")
    assert batch.papers == ()
    assert batch.error.category == "timeout"
    assert batch.error.retryable is True


@pytest.mark.asyncio
async def test_transient_network_reset_is_reported_as_retryable_error():
    client = OpenAlexClient.__new__(OpenAlexClient)
    client.client = SequenceHttpClient([httpx.NetworkError("connection reset")])

    batch = await client.search("paper")

    assert batch.error.category == "network"
    assert batch.error.retryable is True


@pytest.mark.asyncio
async def test_semantic_scholar_search_normalizes_papers_in_batch():
    client = SemanticScholarClient.__new__(SemanticScholarClient)
    client.client = SequenceHttpClient(
        [FakeResponse({"data": [{"paperId": "s2", "title": "Paper"}], "total": 1})]
    )
    batch = await client.search("paper")
    assert batch.source == "semantic_scholar"
    assert batch.papers[0]["semantic_id"] == "s2"
    assert client.client.calls[0][0][0] == "/graph/v1/paper/search"


@pytest.mark.asyncio
async def test_legacy_http_status_errors_are_still_classified():
    client = OpenAlexClient.__new__(OpenAlexClient)
    client.client = SequenceHttpClient([LegacyStatusError(429)])

    batch = await client.search("paper")

    assert batch.error.category == "rate_limit"
    assert batch.error.status_code == 429
    assert batch.error.retryable is True


@pytest.mark.asyncio
async def test_pubmed_search_returns_normalized_summary_batch():
    client = PubMedClient.__new__(PubMedClient)
    client.client = SequenceHttpClient(
        [
            FakeResponse({"esearchresult": {"idlist": ["123"], "count": "25"}}),
            FakeResponse({"result": {"123": {"title": "Paper", "pubdate": "2024"}}}),
            FakeResponse(
                {},
                text=(
                    "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
                    "<PMID>123</PMID><Article><Abstract>"
                    "<AbstractText Label='RESULTS'>Treatment improved outcomes.</AbstractText>"
                    "</Abstract></Article></MedlineCitation></PubmedArticle>"
                    "</PubmedArticleSet>"
                ),
            ),
        ]
    )
    batch = await client.search("paper")
    assert batch.source == "pubmed"
    assert batch.papers[0]["id"] == "pmid:123"
    assert batch.papers[0]["abstract"] == "RESULTS: Treatment improved outcomes."
    assert batch.total == 25
    assert [call[0][0] for call in client.client.calls] == [
        "/entrez/eutils/esearch.fcgi",
        "/entrez/eutils/esummary.fcgi",
        "/entrez/eutils/efetch.fcgi",
    ]


@pytest.mark.asyncio
async def test_crossref_prefers_publication_year_over_record_creation_year():
    client = CrossrefClient.__new__(CrossrefClient)
    client.client = SequenceHttpClient(
        [
            FakeResponse(
                {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.4/X",
                                "title": ["Paper"],
                                "published": {"date-parts": [[2020, 1, 1]]},
                                "created": {"date-parts": [[2024, 1, 1]]},
                            }
                        ],
                        "total-results": 1,
                    }
                }
            )
        ]
    )
    batch = await client.search("paper")
    assert isinstance(batch, SourceBatch)
    assert batch.papers[0]["year"] == 2020
    assert batch.papers[0]["doi"] == "10.4/x"


@pytest.mark.asyncio
async def test_arxiv_search_returns_source_batch_with_arxiv_id():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>42</opensearch:totalResults>
      <entry>
        <id>https://arxiv.org/abs/2401.00001</id>
        <published>2024-01-01T00:00:00Z</published>
        <title>Paper</title>
        <summary>Abstract</summary>
        <author><name>A Author</name></author>
      </entry>
    </feed>"""
    client = ArxivClient.__new__(ArxivClient)
    client.client = SequenceHttpClient([FakeResponse({}, text=xml)])
    batch = await client.search("paper")
    assert isinstance(batch, SourceBatch)
    assert batch.papers[0]["arxiv_id"] == "2401.00001"
    assert batch.total == 42
    assert client.client.calls[0][0][0] == "/api/query"


def test_search_response_distinguishes_partial_from_all_failed():
    failed = SourceBatch(
        "openalex", (), 0, SourceError("timeout", None, True, "slow")
    )
    succeeded = SourceBatch("arxiv", (), 0, None)
    partial = SearchResponse((), QueryPlan("q", ("q",)), (failed, succeeded))
    all_failed = SearchResponse((), QueryPlan("q", ("q",)), (failed,))
    assert partial.partial is True
    assert partial.all_failed is False
    assert all_failed.partial is False
    assert all_failed.all_failed is True
