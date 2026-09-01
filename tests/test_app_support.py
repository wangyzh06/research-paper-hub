from app_support import (
    api_key_fingerprint,
    build_llm_config,
    format_search_diagnostics,
    search_cache_ttl,
)
from engine import LLMConfig, QueryPlan, SearchResponse, SourceBatch, SourceError


def _response(*sources: SourceBatch) -> SearchResponse:
    return SearchResponse((), QueryPlan("test", ("test",)), sources)


def test_api_key_fingerprint_is_stable_and_does_not_expose_secret():
    secret = "super-secret-key"

    fingerprint = api_key_fingerprint(secret)

    assert fingerprint == api_key_fingerprint(secret)
    assert len(fingerprint) == 12
    assert secret not in fingerprint


def test_blank_ui_configuration_uses_glm_defaults():
    assert build_llm_config("", "", "") == LLMConfig(
        "glm", "glm-4.7-flash", ""
    )


def test_partial_search_diagnostics_name_failed_source():
    response = _response(
        SourceBatch(
            "openalex",
            (),
            0,
            SourceError("timeout", None, True, "timed out"),
        ),
        SourceBatch("arxiv", ({"id": "a"},), 1),
    )

    summary = format_search_diagnostics(response)

    assert summary["partial"] is True
    assert summary["failed_sources"] == ["openalex"]
    assert summary["source_counts"] == {"openalex": 0, "arxiv": 1}
    assert summary["errors"]["openalex"]["category"] == "timeout"


def test_authentication_failures_are_never_cached():
    response = _response(
        SourceBatch(
            "semantic_scholar",
            (),
            0,
            SourceError("authentication", 401, False, "bad key"),
        )
    )

    assert search_cache_ttl(response) == 0


def test_retryable_partial_results_use_short_cache():
    response = _response(
        SourceBatch(
            "openalex",
            (),
            0,
            SourceError("rate_limit", 429, True, "slow down"),
        ),
        SourceBatch("pubmed", ({"id": "1"},), 1),
    )

    assert search_cache_ttl(response) == 30


def test_successful_search_uses_normal_cache_ttl():
    response = _response(SourceBatch("openalex", ({"id": "1"},), 1))

    assert search_cache_ttl(response) == 300
