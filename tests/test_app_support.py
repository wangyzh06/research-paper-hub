from app_support import (
    api_key_fingerprint,
    build_llm_config,
    consensus_rows,
    format_search_diagnostics,
    llm_secret_name,
    search_cache_ttl,
)
from consensus import ConsensusAnalysis, ConsensusScore, EvidenceItem
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


def test_glm_uses_zhipu_secret_name():
    assert llm_secret_name("glm") == "ZHIPUAI_API_KEY"


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


def test_consensus_rows_keep_excluded_and_unavailable_evidence_visible():
    evidence = EvidenceItem(
        paper_id="p",
        title="Paper",
        eligible=False,
        exclusion_reason="no direct outcome",
        study_type="unknown",
        stance="not_applicable",
        claim="",
        excerpt="",
        quality="unknown",
        directness=0.0,
        confidence=0.0,
        independence_factor=1.0,
        limitations=("abstract_only",),
    )
    score = ConsensusScore.insufficient({"not_applicable": 1})

    rows = consensus_rows(
        ConsensusAnalysis((evidence,), score, "Insufficient evidence")
    )

    assert rows == [
        {
            "论文": "Paper",
            "纳入": "否",
            "立场": "不适用",
            "研究类型": "unknown",
            "质量": "unknown",
            "抽取置信度": 0.0,
            "核心证据": "no direct outcome",
            "限制": "abstract_only",
        }
    ]
