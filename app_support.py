"""Pure helpers shared by the Streamlit interface and its tests."""

from __future__ import annotations

import hashlib

from engine import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    LLMConfig,
    LLM_PROVIDERS,
    SearchResponse,
)
from consensus import ConsensusAnalysis


def api_key_fingerprint(api_key: str) -> str:
    """Return a non-reversible cache identity without retaining the API key."""
    if not api_key:
        return "no-key"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def build_llm_config(provider: str, model: str, api_key: str) -> LLMConfig:
    """Normalize UI values and use GLM when no provider is configured."""
    selected_provider = provider if provider in LLM_PROVIDERS else DEFAULT_LLM_PROVIDER
    provider_info = LLM_PROVIDERS[selected_provider]
    selected_model = (model or "").strip() or provider_info["default_model"]
    if selected_provider == DEFAULT_LLM_PROVIDER and not selected_model:
        selected_model = DEFAULT_LLM_MODEL
    return LLMConfig(selected_provider, selected_model, (api_key or "").strip())


def llm_secret_name(provider: str) -> str:
    """Return the documented environment/Streamlit secret name for a provider."""
    selected_provider = provider if provider in LLM_PROVIDERS else DEFAULT_LLM_PROVIDER
    return LLM_PROVIDERS[selected_provider].get("env_key", "")


def format_search_diagnostics(response: SearchResponse) -> dict:
    """Create a UI-safe summary of source health and query planning."""
    failed_sources = [batch.source for batch in response.sources if not batch.ok]
    return {
        "partial": response.partial,
        "all_failed": response.all_failed,
        "failed_sources": failed_sources,
        "source_counts": {
            batch.source: len(batch.papers) for batch in response.sources
        },
        "errors": {
            batch.source: {
                "category": batch.error.category,
                "status_code": batch.error.status_code,
                "retryable": batch.error.retryable,
                "detail": batch.error.detail,
            }
            for batch in response.sources
            if batch.error is not None
        },
        "translation_used": response.query_plan.translation_used,
        "ai_error": response.query_plan.ai_error,
        "query_variants": list(response.query_plan.variants),
    }


def search_cache_ttl(response: SearchResponse) -> int:
    """Choose a TTL that never makes authentication or total failures sticky."""
    errors = [batch.error for batch in response.sources if batch.error is not None]
    if response.all_failed or any(error.category == "authentication" for error in errors):
        return 0
    if response.partial:
        return 30 if any(error.retryable for error in errors) else 60
    return 300


def consensus_rows(analysis: ConsensusAnalysis) -> list[dict]:
    """Return all evidence, including exclusions, in a stable UI table shape."""
    stance_labels = {
        "support": "支持",
        "oppose": "反对",
        "mixed": "混合",
        "uncertain": "不确定",
        "not_applicable": "不适用",
    }
    return [
        {
            "论文": item.title,
            "纳入": "是" if item.eligible else "否",
            "立场": stance_labels.get(item.stance, "不确定"),
            "研究类型": item.study_type,
            "质量": item.quality,
            "抽取置信度": item.confidence,
            "核心证据": item.claim if item.eligible else item.exclusion_reason,
            "限制": "、".join(item.limitations),
        }
        for item in analysis.evidence
    ]
