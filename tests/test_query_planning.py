import pytest

from engine import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    LLMConfig,
    LLMRequestError,
    QueryPlanner,
)


class FailingLLM:
    async def translate_query(self, query):
        raise LLMRequestError("glm", "authentication", "unauthorized")

    async def expand_query(self, query):
        raise LLMRequestError("glm", "authentication", "unauthorized")


class UsefulLLM:
    async def translate_query(self, query):
        return "silicon photonic modulator"

    async def expand_query(self, query):
        return [
            "1. silicon photonics",
            "[AI 调用失败：timeout]",
            "optical modulator",
        ]


def test_deepseek_is_the_default_provider_consumed_by_new_sessions():
    assert LLMConfig.default() == LLMConfig("deepseek", "deepseek-v4-flash", "")
    assert (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL) == (
        "deepseek",
        "deepseek-v4-flash",
    )


@pytest.mark.asyncio
async def test_llm_failure_keeps_original_query_instead_of_searching_error_text():
    plan = await QueryPlanner(lambda: FailingLLM()).build("硅光调制器")
    assert plan.variants == ("硅光调制器",)
    assert plan.ai_error == "authentication"


@pytest.mark.asyncio
async def test_chinese_plan_keeps_original_translation_and_clean_expansion():
    plan = await QueryPlanner(lambda: UsefulLLM()).build("硅光调制器")
    assert plan.variants == (
        "硅光调制器",
        "silicon photonic modulator",
        "optical modulator",
    )
    assert plan.translation_used is True
