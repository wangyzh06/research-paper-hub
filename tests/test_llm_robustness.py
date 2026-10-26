"""LLM 客户端健壮性测试。

背景（实测）：推理型模型（deepseek-v4-flash）在大 prompt 下会把 max_tokens
预算全部耗在思考 token 上，返回 HTTP 200 + finish_reason="length" + content=""。
旧实现直接把这个空字符串当成功返回，导致上层 json.loads 失败并静默回退，
表现为「AI 精排/分析不生效但也不报错」。
"""
from __future__ import annotations

import json

import pytest

from engine import LLMClient, LLMRequestError, _llm_error_category


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _client_with(responses):
    """构造 LLMClient，并注入一个按序返回预设响应的假 httpx 客户端。"""
    client = LLMClient(provider="deepseek", api_key="k", model="m")
    queue = list(responses)
    sent = []

    class FakeHTTP:
        async def post(self, url, json=None, **kwargs):
            sent.append({"url": url, "json": json})
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    client._client = FakeHTTP()
    client._sent = sent
    return client


def _ok(content, finish="stop"):
    return FakeResponse({"choices": [{"message": {"content": content}, "finish_reason": finish}]})


@pytest.mark.asyncio
async def test_empty_content_is_retried_then_succeeds():
    """空内容必须重试，而不是当作有效结果返回。"""
    client = _client_with([
        _ok("", finish="length"),   # 推理占满预算
        _ok('{"ok":true}', finish="stop"),
    ])

    result = await client.complete_strict([{"role": "user", "content": "hi"}])

    assert result == '{"ok":true}'
    assert len(client._sent) == 2, "空响应未被重试"


@pytest.mark.asyncio
async def test_persistent_empty_content_raises():
    """持续空响应应抛出可诊断错误，而非返回空字符串。"""
    client = _client_with([_ok("", finish="length")] * 4)

    with pytest.raises(LLMRequestError) as excinfo:
        await client.complete_strict([{"role": "user", "content": "hi"}])

    assert "空内容" in str(excinfo.value)


@pytest.mark.asyncio
async def test_none_content_is_treated_as_empty():
    """content 为 null 同样不应被当作有效结果。"""
    client = _client_with([
        FakeResponse({"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}),
        _ok("recovered"),
    ])

    assert await client.complete_strict([{"role": "user", "content": "x"}]) == "recovered"


@pytest.mark.asyncio
async def test_max_tokens_is_raised_above_default():
    """max_tokens 必须足够大，否则推理会挤掉输出（实测 4096 必现空响应）。"""
    client = _client_with([_ok("ok")])

    await client.complete_strict([{"role": "user", "content": "x"}])

    payload = client._sent[0]["json"]
    assert payload["max_tokens"] >= 8192, (
        f"max_tokens={payload['max_tokens']} 过小，大 prompt 下会被推理耗尽"
    )


def test_llm_request_error_category_is_preserved():
    """已分类的 LLMRequestError 不应被重新归类为 request。

    否则 empty_response 等分类会丢失，重试逻辑失效。
    """
    error = LLMRequestError("deepseek", "empty_response", "empty")

    assert _llm_error_category(error) == "empty_response"


def test_unknown_error_still_classified():
    assert _llm_error_category(ValueError("bad")) == "response_parsing"
    assert _llm_error_category(ConnectionError("down")) == "network"
