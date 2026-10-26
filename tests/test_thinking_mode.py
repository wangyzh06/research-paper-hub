"""LLM 推理模式（thinking）性能回归测试。

背景（实测 2026-10）：
  DeepSeek v4 系列是**推理型模型**，即使面对"读摘要 → 填 JSON"这类结构化
  分类任务，也会先生成大量隐藏 reasoning token 再输出。
  实测同一抽取任务：
      thinking=ON : 3.50s/次, 651 completion tokens, 1729 字符隐藏推理
      thinking=OFF: 1.49s/次, 241 completion tokens, 0 字符隐藏推理
  整条 analyze() 流水线（50 篇，抽取量完全不变）：50.5s → 27.4s（1.84x）。

**关键**：这不是"减少抽取量"，证据条数、纳入条数都保持不变，
因此不会影响统计结果——这正是用户明确要求避免的副作用。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import engine as engine_module
from engine import LLMClient


def test_client_accepts_thinking_flag():
    """LLMClient 必须支持显式的推理开关。"""
    on = LLMClient(provider="deepseek", api_key="k", model="m", thinking=True)
    off = LLMClient(provider="deepseek", api_key="k", model="m", thinking=False)

    assert on.thinking is True
    assert off.thinking is False


def test_thinking_defaults_to_enabled():
    """默认保持原行为（向后兼容）。"""
    client = LLMClient(provider="deepseek", api_key="k", model="m")
    assert client.thinking is True


def test_complete_strict_accepts_thinking_override():
    """complete_strict 必须支持按调用覆盖（抽取关、综述开）。"""
    import inspect

    sig = inspect.signature(LLMClient.complete_strict)
    assert "thinking" in sig.parameters
    assert sig.parameters["thinking"].default is None


def test_evidence_extraction_disables_thinking():
    """证据抽取（含批量与逐篇）必须关闭隐藏推理。"""
    source = Path(__file__).resolve().parents[1].joinpath("consensus.py").read_text(encoding="utf-8")

    # 统计 complete_strict(..., thinking=False) 的调用次数
    count = source.count("thinking=False")
    assert count >= 2, (
        f"证据抽取路径应显式关闭推理（找到 {count} 处，期望 >= 2：批量 + 逐篇）"
    )


def test_extraction_calls_still_request_all_fields():
    """关闭推理不得削减抽取字段——统计口径必须保持完整。"""
    source = Path(__file__).resolve().parents[1].joinpath("consensus.py").read_text(encoding="utf-8")

    required = [
        "eligible", "study_type", "population", "intervention", "comparator",
        "outcome", "stance", "claim", "effect_direction", "sample_size",
        "statistical_evidence", "quality", "directness", "confidence",
        "excerpt", "limitations", "cohort_key",
    ]
    for field_name in required:
        assert field_name in source, f"抽取 schema 缺少字段 {field_name}"


def test_thinking_payload_only_sent_when_disabled(monkeypatch):
    """thinking=disabled 时才发送 thinking 参数；开启时不干预请求体。"""
    sent = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"ok":1}'}, "finish_reason": "stop"}]}

    class FakeHTTP:
        async def post(self, url, json=None, **kw):
            sent.append(json or {})
            return FakeResponse()

    client = LLMClient(provider="deepseek", api_key="k", model="m")
    client._client = FakeHTTP()

    import asyncio

    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        client.complete_strict([{"role": "user", "content": "x"}], thinking=False)
    )
    assert sent[-1].get("thinking") == {"type": "disabled"}

    client2 = LLMClient(provider="deepseek", api_key="k", model="m")
    client2._client = FakeHTTP()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        client2.complete_strict([{"role": "user", "content": "x"}])
    )
    assert "thinking" not in sent[-1], "开启推理时不应发送 thinking 参数"
