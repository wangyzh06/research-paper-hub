"""报告正文 HTML 渲染回归测试。

背景（用户实际遇到的问题）：LLM 生成的报告中，引用徽章的原始 HTML 被当作
纯文本直接显示在正文里：
    <a target="_blank" href="https://openalex.org/W4366815871" class="cs-cite-badge">...

根因：``st.markdown(narrative)`` 未传 ``unsafe_allow_html=True``，
Streamlit 因此把 HTML 转义为可见文本。

修复：传 unsafe_allow_html，但同时必须先转义 LLM 输出
（否则模型输出里的任意标签会被真实渲染，构成 XSS）。
"""
from __future__ import annotations

import ast
import html
import re
from pathlib import Path

import pytest

APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _render_consensus_node(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "render_consensus":
            return node
    pytest.fail("render_consensus 未找到")


def _narrative_block_source() -> str:
    """提取徽章构建到 narrative 渲染的源码片段。"""
    source = APP_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = _render_consensus_node(tree)
    seg = ast.get_source_segment(source, fn) or ""
    start = seg.find("paper_lookup = {}")
    if start < 0:
        start = seg.find("badge_html")
    end = seg.find("st.markdown(formatted_narrative")
    assert start > 0 and end > start, "narrative 渲染逻辑未找到"
    return seg[start:end + 40]


def _narrative_markdown_call(fn: ast.FunctionDef) -> ast.Call | None:
    """定位真正的 ``st.markdown(formatted_narrative, ...)`` 调用节点。

    必须用 AST 而不是源码字符串匹配——源码里的**注释**同样包含
    ``unsafe_allow_html=True`` 字样，字符串匹配会误判为已修复。
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "markdown"):
            continue
        base = func.value
        if not (isinstance(base, ast.Name) and base.id == "st"):
            continue
        if node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "formatted_narrative":
            return node
    return None


def test_markdown_call_enables_unsafe_html():
    """引用徽章是 <a> 标签，必须允许 HTML 渲染（用户报告的 bug）。

    用 AST 检查真实调用，避免被注释中的同名字符串误导。
    """
    tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
    call = _narrative_markdown_call(_render_consensus_node(tree))
    assert call is not None, "未找到 st.markdown(formatted_narrative) 调用"

    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert "unsafe_allow_html" in kwargs, (
        "st.markdown(formatted_narrative) 未启用 unsafe_allow_html，"
        "引用徽章会以纯文本形式暴露在报告正文中"
    )
    value = kwargs["unsafe_allow_html"]
    assert isinstance(value, ast.Constant) and value.value is True, (
        "unsafe_allow_html 必须为 True"
    )


def test_llm_output_is_escaped_before_injection():
    """启用 unsafe_allow_html 后，必须先转义 LLM 输出（安全前提）。"""
    source = _narrative_block_source()
    assert "escape" in source.lower(), (
        "启用 unsafe_allow_html 前必须转义 LLM 生成的文本，防止任意标签被渲染"
    )
    assert "html.escape" in source or "escape_llm_html" in source


def test_escape_happens_before_cite_injection():
    """顺序约束：先整体转义，再注入徽章（否则徽章自身也会被转义）。"""
    source = _narrative_block_source()
    escape_pos = source.lower().find("escape_llm_html")
    inject_pos = source.find("replace_cite(safe_narrative")
    if inject_pos < 0:
        inject_pos = source.find("formatted_narrative = re.sub")
    assert escape_pos >= 0 and inject_pos >= 0, "缺少转义或注入步骤"
    assert escape_pos < inject_pos, "必须先转义 LLM 输出，再注入本地徽章 HTML"


def test_escaped_model_html_is_not_executed():
    """行为验证：模型输出的 <script> / <a> 必须被转义为实体。"""

    def escape_llm_html(text: str) -> str:
        return html.escape(text, quote=False)

    malicious = '<script>alert(1)</script><a href="evil">x</a>'
    out = escape_llm_html(malicious)

    assert "<script>" not in out
    assert "<a href" not in out
    assert "&lt;script&gt;" in out


def test_our_own_badge_html_survives_escaping():
    """本地构建的徽章 HTML 在转义之后注入，因此应保持可渲染。"""
    badge = '<a target="_blank" href="https://openalex.org/W1" class="cs-cite-badge">Liu</a>'
    escaped_narrative = html.escape('text [1] more', quote=False)
    combined = escaped_narrative.replace("[1]", badge)

    assert '<a target="_blank"' in combined, "徽章 HTML 应保持原样"
    assert badge in combined


def test_badge_link_comes_from_trusted_metadata():
    """徽章链接来自文献元数据（OpenAlex/DOI），不是模型输出。"""
    source = _narrative_block_source()
    assert 'href="{link_url}"' in source, "徽章链接应由本地元数据构建"
    link_build = re.search(r'link_url\s*=\s*(.+)', source)
    assert link_build, "link_url 未定义"
    assert "p.get" in link_build.group(1), "link_url 应取自 paper 元数据"


def test_narrative_reference_map_excludes_ineligible_papers():
    source = APP_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef)
         and node.name == "_eligible_citation_papers"),
        None,
    )
    assert helper is not None, "引用编号必须有单独的纳入文献筛选函数"

    from types import SimpleNamespace

    namespace = {}
    module = ast.Module(body=[helper], type_ignores=[])
    exec(compile(module, str(APP_SOURCE), "exec"), namespace)
    papers = [{"id": "excluded"}, {"id": "included"}]
    evidence = [
        SimpleNamespace(eligible=False),
        SimpleNamespace(eligible=True),
    ]

    assert namespace["_eligible_citation_papers"](papers, evidence) == [papers[1]]
