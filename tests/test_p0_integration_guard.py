"""P0 集成守门测试：防止页面再次绕过结构化分析引擎。

背景：项目曾出现“只有纳入检索结果、没有任何分析文字”的故障，
根因是页面直接调用 ``_complete()`` 生成 Markdown，再用正则解析分数，
完全绕过了 ``ConsensusAnalyzer.analyze()`` 与确定性评分。

这组测试锁定“页面必须走结构化分析引擎”这一契约，
避免后续重构（或在压力下抄近路）再次回退到单次 LLM 调用。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(scope="module")
def app_tree() -> ast.Module:
    return ast.parse(APP_SOURCE.read_text(encoding="utf-8"))


def _called_attribute_names(tree: ast.Module) -> set[str]:
    """收集源码中所有 ``x.y(...)`` 形式的被调用方法名。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_page_calls_structured_analyzer(app_tree):
    """页面必须存在 analyzer.analyze(...) 调用。"""
    analyze_calls = [
        node
        for node in ast.walk(app_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "analyze"
    ]
    assert analyze_calls, "app.py 未调用任何 .analyze()——结构化分析引擎被绕过"


def test_page_instantiates_consensus_analyzer(app_tree):
    """页面必须实例化 ConsensusAnalyzer。"""
    called = _called_attribute_names(app_tree)
    assert "ConsensusAnalyzer" in called, (
        "app.py 未实例化 ConsensusAnalyzer——共识分析可能退回单次 LLM 调用"
    )


def test_page_does_not_bypass_with_private_complete(app_tree):
    """分析路径不得直接调用引擎私有方法 ``_complete()``。

    ``app.py`` 中对 ``_complete`` 的唯一合法用途是「设置」页的连接自检，
    因此这里精确限定：**不得出现在 analyze 调用所在的函数作用域内**。
    """
    violations = []
    for func in ast.walk(app_tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body_calls = {
            node.func.attr
            for node in ast.walk(func)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        if "_complete" in body_calls and "analyze" in body_calls:
            violations.append(func.name)

    assert not violations, (
        f"以下函数同时调用了 _complete() 和 analyze()，可能绕过结构化分析：{violations}"
    )


def test_page_does_not_parse_score_from_markdown(app_tree):
    """不得再用正则从 Markdown 里解析共识分（P0 明确删除的逻辑）。"""
    suspicious = []
    for node in ast.walk(app_tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr
            elif isinstance(node.func, ast.Name):
                name = node.func.id
            if name in {"search", "findall", "match"}:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        pattern = arg.value
                        if "共识" in pattern or "consensus" in pattern.lower():
                            suspicious.append(pattern[:60])

    assert not suspicious, (
        f"检测到从文本解析共识分的逻辑，应改为读取结构化结果：{suspicious}"
    )


def test_score_values_come_from_analysis_object(app_tree):
    """页面渲染的分数必须取自 analysis.score，而非模型输出。"""
    source = APP_SOURCE.read_text(encoding="utf-8")
    assert "analysis.score" in source or "score = analysis.score" in source, (
        "app.py 未从 analysis.score 读取分数——存在模型自生成分数的风险"
    )


def test_page_surfaces_insufficient_evidence_instead_of_silence():
    """证据不足时必须给出可见提示与扩充检索建议，不能静默白屏。

    这是原始故障的直接回归防线：用户曾报告“只有纳入检索结果的文献输出了”，
    页面既不显示分析文字也不显示任何提示。
    """
    source = APP_SOURCE.read_text(encoding="utf-8")

    assert "needs_search_expansion" in source, (
        "app.py 未处理证据不足状态——可能再次出现无提示的空输出"
    )
    assert "扩充检索" in source, "证据不足时未向用户给出可操作的扩充检索建议"
    assert "合格的方向性证据不足 5 篇，无法给出置信的共识判定。" not in source, (
        "仍在使用已被门控语义废弃的旧提示分支"
    )


def test_meter_visibility_requires_binary_question_and_displayable_score():
    source = APP_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef)
         and node.name == "_should_render_consensus_meter"),
        None,
    )
    assert helper is not None, "缺少可测试的 Meter 展示资格判断"

    namespace = {
        "is_meter_displayable": lambda score, minimum: score["displayable"],
        "MIN_DIRECTIONAL_STUDIES": 5,
    }
    module = ast.Module(body=[helper], type_ignores=[])
    exec(compile(module, str(APP_SOURCE), "exec"), namespace)
    should_render = namespace["_should_render_consensus_meter"]

    assert should_render("yes_no", {"displayable": True}) is True
    assert should_render("yes_no", {"displayable": False}) is False
    assert should_render("open_ended", {"displayable": True}) is False


def test_meter_uses_qualified_counts_not_raw_counts():
    source = APP_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "render_consensus")
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "qualified_stance_counts" in called
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "consensus"
            and node.attr == "qualified_stance_counts"
        ):
            pytest.fail(
                "qualified_stance_counts must be imported at module scope"
            )
