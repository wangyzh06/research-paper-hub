"""Reproduce the exact UnboundLocalError path reported by the user.

用户报告的 traceback:
    File "D:\\articles\\app.py", line 1235, in render_consensus
        funnel_stats.update(precision_stats)
    UnboundLocalError: cannot access local variable 'funnel_stats'

Root cause: the precision-filter stage (Plan A) was inserted BEFORE the line
that originally created funnel_stats, but the `.update()` call referenced it.

render_consensus() 有两条执行路径：
  路径 A（本次提交的搜索流程）：raw_candidates → 精度筛选 → 粗排 → LLM 精排
  路径 B（从 session_state 渲染已有结果）：analysis = st.session_state.get(...)
两条路径各自绑定 funnel_stats；本测试只约束**路径 A** 的绑定顺序。
"""
import ast
from pathlib import Path

import pytest

APP_SOURCE = Path(__file__).resolve().parents[1] / "app.py"


def _render_consensus_fn(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "render_consensus":
            return node
    pytest.fail("render_consensus not found in app.py")


def _find_start_clicked_block(fn: ast.FunctionDef) -> ast.If | None:
    """定位 `if start_clicked:` 块——搜索流程的整体容器。"""
    for stmt in fn.body:
        if isinstance(stmt, ast.If):
            desc = ast.unparse(stmt.test)
            if "start_clicked" in desc:
                return stmt
    return None


def _flatten(stmts) -> list[ast.stmt]:
    """递归展开嵌套块（with / if / try），返回全部语句。"""
    out: list[ast.stmt] = []
    for stmt in stmts:
        out.append(stmt)
        for attr in ("body", "finalbody", "orelse"):
            sub = getattr(stmt, attr, None)
            if sub and isinstance(sub, list):
                out.extend(_flatten(sub))
        for handler in getattr(stmt, "handlers", []) or []:
            out.extend(_flatten([handler]))
    return out


def _funnel_assign_lines(stmts) -> list[int]:
    """funnel_stats 的绑定行号。

    需同时处理两种形式：
      - ``ast.Assign``     → funnel_stats = {...}
      - ``ast.AnnAssign``  → funnel_stats: dict = {...}（当前代码使用带注解形式）
    """
    lines = []
    for s in _flatten(stmts):
        if isinstance(s, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "funnel_stats" for t in s.targets):
                lines.append(s.lineno)
        elif isinstance(s, ast.AnnAssign):
            if isinstance(s.target, ast.Name) and s.target.id == "funnel_stats":
                lines.append(s.lineno)
    return lines


def _update_lines(stmts) -> list[int]:
    lines = []
    for stmt in stmts:
        for node in ast.walk(stmt):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "funnel_stats"):
                lines.append(node.lineno)
    return lines


def _call_lines(stmts, func_name: str) -> list[int]:
    return [
        node.lineno for stmt in stmts
        for node in ast.walk(stmt)
        if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == func_name)
    ]


def test_search_path_binds_funnel_stats_before_any_update():
    """搜索流程内：所有 funnel_stats.update() 必须出现在首次赋值之后
    ——这正是修复用户报告的 UnboundLocalError 的约束。"""
    tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
    fn = _render_consensus_fn(tree)
    block = _find_start_clicked_block(fn)
    assert block is not None, "未找到 if start_clicked: 搜索流程块"

    stmts = list(block.body)
    assigns = _funnel_assign_lines(stmts)
    updates = _update_lines(stmts)

    assert updates, "搜索流程中未找到 funnel_stats.update() 调用"
    assert assigns, "搜索流程中必须初始化 funnel_stats"
    assert min(assigns) < min(updates), (
        f"首次赋值在第 {min(assigns)} 行，但第一次 update 在第 {min(updates)} 行——"
        "这会导致 UnboundLocalError"
    )


def test_funnel_stats_initialised_before_precision_filter():
    """精度筛选之前 funnel_stats 必须已初始化。"""
    tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
    fn = _render_consensus_fn(tree)
    block = _find_start_clicked_block(fn)
    assert block is not None

    stmts = list(block.body)
    assigns = _funnel_assign_lines(stmts)
    precision_calls = _call_lines(stmts, "precision_filter")

    assert precision_calls, "precision_filter 调用未找到"
    assert assigns, "搜索流程必须绑定 funnel_stats"
    assert min(assigns) < min(precision_calls), (
        "funnel_stats 必须在精度筛选之前初始化"
    )


def test_search_path_assigns_funnel_stats_exactly_once():
    """搜索流程只应赋值一次；后续阶段用 update 累加，避免丢失前置统计。"""
    tree = ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
    fn = _render_consensus_fn(tree)
    block = _find_start_clicked_block(fn)
    assert block is not None

    assigns = _funnel_assign_lines(list(block.body))
    assert len(assigns) == 1, (
        f"搜索流程中 funnel_stats 被赋值 {len(assigns)} 次 {assigns}，应为 1 次"
    )


def test_app_still_parses():
    """保障 app.py 可被解析（语法层面的基本健康检查）。"""
    ast.parse(APP_SOURCE.read_text(encoding="utf-8"))
