import asyncio
import time
import streamlit as st
from engine import (
    init_db, save_paper_to_db, add_to_collection, remove_from_collection,
    get_all_collections, get_collection_detail, create_collection, delete_collection, export_bibtex,
    save_note, get_note, delete_note, get_all_notes,
    create_tag, get_all_tags, delete_tag, add_tag_to_paper, remove_tag_from_paper, get_paper_tags,
    OpenAlexClient, LLMClient, PDFProcessor, SearchEngine, LLM_PROVIDERS,
    DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER,
)
from app_support import (
    api_key_fingerprint,
    build_llm_config,
    consensus_rows,
    format_search_diagnostics,
    llm_secret_name,
    search_cache_ttl,
)
from consensus import (
    ConsensusAnalyzer,
    MAX_SUBQUESTION_PAPERS,
    MIN_DIRECTIONAL_STUDIES,
    STUDY_LABELS,
    is_meter_displayable,
    needs_search_expansion,
    qualified_stance_counts,
)


st.set_page_config(page_title="ResearchPaperHub", page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    .paper-card { border: 1px solid #e0e0e0; border-radius: 12px; padding: 16px; margin: 8px 0; background: #fff; }
    .paper-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .paper-meta { font-size: 0.85rem; color: #666; display: flex; gap: 16px; flex-wrap: wrap; }
    .nav-btn { text-align: left; padding: 10px 16px; border-radius: 8px; border: none; background: transparent;
               cursor: pointer; font-size: 0.95rem; width: 100%; margin: 2px 0; transition: background 0.2s; }
    .nav-btn:hover { background: rgba(67,97,238,0.1); }
    .nav-btn.active { background: rgba(67,97,238,0.15); color: #4361ee; font-weight: 600; }
    .stat-card { text-align: center; padding: 24px; border: 1px solid #e0e0e0; border-radius: 12px; background: #fff; }
    .stat-value { font-size: 2.5rem; font-weight: 700; color: #4361ee; }
    .stat-label { font-size: 0.85rem; color: #888; margin-top: 4px; }

    /* Consensus 官网风格组件样式 */
    .cs-meter-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin: 16px 0 8px 0;
    }
    .cs-meter-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 18px;
    }
    .cs-meter-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
        line-height: 1.4;
    }
    .cs-meter-badge {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.75rem;
        font-weight: 600;
        color: #4b5563;
        background: #f3f4f6;
        padding: 3px 8px;
        border-radius: 8px;
        letter-spacing: 0.5px;
    }
    .cs-meter-bar-container {
        display: flex;
        width: 100%;
        height: 22px;
        border-radius: 6px;
        overflow: hidden;
        background: #f3f4f6;
        margin-bottom: 14px;
        gap: 2px;
    }
    .cs-bar-yes {
        background: #3bcdaa;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 6px;
        color: #ffffff;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem;
        font-weight: 700;
        transition: width 0.5s ease;
    }
    .cs-bar-possibly {
        background: #fdc700;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 4px;
        color: #ffffff;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem;
        font-weight: 700;
        transition: width 0.5s ease;
    }
    .cs-bar-mixed {
        background: #ff6900;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 6px;
        color: #ffffff;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem;
        font-weight: 700;
        transition: width 0.5s ease;
    }
    .cs-bar-no {
        background: #ff6467;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 6px;
        color: #ffffff;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem;
        font-weight: 700;
        transition: width 0.5s ease;
    }
    .cs-meter-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        align-items: center;
    }
    .cs-legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.82rem;
        color: #374151;
    }
    .cs-legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 2px;
    }
    .cs-legend-pct {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: #6b7280;
        font-weight: 600;
    }
    .cs-figure-caption {
        font-size: 0.75rem;
        color: #6b7280;
        margin: 6px 0 20px 2px;
    }
    .cs-figure-tag {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 6px;
    }

    /* 搜索策略 Search Strategy 三联卡片 */
    .cs-strategy-box {
        margin: 16px 0;
    }
    .cs-strategy-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 10px;
    }
    .cs-strategy-cards {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }
    .cs-strategy-card {
        flex: 1;
        min-width: 160px;
        background: #f9fafb;
        border: 1px solid #f3f4f6;
        border-radius: 12px;
        padding: 12px 16px;
    }
    .cs-strategy-num {
        font-size: 1.35rem;
        font-weight: 700;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: #111827;
        line-height: 1.2;
    }
    .cs-strategy-label {
        font-size: 0.75rem;
        color: #6b7280;
        font-weight: 500;
        margin-top: 2px;
    }
    .cs-strategy-desc {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 6px;
        line-height: 1.35;
    }

    /* 正文引用小徽章 (Author Year Badge) */
    .cs-cite-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 1px 6px;
        font-size: 0.72rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        color: #111827;
        font-weight: 600;
        margin: 0 3px;
        vertical-align: middle;
        text-decoration: none;
    }
    .cs-cite-badge:hover {
        background: #e5e7eb;
        border-color: #d1d5db;
    }
    .cs-cite-year {
        color: #6b7280;
        font-weight: 500;
    }

    /* 报告正文排版优化 */
    .cs-report-container {
        font-size: 0.95rem;
        line-height: 1.8;
        color: #1f2937;
    }
    .cs-report-container h2 {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin-top: 28px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #f3f4f6;
    }
    .cs-report-container h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: #111827;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .cs-report-container table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin: 16px 0;
    }
    .cs-report-container th {
        background: #f9fafb;
        color: #374151;
        font-weight: 600;
        text-align: left;
        padding: 10px 14px;
        border-bottom: 2px solid #e5e7eb;
    }
    .cs-report-container td {
        padding: 10px 14px;
        border-bottom: 1px solid #f3f4f6;
        vertical-align: top;
    }
    .cs-report-container tr:hover td {
        background: #fbfbfb;
    }
    .cs-gap-cell {
        display: inline-block;
        padding: 2px 8px;
        background: #fef2f2;
        color: #ef4444;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-weight: 700;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================
defaults = {
    "page": "home", "search_query": "", "search_results": [],
    "search_cache": {}, "search_response": None,
    "show_summary": "", "show_chat": "", "show_graph": "", "show_similar": "", "show_note": "",
    "compare_selected": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def navigate(page: str):
    st.session_state.page = page
    st.rerun()


def get_llm_config():
    """Resolve the active provider without exposing its API key to cache entries."""
    provider = st.session_state.get("llm_provider", DEFAULT_LLM_PROVIDER)
    model = st.session_state.get("llm_model", "")
    api_key = st.session_state.get("llm_api_key", "")

    # Priority: session state > st.secrets > env var
    if not api_key:
        try:
            secrets_key = llm_secret_name(provider)
            api_key = st.secrets.get(secrets_key, "") if secrets_key else ""
        except Exception:
            pass

    return build_llm_config(provider, model, api_key)


def get_llm_client():
    """Create an LLM client using the current page configuration."""
    config = get_llm_config()
    return LLMClient(
        provider=config.provider,
        api_key=config.api_key,
        model=config.model,
    )


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 ResearchPaperHub")
    st.caption("v4.1 — 智能学术研究助手")
    st.divider()
    pages = [("🏠 首页", "home"), ("🔍 论文搜索", "search"), ("📄 PDF 上传", "pdf"),
             ("📚 文献库", "collections"), ("📊 共识分析", "consensus"), ("⚖️ 多论文对比", "compare"),
             ("📝 我的笔记", "notes"), ("🏷️ 标签管理", "tags"), ("🎨 画布", "canvas"),
             ("⚙️ 设置", "settings")]
    for label, key in pages:
        active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if active else "secondary"):
            navigate(key)


# ============================================================
# Home Page
# ============================================================
def render_home():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<h1 style='text-align:center;margin-top:60px'>🎓 ResearchPaperHub</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;color:#666;font-size:1.1rem'>智能学术研究助手 — 论文发现 · AI 阅读 · 共识分析</p>", unsafe_allow_html=True)
        st.divider()

        q = st.text_input("🔍 搜索论文、作者或研究领域...", key="home_search",
                          placeholder="例如：deep learning, attention mechanism, drug discovery")
        c1, c2 = st.columns(2)
        if c1.button("🔍 搜索", use_container_width=True, type="primary") and q.strip():
            st.session_state.search_query = q.strip()
            navigate("search")
        if c2.button("🎲 随便看看", use_container_width=True):
            st.session_state.search_query = "machine learning"
            navigate("search")

        st.divider()
        st.markdown("### 快捷入口")
        cols = st.columns(3)
        with cols[0]:
            with st.container(border=True):
                st.markdown("#### 📚 文献库")
                st.caption("管理你的论文收藏")
                if st.button("进入文献库", key="go_col", use_container_width=True):
                    navigate("collections")
        with cols[1]:
            with st.container(border=True):
                st.markdown("#### 📊 共识分析")
                st.caption("分析研究领域共识度")
                if st.button("开始分析", key="go_con", use_container_width=True):
                    navigate("consensus")
        with cols[2]:
            with st.container(border=True):
                st.markdown("#### ⚖️ 多论文对比")
                st.caption("并排对比多篇论文")
                if st.button("开始对比", key="go_cmp", use_container_width=True):
                    navigate("compare")


# ============================================================
# Search Page
# ============================================================
def _search_cache_key(query: str, limit: int, sort: str, config) -> tuple:
    return (
        " ".join(query.split()).casefold(),
        limit,
        sort,
        config.provider,
        config.model,
        api_key_fingerprint(config.api_key),
    )


def _search_with_cache(query: str, limit: int = 100, sort: str = "relevance",
                       force_refresh: bool = False):
    config = get_llm_config()
    key = _search_cache_key(query, limit, sort, config)
    cache = st.session_state.search_cache
    now = time.monotonic()
    if force_refresh:
        cache.pop(key, None)
    cached = cache.get(key)
    if cached and cached["expires_at"] > now:
        return cached["response"]

    async def _s():
        engine = SearchEngine(llm_config=config)
        return await engine.search_with_diagnostics(
            query, limit, source="auto", sort=sort
        )

    response = asyncio.run(_s())
    ttl = search_cache_ttl(response)
    if ttl:
        cache[key] = {"response": response, "expires_at": now + ttl}
        if len(cache) > 32:
            oldest_key = min(cache, key=lambda item: cache[item]["expires_at"])
            cache.pop(oldest_key, None)
    return response


def _cached_search(query: str, limit: int = 100,
                   sort: str = "relevance") -> list[dict]:
    """Compatibility wrapper used by the comparison page."""
    return list(_search_with_cache(query, limit, sort).papers)


def render_search():
    st.markdown("## 🔍 论文搜索")

    c1, c2 = st.columns([5, 1])
    with c1:
        q = st.text_input("搜索关键词", value=st.session_state.get("search_query", ""),
                          key="search_input", placeholder="输入关键词搜索论文...")
    with c2:
        st.write("")
        retry_requested = st.button("🔄 重试", use_container_width=True)
    if (q or "").strip():
        st.session_state.search_query = (q or "").strip()

    if st.session_state.search_query:
        page_size = 30
        if "search_page" not in st.session_state:
            st.session_state.search_page = 0
        if "search_sort" not in st.session_state:
            st.session_state.search_sort = "relevance"

        sort_options = {"relevance": "相关性", "year": "年份", "citations": "引用数"}
        st.session_state.search_sort = st.radio("排序方式",
            list(sort_options.keys()), format_func=lambda k: sort_options[k],
            horizontal=True, key="sort_radio",
            index=list(sort_options.keys()).index(st.session_state.search_sort))

        search_context = (
            st.session_state.search_query.casefold(),
            st.session_state.search_sort,
        )
        if st.session_state.get("last_search_context") != search_context:
            st.session_state.search_page = 0
            st.session_state.last_search_context = search_context

        with st.spinner("搜索中..."):
            response = _search_with_cache(
                st.session_state.search_query,
                limit=200,
                sort=st.session_state.search_sort,
                force_refresh=retry_requested,
            )
            results = list(response.papers)
            st.session_state.search_results = results
            st.session_state.search_response = response

        diagnostics = format_search_diagnostics(response)
        if diagnostics["all_failed"]:
            failed = "、".join(diagnostics["failed_sources"])
            st.error(f"所有文献源均检索失败（{failed}）。请检查网络或认证配置后重试。")
        elif diagnostics["failed_sources"]:
            failed = "、".join(
                f"{name}（{diagnostics['errors'][name]['category']}）"
                for name in diagnostics["failed_sources"]
            )
            st.warning(f"部分文献源暂时失败：{failed}。当前结果来自其余可用来源。")

        source_counts = " · ".join(
            f"{name}: {count}" for name, count in diagnostics["source_counts"].items()
        )
        if source_counts:
            st.caption(f"来源返回量：{source_counts}")
        if diagnostics["translation_used"]:
            st.caption("已使用 GLM 生成英文检索变体，以提高中文主题的召回率。")
        elif diagnostics["ai_error"]:
            st.caption("GLM 查询扩展不可用，本次已自动回退为原始关键词检索。")

        results = st.session_state.search_results
        total = len(results)
        start = st.session_state.search_page * page_size
        end = min(start + page_size, total)
        page_results = results[start:end]

        if results:
            st.success(f"找到 {total} 篇论文，显示第 {start+1}-{end} 篇")
            for i, paper in enumerate(page_results):
                render_paper_card(paper, f"s_{start+i}")

            # Pagination
            total_pages = (total + page_size - 1) // page_size
            if total_pages > 1:
                cols = st.columns([2, 1, 1, 1, 2])
                with cols[1]:
                    if st.button("← 上一页", disabled=st.session_state.search_page == 0, use_container_width=True):
                        st.session_state.search_page = max(0, st.session_state.search_page - 1)
                        st.rerun()
                with cols[2]:
                    st.caption(f"{st.session_state.search_page+1}/{total_pages} 页")
                with cols[3]:
                    if st.button("下一页 →", disabled=st.session_state.search_page >= total_pages - 1, use_container_width=True):
                        st.session_state.search_page = min(total_pages - 1, st.session_state.search_page + 1)
                        st.rerun()

            # CiteSpace-inspired analysis
            st.divider()
            st.markdown("## 📊 文献分析")
            tab1, tab2, tab3 = st.tabs(["🔥 突变检测", "🔬 聚类分析", "📈 时间线"])

            with tab1:
                st.markdown("### 突变检测（热点论文）")
                st.caption("按年均引用量排序，发现研究热点")
                from engine import detect_burst
                burst = detect_burst(results, top_n=10)
                for i, p in enumerate(burst):
                    with st.container(border=True):
                        c1, c2 = st.columns([4, 1])
                        with c1:
                            st.markdown(f"**{p.get('title', '无标题')}**")
                            meta = []
                            if p.get("year"): meta.append(f"📅 {p['year']}")
                            if p.get("citation_count"): meta.append(f"📊 引用 {p['citation_count']}")
                            if meta: st.caption(" · ".join(meta))
                        with c2:
                            st.metric("年均引用", f"{p['_cpy']}")
                            st.caption(f"发表 {p['_age']} 年")

            with tab2:
                st.markdown("### 聚类分析（主题分组）")
                st.caption("自动将论文按研究主题分组")
                from engine import cluster_papers
                clusters = cluster_papers(results, n_clusters=min(5, max(2, len(results) // 10)))
                for cluster in clusters.get("clusters", []):
                    with st.expander(f"📁 {cluster['label']} ({len(cluster['papers'])} 篇)", expanded=False):
                        for p in cluster["papers"][:5]:
                            st.markdown(f"- {p.get('title', '无标题')} ({p.get('year', '')})")
                        if len(cluster["papers"]) > 5:
                            st.caption(f"... 还有 {len(cluster['papers']) - 5} 篇")

            with tab3:
                st.markdown("### 时间线（研究趋势）")
                st.caption("按年份统计论文数量和引用趋势")
                from engine import timeline_analysis
                timeline = timeline_analysis(results)
                if timeline.get("timeline"):
                    # Show as bar chart
                    try:
                        import plotly.express as px
                        import pandas as pd
                        df = pd.DataFrame(timeline["timeline"])
                        fig = px.bar(df, x="year", y="count", hover_data=["total_citations", "avg_citations"],
                                     labels={"year": "年份", "count": "论文数量", "total_citations": "总引用", "avg_citations": "平均引用"},
                                     title="论文数量随时间变化")
                        st.plotly_chart(fig, use_container_width=True)
                    except ImportError:
                        # Fallback to simple display
                        for t in timeline["timeline"]:
                            st.markdown(f"**{t['year']}**: {t['count']} 篇, 总引用 {t['total_citations']}, 平均引用 {t['avg_citations']}")
                else:
                    st.info("无时间线数据")
        elif not diagnostics["all_failed"]:
            st.warning("未找到相关论文，请尝试其他关键词")


def render_paper_card(paper: dict, key: str):
    with st.expander(f"📄 {paper.get('title', '无标题')}", expanded=True):
        authors = paper.get("authors", [])
        if authors:
            names = ", ".join(a.get("name", "") for a in authors[:5])
            if len(authors) > 5:
                names += f" 等 {len(authors)} 人"
            st.caption(f"👤 {names}")

        meta_items = []
        if paper.get("year"):
            meta_items.append(f"📅 {paper['year']}")
        if paper.get("venue"):
            meta_items.append(f"📖 {paper['venue']}")
        if paper.get("citation_count"):
            meta_items.append(f"📊 引用 {paper['citation_count']}")
        if meta_items:
            st.markdown(f'<div class="paper-meta">{" · ".join(meta_items)}</div>', unsafe_allow_html=True)

        # Tags display
        paper_id = paper.get("id")
        if paper_id:
            paper_tags = get_paper_tags(paper_id)
            if paper_tags:
                tag_html = " ".join(
                    f'<span style="background:{t["color"]};color:white;padding:2px 8px;border-radius:8px;font-size:0.75rem">{t["name"]}</span>'
                    for t in paper_tags
                )
                st.markdown(tag_html, unsafe_allow_html=True)

        abstract = paper.get("abstract", "")
        if abstract:
            st.markdown(abstract[:600] + ("..." if len(abstract) > 600 else ""))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        if c1.button("🤖 AI 摘要", key=f"{key}_sum"):
            _render_ai_summary(paper, key)

        if c2.button("💬 对话", key=f"{key}_chat"):
            st.session_state.show_chat = key
            st.rerun()

        if c3.button("🔗 图谱", key=f"{key}_graph"):
            st.session_state.show_graph = key
            st.rerun()

        if c4.button("🔍 相似", key=f"{key}_similar"):
            st.session_state.show_similar = key
            st.rerun()

        if c5.button("📝 笔记", key=f"{key}_note"):
            st.session_state.show_note = key
            st.rerun()

        with c6.popover("📚 加入文献库"):
            cols = get_all_collections()
            if cols:
                for c in cols:
                    if st.button(f"📁 {c['name']} ({c['papers_count']})", key=f"{key}_add_{c['id']}"):
                        pid = save_paper_to_db(paper)
                        add_to_collection(c["id"], pid)
                        st.toast(f"✅ 已加入「{c['name']}」")
            else:
                st.caption("暂无文献库，请先创建")

    # Inline panels
    if st.session_state.show_chat == key:
        _render_chat_panel(paper, key)
    if st.session_state.show_graph == key:
        _render_graph_panel(paper, key)
    if st.session_state.show_similar == key:
        _render_similar_panel(paper, key)
    if st.session_state.show_note == key:
        _render_note_panel(paper, key)


# ============================================================
# AI Summary
# ============================================================
def _render_ai_summary(paper: dict, key: str):
    abstract = paper.get("abstract", "")
    if not abstract:
        st.warning("该论文无摘要")
        return
    with st.spinner("AI 分析中..."):
        async def _s():
            async with get_llm_client() as c:
                return await c.summarize(abstract)
        result = asyncio.run(_s())
    st.markdown("### 🤖 AI 摘要")
    st.markdown(result)


# ============================================================
# AI Chat Panel
# ============================================================
def _render_chat_panel(paper: dict, key: str):
    ck = f"chat_{key}"
    if ck not in st.session_state:
        st.session_state[ck] = []

    st.divider()
    st.markdown("### 💬 AI 论文对话")
    pid = paper.get("id", key)

    for msg in st.session_state[ck]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input(f"向 AI 提问这篇论文...", key=f"{key}_chat_input")
    if user_msg:
        st.session_state[ck].append({"role": "user", "content": user_msg})

        abstract = paper.get("abstract", "")
        if paper.get("pdf_path"):
            try:
                from engine import PDFProcessor
                abstract = PDFProcessor().extract_text(paper["pdf_path"])
            except Exception:
                pass

        with st.spinner("AI 思考中..."):
            async def _chat():
                async with get_llm_client() as c:
                    return await c.chat(st.session_state[ck], abstract)
            reply = asyncio.run(_chat())
        st.session_state[ck].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("✖ 关闭对话", key=f"{key}_close_chat"):
        st.session_state.show_chat = ""
        st.rerun()


# ============================================================
# Citation Graph
# ============================================================
def _render_graph_panel(paper: dict, key: str):
    st.divider()
    st.markdown("### 🔗 引用图谱")

    if "graph_depth" not in st.session_state:
        st.session_state.graph_depth = 1
    if "graph_year_filter" not in st.session_state:
        st.session_state.graph_year_filter = (1970, 2030)

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        depth = st.select_slider("图谱深度", options=[1, 2], value=st.session_state.graph_depth,
                                  help="深度1：直接引用；深度2：引用的引用")
        st.session_state.graph_depth = depth
    with c2:
        min_year = st.number_input("起始年份", 1970, 2030, st.session_state.graph_year_filter[0], step=1, key=f"{key}_minyr")
    with c3:
        max_year = st.number_input("截止年份", 1970, 2030, st.session_state.graph_year_filter[1], step=1, key=f"{key}_maxyr")
    st.session_state.graph_year_filter = (min_year, max_year)

    with st.spinner("生成引用图谱..."):
        async def _build_deep(d: int):
            pid = paper.get("id", "")
            nodes_map, edges = {}, []

            async def fetch_level(wid: str, current_depth: int, relation: str):
                if current_depth > d or wid in nodes_map:
                    return
                if wid == paper.get("id", ""):
                    cc = paper.get("citation_count", 0)
                    yr = paper.get("year", "")
                    abstract = (paper.get("abstract") or "")[:200]
                    nodes_map[wid] = {"title": paper.get("title", "当前论文")[:60],
                                      "year": yr, "citation_count": cc, "abstract": abstract,
                                      "url": paper.get("url", ""), "is_center": True, "depth": 0}
                async with OpenAlexClient() as oa:
                    cited = await oa.get_cited_by(wid, limit=25)
                    refs = await oa.get_references(wid, limit=25)
                for p in cited.get("results", []):
                    nid = p.get("id", "")
                    if not nid or nid in nodes_map: continue
                    yr = p.get("year") or 9999
                    if yr < min_year or yr > max_year: continue
                    nodes_map[nid] = {"title": (p.get("title") or "")[:60],
                                      "year": yr, "citation_count": p.get("citation_count", 0),
                                      "abstract": "", "url": f"https://openalex.org/{nid}",
                                      "is_center": False, "depth": current_depth}
                    edges.append({"from": nid, "to": wid})
                    if current_depth < d: await fetch_level(nid, current_depth + 1, "citing")
                for p in refs.get("results", []):
                    nid = p.get("id", "")
                    if not nid or nid in nodes_map: continue
                    yr = p.get("year") or 9999
                    if yr < min_year or yr > max_year: continue
                    nodes_map[nid] = {"title": (p.get("title") or "")[:60],
                                      "year": yr, "citation_count": p.get("citation_count", 0),
                                      "abstract": "", "url": f"https://openalex.org/{nid}",
                                      "is_center": False, "depth": current_depth}
                    edges.append({"from": wid, "to": nid})
                    if current_depth < d: await fetch_level(nid, current_depth + 1, "ref")

            await fetch_level(pid, 1, "center")
            return nodes_map, edges

        nodes_map, edges = asyncio.run(_build_deep(depth))

    if not nodes_map:
        st.info("该论文暂无引用数据（或年份筛选无结果）")
    else:
        from pyvis.network import Network as PNetwork
        import networkx as nx
        G = nx.DiGraph()

        # Year color mapping: older=cool blue, newer=warm red
        years = [nd.get("year") or 2020 for nd in nodes_map.values()]
        yr_min, yr_max = min(years), max(years)
        yr_range = max(1, yr_max - yr_min)

        def year_color(yr):
            if not yr: return "#888888"
            ratio = (yr - yr_min) / yr_range
            r = int(100 + 155 * ratio)
            g = int(100 + 55 * (1 - ratio))
            b = int(200 - 100 * ratio)
            return f"#{r:02x}{g:02x}{b:02x}"

        # Citation-based sizing
        max_cc = max((nd.get("citation_count", 0) for nd in nodes_map.values()), default=1)
        def node_size(cc, is_center):
            if is_center: return 40
            return max(8, min(30, 10 + (cc / max(1, max_cc)) * 20))

        for nid, nd in nodes_map.items():
            yr_str = f" ({nd.get('year')})" if nd.get("year") else ""
            lt = nd["title"][:35] + yr_str
            tooltip = f"<b>{nd['title']}</b>"
            if nd.get("year"): tooltip += f"<br/>📅 {nd['year']}"
            tooltip += f"<br/>📊 引用 {nd.get('citation_count', 0)}"
            if nd.get("abstract"): tooltip += f"<br/><br/>{nd['abstract'][:150]}"
            if nd.get("url"): tooltip += f"<br/><a href='{nd['url']}' target='_blank'>打开论文</a>"
            is_c = nd.get("is_center", False)
            G.add_node(nid, label=lt, title=tooltip,
                       size=node_size(nd.get("citation_count", 0), is_c),
                       color=year_color(nd.get("year")) if not is_c else "#e74c3c")
        for e in edges:
            G.add_edge(e["from"], e["to"])

        net = PNetwork(height="650px", width="100%", directed=True)
        net.from_nx(G)
        net.set_options("""
        {
          "nodes": { "font": {"size": 13, "face": "Arial"}, "borderWidth": 2 },
          "edges": { "arrows": {"to": {"enabled": true}}, "smooth": {"type": "curvedCW"},
                     "color": {"color": "#aaa", "opacity": 0.5} },
          "interaction": { "hover": true, "tooltipDelay": 100, "navigationButtons": true },
          "physics": { "solver": "forceAtlas2Based", "stabilization": {"iterations": 300},
                       "forceAtlas2Based": {"gravitationalConstant": -50, "springLength": 200} }
        }
        """)
        net.save_graph("_graph.html")
        with open("_graph.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=670)

    st.caption(
        "🔴 当前论文 · 🔵← 旧论文 · 🔴→ 新论文 · 节点大小=引用数 · 悬停看详情 · 可拖拽")
    if st.button("✖ 关闭图谱", key=f"{key}_close_graph"):
        st.session_state.show_graph = ""
        st.rerun()


# ============================================================
# Similar Papers Panel
# ============================================================
def _render_similar_panel(paper: dict, key: str):
    st.divider()
    st.markdown("### 🔍 相似论文")

    title = paper.get("title", "")
    if not title:
        st.warning("无法查找相似论文")
        return

    with st.spinner("搜索相似论文..."):
        engine = SearchEngine()
        similar = asyncio.run(engine.search(title, limit=10, sort="relevance"))

    if similar:
        for s in similar:
            with st.container(border=True):
                st.markdown(f"**{s.get('title', '无标题')}**")
                authors = s.get("authors", [])
                if authors:
                    st.caption(f"👤 {', '.join(a.get('name', '') for a in authors[:3])}")
                meta = []
                if s.get("year"):
                    meta.append(f"📅 {s['year']}")
                if s.get("citation_count"):
                    meta.append(f"📊 引用 {s['citation_count']}")
                if meta:
                    st.caption(" · ".join(meta))
    else:
        st.info("未找到相似论文")

    if st.button("✖ 关闭", key=f"{key}_close_similar"):
        st.session_state.show_similar = ""
        st.rerun()


# ============================================================
# Note Panel
# ============================================================
def _render_note_panel(paper: dict, key: str):
    st.divider()
    st.markdown("### 📝 论文笔记")

    # Save paper to DB first to get paper_id
    paper_id = save_paper_to_db(paper)

    # Load existing note
    existing_note = get_note(paper_id)

    # Note editor
    note_content = st.text_area(
        "写下你的阅读笔记、想法、与你研究的关联...",
        value=existing_note,
        height=200,
        key=f"{key}_note_editor",
        placeholder="支持 Markdown 格式..."
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("💾 保存笔记", key=f"{key}_save_note"):
        save_note(paper_id, note_content)
        st.toast("✅ 笔记已保存")

    if c2.button("🗑️ 删除笔记", key=f"{key}_delete_note"):
        delete_note(paper_id)
        st.toast("✅ 笔记已删除")
        st.rerun()

    if st.button("✖ 关闭", key=f"{key}_close_note"):
        st.session_state.show_note = ""
        st.rerun()


# ============================================================
# Collections Page
# ============================================================
def render_collections():
    st.markdown("## 📚 文献库")

    if st.button("➕ 新建文献库", type="primary"):
        st.session_state.show_create = True
    if not st.session_state.get("show_create"):
        st.session_state.show_create = False

    if st.session_state.show_create:
        with st.form("create_collection_form"):
            name = st.text_input("文献库名称", key="col_name")
            desc = st.text_area("描述（可选）", key="col_desc")
            c1, c2 = st.columns(2)
            if c1.form_submit_button("✅ 创建"):
                if name.strip():
                    create_collection(name.strip(), desc.strip())
                    st.session_state.show_create = False
                    st.rerun()
            if c2.form_submit_button("取消"):
                st.session_state.show_create = False
                st.rerun()

    collections = get_all_collections()
    if not collections:
        st.info("暂无文献库，点击上方按钮创建")
        return

    # Display collections with enhanced cards
    cols = st.columns(3)
    for i, c in enumerate(collections):
        detail = get_collection_detail(c["id"])
        papers = detail["papers"] if detail else []
        total_cc = sum(p.get("citation_count", 0) or 0 for p in papers)
        avg_cc = round(total_cc / max(1, len(papers)), 1) if papers else 0
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### 📁 {c['name']}")
                st.caption(f"📄 {c['papers_count']} 篇 · 📊 总引用 {total_cc} · 均引用 {avg_cc}")
                if c.get("description"):
                    st.caption(c["description"])
                c1, c2, c3 = st.columns(3)
                if c1.button("查看", key=f"view_{c['id']}", use_container_width=True):
                    st.session_state.view_collection = c["id"]
                    st.rerun()
                if c2.button("📥 BibTeX", key=f"bib_{c['id']}", use_container_width=True):
                    bib = export_bibtex(c["id"])
                    if bib:
                        st.session_state.bibtex_data = bib
                        st.rerun()
                    else:
                        st.warning("该文献库暂无论文")
                if c3.button("删除", key=f"del_{c['id']}", use_container_width=True):
                    delete_collection(c["id"])
                    st.rerun()

    # BibTeX download dialog
    if st.session_state.get("bibtex_data"):
        with st.expander("📥 导出 BibTeX", expanded=True):
            st.code(st.session_state.bibtex_data, language="bibtex")
            st.download_button("💾 下载 .bib 文件", st.session_state.bibtex_data, "papers.bib", "text/plain")
            if st.button("关闭"):
                st.session_state.bibtex_data = None
                st.rerun()

    # View collection detail
    if st.session_state.get("view_collection"):
        cid = st.session_state.view_collection
        detail = get_collection_detail(cid)
        if detail:
            st.divider()
            st.markdown(f"### 📁 {detail['name']}")
            st.caption(detail.get("description", ""))

            papers = detail["papers"]
            total_cc = sum(p.get("citation_count", 0) or 0 for p in papers)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("论文数", len(papers))
            c2.metric("总引用", total_cc)
            c3.metric("平均引用", round(total_cc / max(1, len(papers)), 1))
            c4.metric("最高引用", max((p.get("citation_count", 0) or 0 for p in papers), default=0))

            if not papers:
                st.info("该文献库暂无论文")
            else:
                # Tag filter
                all_tags = get_all_tags()
                if all_tags:
                    tag_names = {t["id"]: t["name"] for t in all_tags}
                    selected_tags = st.multiselect("🏷️ 按标签筛选", list(tag_names.keys()),
                        format_func=lambda x: tag_names[x], key=f"tag_filter_{cid}")
                    if selected_tags:
                        papers = [p for p in papers if any(
                            t in [tp["id"] for tp in get_paper_tags(p["id"])] for t in selected_tags
                        )]
                        st.caption(f"筛选后: {len(papers)} 篇")

                # Sort
                sort_by = st.selectbox("排序", ["引用数", "年份"], key=f"sort_{cid}")
                if sort_by == "引用数":
                    papers.sort(key=lambda p: p.get("citation_count", 0) or 0, reverse=True)
                else:
                    papers.sort(key=lambda p: p.get("year") or 0, reverse=True)

                # Batch remove
                batch_mode = st.checkbox("批量删除模式", key=f"batch_{cid}")
                to_remove = []
                for p in papers:
                    with st.container(border=True):
                        c1, c2, c3 = st.columns([batch_mode and 0.5 or 4, 1.5, 1])
                        with c1:
                            if batch_mode:
                                if st.checkbox("", key=f"sel_{cid}_{p['id']}"):
                                    to_remove.append(p["id"])
                            st.markdown(f"**{p['title'][:80]}**")
                            meta = []
                            if p.get("year"): meta.append(str(p["year"]))
                            if p.get("venue"): meta.append(p["venue"])
                            if p.get("citation_count"): meta.append(f"引用 {p['citation_count']}")
                            if meta: st.caption(" · ".join(meta))
                        with c2:
                            if not batch_mode and st.button("🗑️", key=f"rem_{cid}_{p['id']}"):
                                remove_from_collection(cid, p["id"])
                                st.rerun()
                        with c3:
                            paper_tags = get_paper_tags(p["id"])
                            if paper_tags:
                                tag_html = " ".join(f'<span style="background:{t["color"]};color:#fff;padding:1px 6px;border-radius:6px;font-size:0.7rem">{t["name"]}</span>' for t in paper_tags)
                                st.markdown(tag_html, unsafe_allow_html=True)
                if batch_mode and to_remove:
                    if st.button(f"🗑️ 删除选中的 {len(to_remove)} 篇", type="primary"):
                        for pid in to_remove:
                            remove_from_collection(cid, pid)
                        st.rerun()

            c1, c2 = st.columns(2)
            if c1.button("🤖 推荐相似论文", key=f"rec_{cid}"):
                with st.spinner("搜索推荐中..."):
                    titles = " ".join(p.get("title", "") for p in detail["papers"][:5])
                    if titles:
                        engine = SearchEngine()
                        recs = asyncio.run(engine.search(titles[:200], limit=15, source="auto"))
                        if recs:
                            st.markdown("### 推荐论文")
                            for r in recs[:10]:
                                with st.container(border=True):
                                    st.markdown(f"**{r.get('title', '')}**")
                                    if r.get("year"): st.caption(f"📅 {r['year']}")

        if st.button("✖ 关闭"):
            st.session_state.view_collection = None
            st.rerun()


# ============================================================
# Consensus Page
# ============================================================
def _should_render_consensus_meter(question_type: str, score) -> bool:
    return question_type == "yes_no" and is_meter_displayable(
        score, MIN_DIRECTIONAL_STUDIES
    )


def _eligible_citation_papers(papers: list[dict], evidence) -> list[dict]:
    return [
        paper for paper, item in zip(papers, evidence)
        if item.eligible
    ]


def _render_subquestion_reviews(analysis) -> None:
    subquestions = list(getattr(analysis, "subquestions", ()) or ())
    review = getattr(analysis, "evidence_review", None) or {}
    claims = review.get("claims") or []
    if claims:
        st.markdown("#### 🔗 结论与证据关联")
        st.dataframe(claims, use_container_width=True, hide_index=True)

    if len(subquestions) > 1:
        st.markdown("#### 🧭 子问题拆解")
        for idx, question in enumerate(subquestions, start=1):
            st.markdown(f"{idx}. {question}")

    for sub_review in (getattr(analysis, "subquestion_reviews", ()) or ()):
        screened_ids = sub_review.get("screened_paper_ids") or []
        included_ids = sub_review.get("paper_ids") or []
        search_stats = sub_review.get("search_stats") or {}
        with st.expander(
            f"子问题证据：{sub_review.get('question', '未命名问题')} "
            f"（筛选 {len(screened_ids)} 篇 / 纳入 {len(included_ids)} 篇）",
            expanded=False,
        ):
            st.caption(
                f"独立检索召回 {search_stats.get('retrieved', 0)} 篇；"
                f"精度筛选后 {search_stats.get('precision_kept', len(screened_ids))} 篇；"
                "有效独立研究数："
                f"{sub_review.get('effective_independent_studies', 0):g}"
            )
            sub_review_data = sub_review.get("evidence_review") or {}
            sub_table = sub_review_data.get("evidence_table") or []
            if sub_table:
                st.dataframe(sub_table, use_container_width=True, hide_index=True)
            sub_claims = sub_review_data.get("claims") or []
            if sub_claims:
                st.markdown("**该子问题的结论—证据关联**")
                st.dataframe(sub_claims, use_container_width=True, hide_index=True)


def render_consensus():

    st.markdown("## 📊 共识分析")
    st.caption("检索相关研究，逐篇抽取结构化证据，计算可审计的共识度")

    topic = st.text_area("研究主题或问题", placeholder="例如：Does BERT outperform LSTM in sequence classification tasks? 或：in situ退火的具体方法以及对增益等参数的影响",
                         key="consensus_topic")

    col_btn, col_mode = st.columns([2, 4])
    with col_btn:
        start_clicked = st.button("🔬 开始分析", type="primary", use_container_width=True)
    with col_mode:
        analysis_mode = st.radio(
            "分析模式",
            ["pro", "deep"],
            index=0,
            horizontal=True,
            format_func=lambda x: "⚡ Pro 模式 (20篇·工艺参数对比/折中建议)" if x == "pro" else "🔬 Deep 模式 (50篇·学术引文网络/系统综述)",
            key="consensus_mode_selection",
        )

    if start_clicked:
        topic_text = (topic or "").strip()
        if not topic_text:
            st.warning("请先输入研究主题或问题。")
            return
        st.session_state.consensus_analysis = None
        st.session_state.consensus_papers = []
        st.session_state.consensus_search_response = None
        st.session_state.consensus_funnel_stats = None
        st.session_state.consensus_error = None
        st.session_state.consensus_mode = analysis_mode
        st.session_state.consensus_status = "searching"

        # 阶段 1：解析研究问题 PICO、领域与中立学术检索式
        with st.spinner("解析研究问题与学术检索式..."):
            try:
                from consensus import build_question_spec
                question_spec = asyncio.run(build_question_spec(topic_text, get_llm_client))
            except Exception:
                from consensus import QuestionSpec, heuristic_question_type
                question_spec = QuestionSpec.fallback(topic_text, heuristic_question_type(topic_text))

        # 阶段 1.5：按 Consensus Deep 的 7 个战略维度生成短技术检索式
        # （实测：长自然语言句子在 OpenAlex/Europe PMC/S2 上召回为 0，必须拆维度）
        dimension_queries: dict = {}
        subquestions: list[str] = []
        with st.spinner("生成 7 维战略检索式（对标 Consensus Deep）..."):
            try:
                from consensus import build_dimension_queries
                dimension_queries = asyncio.run(
                    build_dimension_queries(topic_text, get_llm_client, question_spec)
                )
            except Exception:
                dimension_queries = {}

            try:
                from consensus import decompose_question, is_complex_question
                if is_complex_question(topic_text, question_spec):
                    subquestions = asyncio.run(
                        decompose_question(topic_text, get_llm_client)
                    )
                    if len(subquestions) == 1 and subquestions[0].casefold() == topic_text.casefold():
                        subquestions = []
            except Exception:
                subquestions = []

        # 阶段 2：多维定向检索（OpenAlex + Europe PMC + S2 bulk，按维度逐组召回）
        search_query = (question_spec.neutral_query or topic_text).strip()
        search_variants = tuple(getattr(question_spec, "search_queries", (search_query,)))
        domain = getattr(question_spec, "domain", "general")
        limit_pool = 200 if analysis_mode == "deep" else 120


        # 展平维度检索式（去重），作为多查询召回输入
        flat_dimension_queries: list[str] = []
        for _dim_queries in dimension_queries.values():
            for _q in _dim_queries:
                if _q and _q not in flat_dimension_queries:
                    flat_dimension_queries.append(_q)

        with st.spinner(f"定向检索学术文献（领域：{domain}，{len(flat_dimension_queries) or 1} 组检索式）..."):
            async def _do_search():
                config = get_llm_config()
                engine = SearchEngine(llm_config=config)
                return await engine.search_consensus_candidates(
                    query=search_query,
                    domain=domain,
                    limit=limit_pool,
                    query_variants=search_variants,
                    dimension_queries=tuple(flat_dimension_queries),
                    # 引文网络扩展：Deep 模式更激进（对标 Consensus Deep 的
                    # 「following citation connections」），Pro 模式轻量。
                    # 分层 BFS：max_depth 可挖到多跳的奠基性文献；
                    # 双上限（入库 / 扩展）防止引文网络指数爆炸。
                    citation_expansion=True,
                    citation_seeds=6 if analysis_mode == "deep" else 3,
                    citation_neighbors=50 if analysis_mode == "deep" else 25,
                    citation_depth=2 if analysis_mode == "deep" else 1,
                    citation_papers_per_level=60 if analysis_mode == "deep" else 30,
                    citation_expansion_per_level=8 if analysis_mode == "deep" else 4,
                )
            search_response = asyncio.run(_do_search())

        diagnostics = format_search_diagnostics(search_response)
        if diagnostics["all_failed"]:
            st.session_state.consensus_status = "error"
            st.error("所有文献源均失败，无法开始共识分析。请检查网络和数据源配置。")
            return

        raw_candidates = list(search_response.papers)
        if not raw_candidates:
            st.session_state.consensus_status = "empty"
            st.warning("未找到可分析的相关论文，请尝试更通用或英文关键词。")
            return

        # 漏斗统计在此初始化：后续各阶段（精度筛选 / 粗排 / LLM 精排）依次累加。
        funnel_stats: dict = {"retrieved": len(raw_candidates)}

        # 阶段 3：本地实体预筛（方案 B）→ 精度筛选 → 本地粗排 → LLM 精排
        from consensus import (
            build_concept_terms,
            build_subject_terms,
            entity_subject_terms,
            local_entity_screen,
            precision_filter,
            rerank_consensus_papers,
            rerank_with_llm,
        )
        # 深度证据抽取的目标篇数。
        target_n = 24 if analysis_mode == "deep" else 16
        # 概念词来自 PICO 与 7 维度检索式；研究对象词（HBT/GaAs）才是切题判据
        concept_terms = build_concept_terms(
            question_spec, dimension_queries=flat_dimension_queries,
        )
        subject_terms = build_subject_terms(question_spec)
        entity_terms = entity_subject_terms(subject_terms)

        # 阶段 3a：本地实体级预筛（方案 B）
        # 召回调里大量论文在材料/器件层面就与问题无关（如问 GaAs HBT 却召回
        # 钙钛矿太阳能电池、IGZO TFT）。这类论文无需任何 LLM 调用即可排除。
        # 实测：200 篇中可安全排除约 74%，且与后续 precision_filter 判定完全一致
        # （额外排除的"切题论文"为 0），因此不改变最终纳入集合。
        with st.spinner(f"本地实体预筛（{len(raw_candidates)} 篇候选）..."):
            entity_kept, entity_excluded, entity_stats = local_entity_screen(
                raw_candidates, tuple(subject_terms),
            )
        funnel_stats["entity_screened"] = entity_stats["screened"]
        funnel_stats["entity_excluded"] = entity_stats["local_excluded"]
        if not entity_stats.get("disabled") and entity_stats["local_excluded"]:
            st.caption(
                f"本地实体预筛：{entity_stats['screened']} 篇 → "
                f"排除 {entity_stats['local_excluded']} 篇材料/器件不符"
                f"（实体判据：{'、'.join(sorted(entity_terms)) or '未识别'}）"
            )

        # 阶段 3b：精度优先——只保留研究对象切题的论文
        # 本领域窄问题的相关文献客观稀少（实测 Europe PMC 上 "InGaP GaAs HBT"
        # 仅 9 篇），强行凑数只会掺入大量跑题论文并使 Meter 失去意义。
        with st.spinner(f"精度筛选（{len(entity_kept)} 篇候选 → 切题文献）..."):
            on_topic, precision_stats = precision_filter(
                entity_kept,
                tuple(concept_terms),
                spec=question_spec,
                subject_terms=tuple(subject_terms),
                require_subject=bool(subject_terms),
            )
        funnel_stats.update(precision_stats)
        if on_topic:
            st.caption(
                f"精度筛选：{precision_stats['precision_input']} 篇 → "
                f"{precision_stats['precision_kept']} 篇切题"
                f"（剔除 {precision_stats['subject_filtered']} 篇主题不符）"
            )
        else:
            # 无切题文献：如实告知，而不是拿杂文献编造共识
            st.warning(
                f"未找到与「{question_spec.population if question_spec else ''}」"
                "直接相关的文献。共识分析需要切题证据，"
                "建议放宽问题范围或改用更常见的材料/器件名称后重试。"
            )
            st.session_state.consensus_status = "empty"
            return

        # 阶段 3c：本地粗排（概念覆盖 + 引用量 + 实证信号）
        coarse_papers, coarse_stats = rerank_consensus_papers(
            on_topic,
            spec=question_spec,
            target_count=60 if analysis_mode == "deep" else 40,
            min_abstract_len=80,
            concept_terms=tuple(concept_terms),
            min_concept_hits=1,
        )
        # 本地排除的论文保持可见（不静默丢弃），供审计
        if entity_excluded:
            st.session_state.consensus_entity_excluded = entity_excluded
        funnel_stats.update(coarse_stats)
        if not coarse_papers:
            coarse_papers = on_topic[: target_n * 2]

        # 阶段 3.5：LLM 精排 top N（复刻 Consensus「高精度模型只施用于 top 20」）
        # 过滤掉字面命中但主题无关的论文（如把 quantum annealing 误判为热退火）
        selected_papers = coarse_papers
        with st.spinner(f"AI 精排相关性（候选 {len(coarse_papers)} 篇 → 纳入 {target_n} 篇）..."):
            try:
                selected_papers, rerank_stats = asyncio.run(
                    rerank_with_llm(
                        coarse_papers,
                        topic_text,
                        get_llm_client,
                        spec=question_spec,
                        candidate_limit=len(coarse_papers),
                        target_count=target_n,
                    )
                )
                funnel_stats.update(rerank_stats)
            except Exception as rerank_error:
                # 精排失败不阻塞：回退本地粗排结果
                st.warning(f"AI 精排未完成，已回退本地排序：{type(rerank_error).__name__}")
                selected_papers = coarse_papers[:target_n]
        if not selected_papers:
            selected_papers = coarse_papers[:target_n]

        # P3：复杂问题的每个子问题独立检索、独立筛选，避免把一个子问题的
        # 候选集/筛选结果误当成其他子问题的证据。
        subquestion_papers: dict[str, list[dict]] = {}
        subquestion_search_stats: dict[str, dict] = {}
        if subquestions:
            with st.spinner(f"为 {len(subquestions)} 个子问题独立检索并筛选证据..."):
                async def _search_screen_subquestion(subquestion: str):
                    try:
                        sub_spec = await build_question_spec(subquestion, get_llm_client)
                        sub_engine = SearchEngine(llm_config=get_llm_config())
                        sub_response = await sub_engine.search_consensus_candidates(
                            query=(sub_spec.neutral_query or subquestion).strip(),
                            domain=getattr(sub_spec, "domain", domain),
                            limit=max(24, target_n * 2),
                            query_variants=tuple(
                                getattr(sub_spec, "search_queries", ())
                                or (sub_spec.neutral_query or subquestion,)
                            ),
                            dimension_queries=(),
                            citation_expansion=False,
                        )
                        candidates = list(sub_response.papers)
                        sub_concepts = build_concept_terms(sub_spec)
                        sub_subjects = build_subject_terms(sub_spec)
                        entity_kept, _, entity_stats = local_entity_screen(
                            candidates, tuple(sub_subjects)
                        )
                        screened, precision_stats = precision_filter(
                            entity_kept,
                            tuple(sub_concepts),
                            spec=sub_spec,
                            subject_terms=tuple(sub_subjects),
                            require_subject=bool(sub_subjects),
                        )
                        selected_for_analysis = screened[:MAX_SUBQUESTION_PAPERS]
                        stats = {
                            "domain": getattr(sub_spec, "domain", domain),
                            "retrieved": len(candidates),
                            "entity_excluded": entity_stats.get("local_excluded", 0),
                            "precision_kept": precision_stats.get("precision_kept", len(screened)),
                            "screened": len(selected_for_analysis),
                        }
                        return subquestion, selected_for_analysis, stats, ""
                    except Exception as error:
                        return subquestion, [], {"retrieved": 0, "screened": 0}, type(error).__name__

                async def _search_all_subquestions():
                    semaphore = asyncio.Semaphore(2)

                    async def bounded(subquestion: str):
                        async with semaphore:
                            return await _search_screen_subquestion(subquestion)

                    return await asyncio.gather(
                        *(bounded(subquestion) for subquestion in subquestions)
                    )

                for subquestion, candidates, stats, error_name in asyncio.run(
                    _search_all_subquestions()
                ):
                    subquestion_papers[subquestion] = candidates
                    subquestion_search_stats[subquestion] = stats
                    if error_name:
                        st.warning(f"子问题「{subquestion}」独立检索失败：{error_name}")
                    else:
                        st.caption(
                            f"子问题独立检索/筛选：{subquestion}，"
                            f"召回 {stats['retrieved']} 篇 → 纳入候选 {stats['screened']} 篇"
                        )

        # 阶段 4：证据抽取与双轨学术报告生成（严禁 AI 过程自述，输出纯学术报告）
        st.session_state.consensus_status = "analyzing"
        search_stats = {
            "retrieved": funnel_stats.get("retrieved", len(raw_candidates)),
            "filtered_valid": funnel_stats.get("with_abstract", len(selected_papers)),
        }
        with st.spinner(f"正在抽取结构化证据并生成报告（模式：{'Pro 技术分析' if analysis_mode == 'pro' else 'Deep 深度综述'}，纳入 {len(selected_papers)} 篇）..."):
            try:
                import consensus
                analyzer = consensus.ConsensusAnalyzer(get_llm_client, max_concurrency=4)
                # 使用批量抽取并传入选定模式
                analysis = asyncio.run(
                    analyzer.analyze(
                        topic_text,
                        selected_papers,
                        question_spec,
                        search_stats,
                        use_batch=True,
                        mode=analysis_mode,
                        subquestions=subquestions,
                        subquestion_papers=subquestion_papers,
                        subquestion_search_stats=subquestion_search_stats,
                    )
                )

                st.session_state.consensus_status = "done"
            except Exception as error:
                st.session_state.consensus_error = f"{type(error).__name__}: {error}"
                st.session_state.consensus_status = "error"
                analysis = None

        st.session_state.consensus_analysis = analysis
        st.session_state.consensus_question_spec = question_spec
        st.session_state.consensus_papers = selected_papers
        st.session_state.consensus_search_response = search_response
        st.session_state.consensus_funnel_stats = funnel_stats
        st.session_state.consensus_question = topic_text
        st.session_state.consensus_mode = analysis_mode
        st.rerun()

    analysis = st.session_state.get("consensus_analysis")
    papers = st.session_state.get("consensus_papers", [])
    search_response = st.session_state.get("consensus_search_response")
    funnel_stats = st.session_state.get("consensus_funnel_stats") or {}
    consensus_error = st.session_state.get("consensus_error")
    consensus_status = st.session_state.get("consensus_status")
    if consensus_error:
        st.error(f"共识分析失败：{consensus_error}")
        st.info("可能原因：AI 引擎限流或未配置。请在「⚙️ 设置」检查模型和 API Key。")
        return
    if analysis is None:
        if consensus_status == "analyzing":
            st.info("分析正在进行中，请稍候…如页面中断，请重新点击「开始分析」。")
        elif consensus_status in {"searching", "empty"}:
            st.info("上次运行未完成（可能因浏览器断连或等待超时）。请重新点击「开始分析」。")
        return

    if search_response is not None:
        diagnostics = format_search_diagnostics(search_response)
        if diagnostics["failed_sources"]:
            failed = "、".join(diagnostics["failed_sources"])
            st.warning(f"部分文献源受限或失败：{failed}。共识结果已由成功返回的来源自动保障。")

    if getattr(analysis, "stop_reason", ""):
        st.warning(f"证据抽取部分失败：{analysis.stop_reason}。以下内容为可用证据的确定性综述。")

    score = analysis.score
    spec = analysis.question_spec

    # === Consensus Meter 仪表盘（对标 consensus.app 原生视觉样式）===
    rc = score.raw_counts
    unclear_n = rc.get("uncertain", 0) + rc.get("not_applicable", 0)
    excluded_n = rc.get("excluded", 0)
    qualified_counts = qualified_stance_counts(
        analysis.evidence,
        domain=getattr(spec, "domain", "general") if spec else "general",
    )

    support_n = qualified_counts["support"]
    oppose_n = qualified_counts["oppose"]
    mixed_n = qualified_counts["mixed"]
    directional_n = qualified_counts["effective_independent_studies"]

    if spec and spec.question_type == "yes_no":
        if _should_render_consensus_meter(spec.question_type, score):
            total_d = max(1e-9, directional_n)
            yes_pct = round(100 * support_n / total_d)
            mixed_pct = round(100 * mixed_n / total_d)
            no_pct = round(100 * oppose_n / total_d)
            w_yes = max(6, yes_pct) if support_n > 0 else 0
            w_mixed = max(6, mixed_pct) if mixed_n > 0 else 0
            w_no = max(6, no_pct) if oppose_n > 0 else 0

            st.markdown(f"""
            <div class="cs-meter-card">
                <div class="cs-meter-header">
                    <div class="cs-meter-title">{analysis.question_spec.question}</div>
                    <div class="cs-meter-badge">N = {directional_n:g}</div>
                </div>
                <div class="cs-meter-bar-container">
                    {"<div class='cs-bar-yes' style='width: " + str(w_yes) + "%;'>" + f"{support_n:g}" + "</div>" if support_n > 0 else ""}
                    {"<div class='cs-bar-mixed' style='width: " + str(w_mixed) + "%;'>" + f"{mixed_n:g}" + "</div>" if mixed_n > 0 else ""}
                    {"<div class='cs-bar-no' style='width: " + str(w_no) + "%;'>" + f"{oppose_n:g}" + "</div>" if oppose_n > 0 else ""}
                </div>
                <div class="cs-meter-legend">
                    <div class="cs-legend-item">
                        <div class="cs-legend-dot" style="background:#3bcdaa;"></div>
                        <span>Yes:</span>
                        <span class="cs-legend-pct">{yes_pct}%</span>
                    </div>
                    <div class="cs-legend-item">
                        <div class="cs-legend-dot" style="background:#ff6900;"></div>
                        <span>Mixed:</span>
                        <span class="cs-legend-pct">{mixed_pct}%</span>
                    </div>
                    <div class="cs-legend-item">
                        <div class="cs-legend-dot" style="background:#ff6467;"></div>
                        <span>No:</span>
                        <span class="cs-legend-pct">{no_pct}%</span>
                    </div>
                    <div style="margin-left: auto; font-size: 0.78rem; color: #6b7280;">
                        共识方向：<strong>{score.direction}</strong> · 一致率 <strong>{score.directional_agreement:.0%}</strong>
                    </div>
                </div>
            </div>
            <div class="cs-figure-caption"><span class="cs-figure-tag">Figure 1</span> 共识仪表盘：针对核心研究假设的证据方向分布与一致率。</div>
            """, unsafe_allow_html=True)
        elif needs_search_expansion(score):
            st.warning(
                f"⚠️ 方向性证据不足（有效独立研究 {directional_n:g}/{MIN_DIRECTIONAL_STUDIES} 篇）："
                "已隐藏 Consensus Meter，避免以不足证据展示共识百分比。"
                "**建议扩充检索**——放宽关键词、增加同义词/上位概念，或扩大年份范围后重新分析。"
            )
        else:
            st.info(f"有效独立研究不足 {MIN_DIRECTIONAL_STUDIES} 篇，暂不展示 Consensus Meter。")


        # === Consensus Snapshot：各立场的质量概览（对标官方 Snapshot）===
        snapshot = getattr(analysis, "snapshot", None) or {}
        by_stance = snapshot.get("by_stance") or {}
        if by_stance:
            with st.expander("📊 Consensus Snapshot（各立场质量概览）", expanded=False):
                _stance_names = [
                    ("support", "Yes（支持）"),
                    ("oppose", "No（反对）"),
                    ("mixed", "Mixed（分歧）"),
                    ("uncertain", "Unclear（不明确）"),
                ]
                _rows = []
                for key, label in _stance_names:
                    s = by_stance.get(key) or {}
                    if not s.get("count"):
                        continue
                    designs = s.get("designs") or {}
                    design_txt = "、".join(
                        f"{STUDY_LABELS.get(k, k)}×{v}" for k, v in list(designs.items())[:4]
                    ) or "-"
                    _rows.append({
                        "立场": label,
                        "论文数": s.get("count", 0),
                        "平均年份": s.get("avg_year") or "-",
                        "研究设计构成": design_txt,
                        "高质量设计数": s.get("strong_design_count", 0),
                        "引用合计": s.get("citations_total", 0),
                        "期刊数": len(s.get("venues") or []),
                    })
                if _rows:
                    st.dataframe(_rows, use_container_width=True, hide_index=True)
                ov = snapshot.get("overall") or {}
                if ov:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("方向性证据", f"{ov.get('count', 0)} 篇")
                    c2.metric("平均发表年份", ov.get("avg_year") or "-")
                    c3.metric("引用合计", ov.get("citations_total", 0))
                    _venues = ov.get("venues") or []
                    if _venues:
                        st.caption("来源期刊：" + "、".join(_venues[:8])
                                   + ("…" if len(_venues) > 8 else ""))
                # 如实说明未接入的指标，避免用户误以为有 SJR
                for note in (snapshot.get("notes") or []):
                    st.caption(f"注：{note}")
    else:
        st.info(f"开放问题：不计算共识百分比，输出结构化证据综述（纳入 {directional_n} 篇方向性证据）")
        if needs_search_expansion(score):
            st.warning(
                f"⚠️ 仅纳入 {directional_n} 篇方向性证据，综述覆盖度有限。"
                "**建议扩充检索**后再生成深度综述。"
            )

        # === Evidence Review 主体（模式二）===
        # 开放问题的论文之间不是投票关系，因此**不产出任何百分比**，
        # 改为输出证据表 + 方法族归纳 + 主要分歧 + 研究空白。
        review = getattr(analysis, "evidence_review", None) or {}
        _table = review.get("evidence_table") or []

        if _table:
            st.markdown("#### 📋 证据表（逐篇方法、对象、结果与局限）")
            _rows = []
            for row in _table:
                _rows.append({
                    "论文": (row.get("title") or "")[:70],
                    "年份": row.get("year") or "-",
                    "研究设计": row.get("study_label") or row.get("study_type"),
                    "研究对象": (row.get("population") or "-")[:26],
                    "方法/干预": (row.get("intervention") or "-")[:26],
                    "结局": (row.get("outcome") or "-")[:22],
                    "方向": row.get("effect_direction") or "-",
                    "质量": row.get("quality") or "-",
                    "主要发现": (row.get("claim") or "-")[:80],
                })
            st.dataframe(_rows, use_container_width=True, hide_index=True)

        _designs = review.get("design_groups") or []
        if _designs:
            st.markdown("#### 🧪 方法族构成（按研究设计聚类）")
            _drows = [{
                "研究设计": d.get("study_label") or d.get("study_type"),
                "篇数": d.get("count", 0),
                "设计适配度": d.get("design_fit", 0),
            } for d in _designs]
            st.dataframe(_drows, use_container_width=True, hide_index=True)
            st.caption("设计适配度：该研究设计回答本类问题的适合程度（按问题域计算，非研究质量评分）")

        _conflicts = [d for d in (review.get("disagreements") or []) if d.get("is_conflicting")]
        if _conflicts:
            st.markdown("#### ⚔️ 主要分歧（同一结局上方向不一致）")
            for conflict in _conflicts:
                parts = "；".join(f"{k}×{len(v)}" for k, v in (conflict.get("directions") or {}).items())
                st.markdown(f"- **{conflict.get('outcome')}**：{parts}")

        _gaps = review.get("research_gaps") or []
        if _gaps:
            st.markdown("#### 🔍 研究空白")
            for gap in _gaps:
                st.markdown(f"- {gap}")

        for note in (review.get("notes") or []):
            st.caption(f"注：{note}")

    _render_subquestion_reviews(analysis)

    # === 方法学 Search Strategy 流程图卡片（对标 consensus.app Figure 2）===
    total_retrieved = funnel_stats.get("retrieved", len(papers))
    eligible_count = funnel_stats.get("with_abstract", len(papers))
    included_count = len(papers)
    domain_val = getattr(spec, "domain", "general")
    domain_name = {"cs_physics": "计算机/物理/工程", "biomedicine": "生物医药/临床", "general": "综合跨学科"}.get(domain_val, domain_val)

    st.markdown(f"""
    <div class="cs-strategy-box">
        <div class="cs-strategy-title">Search Strategy</div>
        <div class="cs-strategy-cards">
            <div class="cs-strategy-card">
                <div class="cs-strategy-num">{total_retrieved}</div>
                <div class="cs-strategy-label">Retrieved</div>
                <div class="cs-strategy-desc">多数据源（OpenAlex、Semantic Scholar 等）定向召回的所有候选论文总数。</div>
            </div>
            <div class="cs-strategy-card">
                <div class="cs-strategy-num">{eligible_count}</div>
                <div class="cs-strategy-label">Eligible</div>
                <div class="cs-strategy-desc">去重后具备完整学术摘要、符合实证类型门槛的合格论文。</div>
            </div>
            <div class="cs-strategy-card">
                <div class="cs-strategy-num">{included_count}</div>
                <div class="cs-strategy-label">Included</div>
                <div class="cs-strategy-desc">经 PICO 交叉匹配与实证信号精排后，最终纳入深度共识分析的核心文献。</div>
            </div>
        </div>
        <div class="cs-figure-caption"><span class="cs-figure-tag">Figure 2</span> 文献筛选流程示意图：从初筛到最终纳入的数量变化（所属领域：{domain_name}）。</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # === 深度综述报告（Consensus Deep Review 主体，注入引用徽章与高保真样式）===
    if analysis.narrative and analysis.narrative.strip():
        # 把叙述中的 [编号] 替换为 Consensus 风格的学术引用徽章
        # 建立编号映射表
        paper_lookup = {}
        citable_papers = _eligible_citation_papers(papers, analysis.evidence)
        for idx, p in enumerate(citable_papers, start=1):
            authors = p.get("authors") or []
            first_author = ""
            if authors and isinstance(authors, list):
                a0 = authors[0]
                first_author = a0.get("name", "") if isinstance(a0, dict) else str(a0)
            if not first_author:
                first_author = (p.get("title") or "Study").split()[0]
            # 提取英文姓氏或截取简写
            surname = first_author.split()[-1] if " " in first_author else first_author
            year_val = p.get("year") or ""
            link_url = p.get("url") or (f"https://doi.org/{p['doi']}" if p.get("doi") else "#")
            badge_html = (
                f'<a target="_blank" href="{link_url}" class="cs-cite-badge">'
                f'<span>{surname[:12]}</span>'
                f'<span class="cs-cite-year">{year_val}</span>'
                f'</a>'
            )
            paper_lookup[idx] = badge_html

        raw_narrative = analysis.narrative or ""

        # 安全处理：narrative 由 LLM 生成，**必须先整体转义**，再把我们自己
        # 构建的引用徽章 / GAP 高亮 HTML 注入回去。否则 unsafe_allow_html
        # 会原样渲染模型输出中的任意标签（XSS 风险）。
        import re as _re
        import html as _html
        import re

        def _escape_llm_html(text: str) -> str:
            """转义 LLM 文本中的 HTML 特殊字符，保留反引号代码块。"""
            return _html.escape(text, quote=False)

        safe_narrative = _escape_llm_html(raw_narrative)

        # 将 Markdown 中的 [1]、[2] 替换为精致徽章（徽章 HTML 由本地构建，可信）
        def replace_cite(match):
            try:
                num = int(match.group(1))
                return paper_lookup.get(num, match.group(0))
            except Exception:
                return match.group(0)

        formatted_narrative = re.sub(r'\[(\d+)\]', replace_cite, safe_narrative)

        # GAP 样式高亮替换
        formatted_narrative = formatted_narrative.replace(
            "| GAP |", '| <span class="cs-gap-cell">GAP</span> |'
        ).replace(
            " GAP ", ' <span class="cs-gap-cell">GAP</span> '
        )

        # 注意：必须传 unsafe_allow_html=True，否则上面生成的引用徽章 <a> 与
        # GAP 高亮 <span> 会被 Streamlit 转义成纯文本，直接暴露在报告正文里
        # （这是用户实际遇到的问题）。
        # 安全性由上面的 _escape_llm_html 保证：模型输出的标签已被转义，
        # 只有本地构建的徽章/高亮 HTML 会被真实渲染。
        st.markdown(formatted_narrative, unsafe_allow_html=True)

        # 渲染标准 APA 7th 参考文献列表（对标 Consensus 官网导出的 References 章节）
        try:
            import consensus
            apa_refs = consensus.ConsensusAnalyzer.format_apa_references(citable_papers)
            if apa_refs:
                st.markdown("### References")
                st.markdown(
                    "\n\n".join(f'<div style="font-size:0.85rem;line-height:1.6;margin-bottom:8px;padding-left:24px;text-indent:-24px;color:#374151;">{ref}</div>' for ref in apa_refs),
                    unsafe_allow_html=True
                )
        except Exception:
            pass
    else:
        st.warning("未生成综述正文。请查看下方证据审计表。")

    # === Figure 3: Results Timeline（对标 consensus.app 时间分布图）===
    try:
        import plotly.express as px
        import pandas as pd
        years = [p.get("year") for p in papers if p.get("year")]
        if years:
            timeline_df = pd.DataFrame({
                "year": [p.get("year") or 0 for p in papers],
                "citations": [p.get("citation_count") or 0 for p in papers],
                "title": [(p.get("title") or "")[:60] for p in papers],
            })
            fig = px.scatter(
                timeline_df, x="year", y="citations",
                size="citations", size_max=45, hover_name="title",
                color="year", color_continuous_scale="Plasma",
                labels={"year": "年份", "citations": "引用数", "title": "论文"},
            )
            fig.update_layout(height=340, showlegend=False,
                              title={"text": "Figure 3 · 主要成果按时间分布（标记越大代表被引用次数越多）",
                                     "x": 0, "xanchor": "left", "font": {"size": 13}})
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    # === Figure 4: Top Contributors（对标 consensus.app 高频作者/期刊表）===
    try:
        from collections import Counter
        author_counter = Counter()
        venue_counter = Counter()
        for p in papers:
            for a in (p.get("authors") or []):
                name = a.get("name", "") if isinstance(a, dict) else str(a)
                if name and name.strip():
                    author_counter[name.strip()] += 1
            if p.get("venue"):
                venue_counter[p["venue"]] += 1
        if author_counter or venue_counter:
            st.markdown("#### Figure 4 · Top Contributors")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**出现频率最高的作者**")
                top_authors = author_counter.most_common(8)
                if top_authors:
                    author_rows = "| 作者 | 论文数 |\n|---|---|\n" + "".join(
                        f"| {a} | {n} |\n" for a, n in top_authors
                    )
                    st.markdown(author_rows)
                else:
                    st.caption("无作者信息")
            with c2:
                st.markdown("**出现频率最高的期刊**")
                top_venues = venue_counter.most_common(8)
                if top_venues:
                    venue_rows = "| 期刊 | 论文数 |\n|---|---|\n" + "".join(
                        f"| {v} | {n} |\n" for v, n in top_venues
                    )
                    st.markdown(venue_rows)
                else:
                    st.caption("无期刊信息")
    except Exception:
        pass

    st.divider()

    # === 引用与证据核验（防止报告传播错误/编造文献）===
    verification = getattr(analysis, "verification", None) or {}
    if verification.get("total"):
        _v_total = verification.get("total", 0)
        _v_verified = verification.get("verified", 0)
        _v_warn = verification.get("plausible_with_warnings", 0)
        _v_excl = verification.get("excluded", 0)
        _gate = verification.get("gate") or {}

        with st.expander(
            f"🔎 引用核验（通过 {_v_verified}/{_v_total}）", expanded=not _gate.get("pass", True)
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("通过核验", _v_verified)
            c2.metric("有警告", _v_warn)
            c3.metric("已排除", _v_excl)
            c4.metric("可定位摘录", verification.get("traceable_excerpts", 0))

            _vw = verification.get("verification_warnings") or []
            if _vw:
                for w in _vw:
                    st.warning(w)

            _issues = _gate.get("issues") or []
            _recs = _gate.get("recommendations") or []
            if _issues:
                st.markdown("**质量门控**：未通过")
                for rec in _recs:
                    st.markdown(f"- {rec}")
            elif verification:
                st.success("核验通过：引用可追溯，未发现格式非法或缺失的引用。")

            # 逐条核验明细（只展示有问题的，避免噪声）
            _flagged = [r for r in (verification.get("records") or [])
                        if r.get("status") != "verified"]
            if _flagged:
                _labels = {
                    "invalid_doi_format": "DOI 格式非法",
                    "missing_url": "缺少链接",
                    "no_abstract": "无摘要依据",
                    "short_abstract": "摘要过短",
                    "untraceable_excerpt": "摘录无法定位",
                    "undetermined_stance": "立场不可判定",
                }
                _rows = [{
                    "论文": (r.get("title") or "")[:56],
                    "状态": r.get("status"),
                    "SciFact": r.get("scifact_label"),
                    "证据级别": r.get("evidence_basis"),
                    "问题": "、".join(_labels.get(i, i) for i in (r.get("issues") or [])),
                } for r in _flagged]
                st.dataframe(_rows, use_container_width=True, hide_index=True)
            else:
                st.caption("全部证据均通过核验。")

    st.divider()

    # === 附录：证据审计表 ===
    with st.expander("📋 逐篇证据审计表（附录）", expanded=False):
        rows = consensus_rows(analysis)
        if rows:
            header = "| 论文 | 纳入 | 立场 | 研究类型 | 质量 | 核心证据 | 限制 |"
            sep = "|------|------|------|----------|------|----------|------|"
            lines = [header, sep]
            for row in rows:
                cells = [
                    str(row.get("论文", ""))[:60],
                    str(row.get("纳入", "")),
                    str(row.get("立场", "")),
                    str(row.get("研究类型", "")),
                    str(row.get("质量", "")),
                    str(row.get("核心证据", ""))[:80],
                    str(row.get("限制", ""))[:40],
                ]
                cells = [c.replace("|", "/").replace("\n", " ") for c in cells]
                lines.append("| " + " | ".join(cells) + " |")
            st.markdown("\n".join(lines))
        else:
            st.info("无证据记录")
        if excluded_n:
            st.caption(f"已排除 {excluded_n} 篇（原因见上表「纳入=否」行）")

    # === 本地实体预筛排除的论文（透明审计）===
    _entity_excluded = st.session_state.get("consensus_entity_excluded") or []
    if _entity_excluded:
        with st.expander(
            f"🚫 本地实体预筛排除（{len(_entity_excluded)} 篇，未消耗 LLM 调用）",
            expanded=False,
        ):
            st.caption(
                "以下论文在**材料/器件**层面与问题不符，已在调用 LLM 前本地排除。"
                "此步骤不改变最终纳入集合（与后续精度筛选判定一致），仅节省抽取成本。"
            )
            for p in _entity_excluded[:50]:
                st.markdown(
                    f"- {(p.get('title') or '无标题')[:90]} "
                    f"<span style='color:#9ca3af'>({p.get('year') or 'n.d.'})</span>",
                    unsafe_allow_html=True,
                )
            if len(_entity_excluded) > 50:
                st.caption(f"…… 其余 {len(_entity_excluded) - 50} 篇略")

    st.divider()
    st.markdown(f"### 📚 纳入检索结果 ({len(papers)} 篇)")
    for i, paper in enumerate(papers):
        with st.expander(f"[{i+1}] {paper.get('title', '无标题')}"):
            st.caption((paper.get("abstract") or "无摘要")[:800])
            meta = []
            if paper.get("year"):
                meta.append(f"📅 {paper['year']}")
            if paper.get("citation_count"):
                meta.append(f"📊 引用 {paper['citation_count']}")
            if paper.get("sources"):
                meta.append("来源 " + ", ".join(paper["sources"]))
            if meta:
                st.caption(" · ".join(meta))


# ============================================================
# Compare Page
# ============================================================
def render_compare():
    st.markdown("## ⚖️ 多论文对比")

    sq = st.text_input("搜索要对比的论文", key="compare_search", placeholder="输入关键词...")

    if sq.strip():
        with st.spinner("搜索中..."):
            results = _cached_search(sq.strip(), limit=10)
            if results:
                for p in results:
                    pid = p.get("id", "")
                    selected_ids = [x.get("id") for x in st.session_state.compare_selected]
                    checked = pid in selected_ids
                    label = f"**{p.get('title', '无标题')}** ({p.get('year', '')})"
                    if st.checkbox(label, value=checked, key=f"cmp_{pid}"):
                        if not checked and len(st.session_state.compare_selected) < 4:
                            st.session_state.compare_selected.append(p)
                    else:
                        if checked:
                            st.session_state.compare_selected = [x for x in st.session_state.compare_selected if x.get("id") != pid]
            else:
                st.info("未找到相关论文")

    if st.session_state.compare_selected:
        n = len(st.session_state.compare_selected)
        st.markdown(f"### 已选择 ({n}/4)")
        if st.button("⚖️ 开始对比", type="primary", disabled=n < 2):
            with st.spinner("AI 对比分析中..."):
                papers_data = [{"title": p.get("title", ""),
                                "abstract": p.get("abstract", "")} for p in st.session_state.compare_selected]
                async def _compare():
                    async with get_llm_client() as c:
                        return await c.compare_papers(papers_data)
                comparison = asyncio.run(_compare())
            st.markdown("### 对比结果")
            st.markdown(comparison)
            if st.button("清除选择"):
                st.session_state.compare_selected = []
                st.rerun()


# ============================================================
# PDF Upload Page
# ============================================================
def render_pdf_upload():
    st.markdown("## 📄 PDF 上传与分析")

    uploaded_file = st.file_uploader("上传 PDF 文件", type=["pdf"], key="pdf_upload")

    if uploaded_file:
        file_content = uploaded_file.read()
        st.success(f"已上传: {uploaded_file.name} ({len(file_content) / 1024:.1f} KB)")

        if st.button("🤖 生成结构化摘要", type="primary"):
            with st.spinner("正在处理 PDF..."):
                processor = PDFProcessor()
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_content)
                    tmp_path = tmp.name
                try:
                    text = processor.extract_text(tmp_path)
                finally:
                    os.unlink(tmp_path)

            if not text:
                st.error("PDF 文本提取失败，可能是扫描版 PDF")
                return

            st.info(f"提取文本长度: {len(text)} 字符")

            with st.spinner("AI 生成结构化摘要..."):
                async def _summarize():
                    async with get_llm_client() as c:
                        return await c.summarize(text)
                summary = asyncio.run(_summarize())

            st.markdown("### 📋 结构化摘要")
            st.markdown(summary)

            with st.spinner("提取关键点..."):
                async def _keypoints():
                    async with get_llm_client() as c:
                        return await c.extract_key_points(text)
                key_points = asyncio.run(_keypoints())

            if key_points:
                st.markdown("### 🔑 关键发现")
                for kp in key_points:
                    category = kp.get("category", "general")
                    icon = {"method": "🔧", "result": "📊", "conclusion": "💡"}.get(category, "📌")
                    st.markdown(f"{icon} **{category}**: {kp.get('content', '')}")

            st.divider()
            st.markdown("### 💬 向 AI 提问")
            if "pdf_chat" not in st.session_state:
                st.session_state.pdf_chat = []

            for msg in st.session_state.pdf_chat:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_msg = st.chat_input("关于这篇论文你想问什么？", key="pdf_chat_input")
            if user_msg:
                st.session_state.pdf_chat.append({"role": "user", "content": user_msg})
                with st.spinner("AI 思考中..."):
                    async def _chat():
                        async with get_llm_client() as c:
                            return await c.chat(st.session_state.pdf_chat, text[:6000])
                    reply = asyncio.run(_chat())
                st.session_state.pdf_chat.append({"role": "assistant", "content": reply})
                st.rerun()


# ============================================================
# Notes Page
# ============================================================
def render_notes():
    st.markdown("## 📝 我的笔记")

    notes = get_all_notes()
    if not notes:
        st.info("暂无笔记。在论文详情页可以添加笔记。")
        return

    for note in notes:
        with st.expander(f"📄 {note['paper_title']}", expanded=False):
            st.caption(f"更新于: {note['updated_at'][:19]}")
            st.markdown(note["content"][:500] if note["content"] else "空笔记")
            if st.button("查看论文", key=f"note_view_{note['paper_id']}"):
                st.session_state.page = "search"
                st.rerun()


# ============================================================
# Tags Page
# ============================================================
def render_tags():
    st.markdown("## 🏷️ 标签管理")

    # Create tag
    with st.form("create_tag_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            tag_name = st.text_input("标签名称", key="new_tag_name", placeholder="例如：HBT、GaAs、我的研究方向")
        with c2:
            tag_color = st.color_picker("颜色", value="#4361ee", key="new_tag_color")
        with c3:
            st.form_submit_button("➕ 创建标签")

    if tag_name and tag_name.strip():
        create_tag(tag_name.strip(), tag_color)
        st.toast(f"✅ 标签「{tag_name.strip()}」已创建")

    st.divider()

    tags = get_all_tags()
    if not tags:
        st.info("暂无标签")
        return

    # Display tags as colored badges
    st.markdown("### 现有标签")
    for tag in tags:
        c1, c2, c3 = st.columns([4, 2, 1])
        with c1:
            st.markdown(f'<span style="background:{tag["color"]};color:white;padding:4px 12px;border-radius:12px;font-size:0.9rem">{tag["name"]}</span>', unsafe_allow_html=True)
        with c2:
            st.caption(f"{tag['papers_count']} 篇论文")
        with c3:
            if st.button("🗑️", key=f"del_tag_{tag['id']}"):
                delete_tag(tag["id"])
                st.rerun()


# ============================================================
# Canvas Page (Visual Board)
# ============================================================
def render_canvas():
    st.markdown("## 🎨 可视化画布")
    st.caption("自由拖拽论文卡片，按主题组织你的研究思路")

    # Get all papers from collections
    collections = get_all_collections()
    if not collections:
        st.info("暂无文献库。请先创建文献库并添加论文。")
        return

    # Select collection to visualize
    col_names = {c["id"]: c["name"] for c in collections}
    selected_col = st.selectbox("选择文献库", list(col_names.keys()),
                                format_func=lambda x: col_names[x], key="canvas_col")

    if selected_col:
        detail = get_collection_detail(selected_col)
        if not detail or not detail["papers"]:
            st.info("该文献库暂无论文")
            return

        papers = detail["papers"]
        st.markdown(f"### 📁 {detail['name']} ({len(papers)} 篇)")

        # PCA topic visualization
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import PCA
            titles = [p["title"] or "" for p in papers]
            vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
            X = vectorizer.fit_transform(titles)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X.toarray())
            import plotly.express as px
            import pandas as pd
            df = pd.DataFrame({
                "x": coords[:, 0], "y": coords[:, 1],
                "title": [p["title"][:40] for p in papers],
                "year": [p.get("year", 0) for p in papers],
                "citations": [p.get("citation_count", 0) or 0 for p in papers],
            })
            fig = px.scatter(df, x="x", y="y", hover_name="title",
                             hover_data=["year", "citations"],
                             size="citations", size_max=30, color="year",
                             color_continuous_scale="Plasma",
                             title="📊 论文学术分布图（位置=相似度 · 颜色=年份 · 大小=引用数）")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

        # Visual grid layout with paper cards
        show_tags = st.checkbox("显示标签", True, key="canvas_show_tags")
        cols_per_row = 4
        for i in range(0, len(papers), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(papers):
                    p = papers[idx]
                    with col:
                        with st.container(border=True):
                            st.markdown(f"**{p.get('title', '无标题')[:60]}**")
                            if p.get("year"):
                                st.caption(f"📅 {p['year']}")
                            if p.get("citation_count"):
                                st.caption(f"📊 引用 {p['citation_count']}")
                            abstract = p.get("abstract") or ""
                            if abstract:
                                st.caption(abstract[:100] + "...")
                            # Tags
                            paper_tags = get_paper_tags(p["id"])
                            if paper_tags:
                                tag_html = " ".join(
                                    f'<span style="background:{t["color"]};color:white;padding:2px 8px;border-radius:8px;font-size:0.75rem">{t["name"]}</span>'
                                    for t in paper_tags
                                )
                                st.markdown(tag_html, unsafe_allow_html=True)

        # Add tags to papers
        st.divider()
        st.markdown("### 🏷️ 给论文打标签")
        all_tags = get_all_tags()
        if all_tags and papers:
            paper_titles = {p["id"]: p["title"][:50] for p in papers}
            c1, c2, c3 = st.columns([3, 2, 1])
            with c1:
                sel_paper = st.selectbox("选择论文", list(paper_titles.keys()),
                                         format_func=lambda x: paper_titles[x], key="canvas_paper")
            with c2:
                sel_tag = st.selectbox("选择标签", [t["id"] for t in all_tags],
                                       format_func=lambda x: next(t["name"] for t in all_tags if t["id"] == x), key="canvas_tag")
            with c3:
                if st.button("➕ 添加"):
                    if sel_paper is not None and sel_tag is not None:
                        add_tag_to_paper(sel_paper, sel_tag)
                        st.toast("✅ 标签已添加")
                        st.rerun()


# ============================================================
# Settings Page
# ============================================================
def render_settings():
    st.markdown("## ⚙️ 设置")

    # Initialize session state for LLM settings
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = DEFAULT_LLM_PROVIDER
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = DEFAULT_LLM_MODEL
    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""

    st.markdown("### 🤖 大模型配置")

    # Provider selection
    provider_options = {k: v["name"] for k, v in LLM_PROVIDERS.items()}
    selected_provider = st.selectbox(
        "选择模型提供商",
        list(provider_options.keys()),
        format_func=lambda k: provider_options[k],
        index=list(provider_options.keys()).index(st.session_state.llm_provider),
        key="settings_provider"
    )
    st.session_state.llm_provider = selected_provider

    # Model selection
    provider_cfg = LLM_PROVIDERS[selected_provider]
    model_options = provider_cfg["models"]
    default_model = provider_cfg["default_model"]

    if st.session_state.llm_model not in model_options:
        st.session_state.llm_model = default_model

    selected_model = st.selectbox(
        "选择模型",
        model_options,
        index=model_options.index(st.session_state.llm_model) if st.session_state.llm_model in model_options else 0,
        key="settings_model"
    )
    st.session_state.llm_model = selected_model

    # API Key
    api_key_label = f"API Key ({provider_cfg['name']})"
    if selected_provider == "ollama":
        st.info("Ollama 是本地模型，无需 API Key。请确保 Ollama 服务已启动（默认端口 11434）。")
    else:
        api_key = st.text_input(
            api_key_label,
            value=st.session_state.llm_api_key,
            type="password",
            key="settings_api_key",
            placeholder=f"输入 {provider_cfg['name']} API Key"
        )
        st.session_state.llm_api_key = api_key

    # Test connection
    st.divider()
    if st.button("🧪 测试连接", type="primary"):
        with st.spinner("测试中..."):
            async def _test():
                async with LLMClient(
                    provider=st.session_state.llm_provider,
                    api_key=st.session_state.llm_api_key,
                    model=st.session_state.llm_model
                ) as c:
                    return await c._complete([{"role": "user", "content": "Hello, respond with 'OK' only."}], temperature=0.1)
            result = asyncio.run(_test())
        if "OK" in result.upper() or len(result) < 20:
            st.success(f"✅ 连接成功！模型响应: {result[:50]}")
        elif "AI 调用失败" in result:
            st.error(f"❌ 连接失败: {result}")
        else:
            st.success(f"✅ 连接成功！模型响应: {result[:100]}")

    # Current configuration summary
    st.divider()
    st.markdown("### 📋 当前配置")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**提供商**: {provider_cfg['name']}")
        st.markdown(f"**模型**: {st.session_state.llm_model}")
    with col2:
        # Check if API key is available from secrets
        has_secrets_key = False
        try:
            secrets_key = llm_secret_name(selected_provider)
            has_secrets_key = bool(st.secrets.get(secrets_key, "")) if secrets_key else False
        except Exception:
            pass

        if st.session_state.llm_api_key:
            key_display = "✅ 已配置（页面输入）"
        elif has_secrets_key:
            key_display = "✅ 已配置（Streamlit Secrets）"
        elif selected_provider == "ollama":
            key_display = "本地模型"
        else:
            key_display = "❌ 未配置"
        st.markdown(f"**API Key**: {key_display}")
        st.markdown(f"**Base URL**: {provider_cfg['base_url']}")

    # Streamlit Cloud secrets configuration
    st.divider()
    st.markdown("### ☁️ Streamlit Cloud 永久配置")
    st.markdown("""
    如果部署到 Streamlit Cloud，可以通过 **Secrets** 永久保存 API Key：

    1. 在 Streamlit Cloud 应用页面，点击右上角 **⋮** → **Settings** → **Secrets**
    2. 添加以下内容：

    ```toml
    ZHIPUAI_API_KEY = "xxx"
    OPENAI_API_KEY = "sk-xxx"
    ANTHROPIC_API_KEY = "sk-xxx"
    DASHSCOPE_API_KEY = "sk-xxx"
    DEEPSEEK_API_KEY = "sk-xxx"
    ```

    3. 保存后，API Key 将永久生效，无需每次输入。
    """)

    # Environment variables help
    st.markdown("### 🔑 本地环境变量（可选）")
    st.markdown("""
    也可以通过 `.env` 文件配置 API Key，优先级：**页面输入 > Streamlit Secrets > 环境变量**

    ```env
    ZHIPUAI_API_KEY=xxx
    OPENAI_API_KEY=sk-xxx
    ANTHROPIC_API_KEY=sk-xxx
    DEEPSEEK_API_KEY=sk-xxx
    ```
    """)


# ============================================================
# Page Router (must come after all function definitions)
# ============================================================
page = st.session_state.page
if page == "home":
    render_home()
elif page == "search":
    render_search()
elif page == "collections":
    render_collections()
elif page == "consensus":
    render_consensus()
elif page == "compare":
    render_compare()
elif page == "pdf":
    render_pdf_upload()
elif page == "notes":
    render_notes()
elif page == "tags":
    render_tags()
elif page == "canvas":
    render_canvas()
elif page == "settings":
    render_settings()


if __name__ == "__main__":
    init_db()
