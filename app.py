import asyncio
import streamlit as st
from engine import (
    init_db, save_paper_to_db, add_to_collection, remove_from_collection,
    get_all_collections, get_collection_detail, create_collection, delete_collection, export_bibtex,
    save_note, get_note, delete_note, get_all_notes,
    create_tag, get_all_tags, delete_tag, add_tag_to_paper, remove_tag_from_paper, get_paper_tags,
    OpenAlexClient, LLMClient, DeepSeekClient, PDFProcessor, SearchEngine,
    SEMANTIC_SCHOLAR_API_KEY, DEEPSEEK_API_KEY, LLM_PROVIDERS,
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State
# ============================================================
defaults = {
    "page": "home", "search_query": "", "search_results": [],
    "show_summary": "", "show_chat": "", "show_graph": "", "show_similar": "", "show_note": "",
    "compare_selected": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def navigate(page: str):
    st.session_state.page = page
    st.rerun()


def get_llm_client():
    """Create LLMClient with current session settings (secrets > session state > env)."""
    provider = st.session_state.get("llm_provider", "deepseek")
    model = st.session_state.get("llm_model", "")
    api_key = st.session_state.get("llm_api_key", "")

    # Priority: session state > st.secrets > env var
    if not api_key:
        try:
            secrets_key = f"{provider.upper()}_API_KEY"
            api_key = st.secrets.get(secrets_key, "")
        except Exception:
            pass

    return LLMClient(provider=provider, api_key=api_key, model=model)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 ResearchPaperHub")
    st.caption("v4.0 — 智能学术研究助手")
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
@st.cache_data(ttl=300, show_spinner=False)
def _cached_search(query: str, limit: int = 100, sort: str = "relevance") -> list[dict]:
    async def _s():
        engine = SearchEngine()
        return await engine.search(query, limit, source="auto", sort=sort)
    return asyncio.run(_s())


def render_search():
    st.markdown("## 🔍 论文搜索")

    c1, c2 = st.columns([5, 1])
    with c1:
        q = st.text_input("搜索关键词", value=st.session_state.get("search_query", ""),
                          key="search_input", placeholder="输入关键词搜索论文...")
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

        with st.spinner("搜索中..."):
            results = _cached_search(st.session_state.search_query, limit=200,
                                     sort=st.session_state.search_sort)
            # 缓存可能返回空（旧的失败结果），直接搜索一次作为兜底
            if not results:
                engine = SearchEngine()
                results = asyncio.run(engine.search(st.session_state.search_query, limit=200, source="auto", sort=st.session_state.search_sort))
            st.session_state.search_results = results

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
        else:
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
def render_consensus():
    st.markdown("## 📊 共识分析")
    st.caption("输入研究主题，分析相关论文的共识程度")

    topic = st.text_area("研究主题或问题", placeholder="例如：Does BERT outperform LSTM in sequence classification tasks?",
                         key="consensus_topic")

    if st.button("🔬 开始分析", type="primary", disabled=not topic.strip()):
        topic_text = topic.strip()
        candidates = []
        all_rounds = 0
        max_rounds = 2  # 搜2轮共400篇

        # 先搜完所有轮次，再统一过滤
        search_tasks = []
        with st.spinner("搜索相关论文..."):
            for rnd in range(max_rounds):
                engine = SearchEngine()
                round_results = asyncio.run(engine.search(topic_text, limit=200, source="auto"))
                if round_results:
                    candidates.extend(round_results)
                    all_rounds += 1
                    st.toast(f"第{rnd+1}轮搜索：{len(round_results)}篇")

        if not candidates:
            st.warning("未找到相关论文")
            return

        # AI 领域过滤：大批次一次过滤（100篇/组，更快）
        with st.spinner(f"AI 筛选领域相关论文（共{len(candidates)}篇）..."):
            batch_size = 100
            relevant_indices = set()
            for batch_start in range(0, len(candidates), batch_size):
                batch = candidates[batch_start:batch_start + batch_size]
                titles_list = "\n".join(
                    f"[{i}] {p.get('title', '')[:100]}"
                    for i, p in enumerate(batch)
                )
                async def _filter(blist=titles_list):
                    async with get_llm_client() as c:
                        prompt = (
                            f"研究主题：{topic_text}\n\n"
                            f"从以下论文标题中，找出与上述主题**相关**的论文。只要内容大致相关就保留，"
                            f"只有明显不相关的研究对象才排除。只返回相关论文的编号，JSON数组如[0,3,5]。\n\n{blist}"
                        )
                        result = (await c._complete([
                            {"role": "system", "content": "只输出JSON数组如[0,3,5]，不要其他内容。"},
                            {"role": "user", "content": prompt},
                        ], temperature=0.0)).strip()
                        import json, re
                        try:
                            return set(int(n) for n in re.findall(r'\d+', result))
                        except Exception:
                            return set()
                try:
                    batch_relevant = asyncio.run(_filter())
                    relevant_indices.update(batch_relevant)
                except Exception:
                    relevant_indices.update(range(len(batch)))

            filtered = [p for i, p in enumerate(candidates) if i in relevant_indices]
            n_filtered = len(filtered)
            st.toast(f"AI筛选：{len(candidates)}篇 → {n_filtered}篇相关")

        if not filtered:
            st.warning("未找到领域相关的论文")
            return

        candidates = filtered

        # TF-IDF 语义排序 + 取前30篇
        with st.spinner("计算语义相关性..."):
            texts = [topic_text] + [f"{p.get('title', '')} {(p.get('abstract') or '')[:500]}" for p in candidates]
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
                tfidf = vectorizer.fit_transform(texts)
                sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
            except ImportError:
                t_terms = set(topic_text.lower().split())
                sims = [
                    sum(1 for term in t_terms if term in f"{p.get('title','')} {(p.get('abstract') or '')}".lower())
                    / max(1, len(t_terms))
                    for p in candidates
                ]
            for i, p in enumerate(candidates):
                p["_sim"] = float(sims[i]) if i < len(sims) else 0
            candidates.sort(key=lambda p: p.get("_sim", 0), reverse=True)

        papers = candidates[:30]
        st.info(f"共搜索 {all_rounds} 轮，从 {len(candidates)} 篇相关论文中选取了 {len(papers)} 篇")

        with st.spinner("AI 共识分析中..."):
            paper_list = "\n".join(f"[{i+1}] {p.get('title','')} ({p.get('year','')})" for i, p in enumerate(papers))
            paper_texts = "\n\n".join(f"[{i+1}] {p.get('title','')}\n{(p.get('abstract') or '')[:800]}" for i, p in enumerate(papers))
            async def _analyze():
                async with get_llm_client() as c:
                    prompt = (
                        f"你是一个学术研究分析专家。请对以下 {len(papers)} 篇关于「{topic_text}」的论文进行深度共识分析。\n\n"
                        "请按以下格式输出完整报告：\n\n"
                        "## 1. 共识度评分\n"
                        "给出0-100的共识度评分，说明评分依据。\n\n"
                        "## 2. 研究方法分布\n"
                        "统计这组论文采用的研究方法（实验/理论/模拟/综述等），评估整体证据质量（高/中/低）。\n\n"
                        "## 3. 主要一致观点\n"
                        "列出3-5个学术界一致认同的发现，每点标注支持论文编号。\n\n"
                        "## 4. 主要争议与分歧\n"
                        "列出存在分歧的观点，说明各方立场及论文编号，分析分歧原因（方法差异/数据差异/理论框架差异）。\n\n"
                        "## 5. 研究空白\n"
                        "基于现有论文，识别2-3个尚未充分研究的问题或方向。\n\n"
                        "## 6. 方法论评估\n"
                        "对比各论文的研究方法优劣，指出可能的方法学局限（样本量、实验设计、统计方法等）。\n\n"
                        "## 7. 统计证据强度\n"
                        "评估关键结论的统计证据强度（效应量、显著性、可重复性），标注需要更多验证的结论。\n\n"
                        "## 8. 论文立场标注\n"
                        "按以下格式标注每篇论文在核心议题上的立场：\n"
                        "[1] 论文标题 — ✅支持/❌反对/➖中立 — 关键论据\n"
                        "[2] ...\n\n"
                        f"论文列表：\n{paper_list}\n\n"
                        f"论文详情：\n{paper_texts}"
                    )
                    return await c._complete([
                        {"role": "system", "content": "你是专业的学术共识分析专家。请生成详细、有深度的分析报告，每项声明都必须引用论文编号。"},
                        {"role": "user", "content": prompt},
                    ])
            analysis = asyncio.run(_analyze())

        # Extract score (take the LAST number on the consensus line, not the section number)
        import re
        score = 50
        for line in analysis.split("\n"):
            if "共识度评分" in line or ("共识度" in line and re.search(r'\d+', line)):
                nums = re.findall(r'\d+', line)
                if nums:
                    score = min(100, max(0, int(nums[-1])))  # LAST number, not first
                    break

        st.markdown("---")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{score}%</div><div class="stat-label">共识度</div></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown(analysis)

        st.divider()
        st.markdown(f"### 相关论文 ({len(papers)} 篇)")
        for i, p in enumerate(papers[:30]):
            with st.expander(f"[{i+1}] {p.get('title', '无标题')}"):
                st.caption((p.get("abstract") or "")[:500])
                meta = []
                if p.get("year"): meta.append(f"📅 {p['year']}")
                if p.get("citation_count"): meta.append(f"📊 引用 {p['citation_count']}")
                if meta: st.caption(" · ".join(meta))


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
        st.session_state.llm_provider = "deepseek"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = ""
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
            secrets_key = f"{selected_provider.upper()}_API_KEY"
            has_secrets_key = bool(st.secrets.get(secrets_key, ""))
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
    DEEPSEEK_API_KEY = "sk-xxx"
    OPENAI_API_KEY = "sk-xxx"
    ANTHROPIC_API_KEY = "sk-xxx"
    DASHSCOPE_API_KEY = "sk-xxx"
    ZHIPUAI_API_KEY = "xxx"
    ```

    3. 保存后，API Key 将永久生效，无需每次输入。
    """)

    # Environment variables help
    st.markdown("### 🔑 本地环境变量（可选）")
    st.markdown("""
    也可以通过 `.env` 文件配置 API Key，优先级：**页面输入 > Streamlit Secrets > 环境变量**

    ```env
    DEEPSEEK_API_KEY=sk-xxx
    OPENAI_API_KEY=sk-xxx
    ANTHROPIC_API_KEY=sk-xxx
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
