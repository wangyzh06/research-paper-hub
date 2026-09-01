# ResearchPaperHub v4.0

智能学术研究助手 — 论文发现 · AI 阅读 · 共识分析

## 功能特点

- 🔍 **论文搜索**: OpenAlex、arXiv、Crossref、PubMed，以及可选的 Semantic Scholar；支持中英文查询扩展、多源融合、翻页和失败诊断
- 📊 **文献分析**: 突变检测、聚类分析、时间线可视化
- 🤖 **AI 摘要**: 结构化论文摘要、关键点提取
- 💬 **论文对话**: 基于论文内容的 AI 问答
- 📈 **共识分析**: GLM 逐篇抽取结构化证据，使用确定性权重计算可审计的共识度
- 📚 **文献库管理**: 收藏、标签、笔记
- ⚖️ **多论文对比**: AI 驱动的论文对比分析
- 🎨 **可视化画布**: 论文卡片可视化组织

## 快速开始

### 方案 A: 直接运行（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动应用
streamlit run app.py
```

### 方案 B: Docker

```bash
docker build -t research-paper-hub .
docker run -p 8501:8501 research-paper-hub
```

### 方案 C: Windows 安装包

1. 双击 `install.bat`
2. 按提示完成安装
3. 双击 `start_app.bat` 启动

## 配置

### 大模型配置

在应用内「⚙️ 设置」页面配置：

- 智谱 GLM（默认模型 `glm-4.7-flash`）
- OpenAI
- Claude
- 通义千问
- DeepSeek（可选）
- Ollama（本地模型）

### 环境变量（可选）

创建 `.env` 文件：

```env
ZHIPUAI_API_KEY=your_zhipu_api_key
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-xxx
SEMANTIC_SCHOLAR_API_KEY=your_s2_api_key
OPENALEX_EMAIL=you@example.com
```

也可以在应用的「⚙️ 设置」页面输入 API Key。配置优先级为：页面输入、Streamlit Secrets、环境变量。GLM 的 Secrets/环境变量名称均为 `ZHIPUAI_API_KEY`。

### 搜索与共识结果说明

- 检索结果先在全部来源和查询变体中合并，再按 DOI 或规范化题名去重和排序，不会在单个来源占满候选后提前截断。
- 页面会区分“无匹配结果”“部分来源失败”和“全部来源失败”；认证失败不会写入搜索缓存，可使用“重试”按钮刷新当前查询。
- 共识分不是从大模型报告中解析出来的数字。GLM 只抽取研究立场、研究类型、质量、直接性、置信度和可追溯摘要片段，程序再按固定公式计算方向一致率与证据确定性。
- 共识分析仅基于题名与摘要，不能确认全文中未报告的样本量、效应量、P 值、置信区间或偏倚风险，不能替代系统综述或 Meta 分析。

## 技术栈

- **前端**: Streamlit
- **后端**: Python 3.10+
- **数据库**: SQLite
- **AI**: GLM（默认）/ OpenAI / Claude / 通义千问 / DeepSeek / Ollama
- **数据源**: OpenAlex, arXiv, Crossref, PubMed, Semantic Scholar（可选）

## 许可证

MIT License
