# ResearchPaperHub v4.0

智能学术研究助手 — 论文发现 · AI 阅读 · 共识分析

## 功能特点

- 🔍 **论文搜索**: OpenAlex + arXiv 双数据源，支持中英文搜索
- 📊 **文献分析**: 突变检测、聚类分析、时间线可视化
- 🤖 **AI 摘要**: 结构化论文摘要、关键点提取
- 💬 **论文对话**: 基于论文内容的 AI 问答
- 📈 **共识分析**: 多论文共识度评估，带来源引文
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

- DeepSeek（默认）
- OpenAI
- Claude
- 通义千问
- 智谱清言
- Ollama（本地模型）

### 环境变量（可选）

创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-xxx
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-xxx
```

## 技术栈

- **前端**: Streamlit
- **后端**: Python 3.10+
- **数据库**: SQLite
- **AI**: DeepSeek / OpenAI / Claude
- **数据源**: OpenAlex, arXiv

## 许可证

MIT License
