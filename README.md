# RAG-pdf
RAG 智能 PDF 阅读助手（Streamlit + LangChain + Chroma）

## 功能
- 本地 PDF 向量化入库（Chroma）
- Agent 调用检索工具 `rag_retrieve` 后再回答

## 快速开始

### 1) 准备环境
建议使用 Python 3.10/3.11（部分依赖在 3.13 上可能不完整）。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 配置 `.env`
项目使用环境变量读取模型配置；请从示例文件创建你的本地配置：

```bash
cp .env.example .env
```

然后编辑 `.env`，至少需要：
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`（DeepSeek 的 OpenAI 兼容接口地址）
- `EMBEDDING_MODEL`（如 `sentence-transformers/all-mpnet-base-v2`）

如网络访问 HuggingFace 较慢，可在 `.env` 中配置镜像：
`HF_ENDPOINT=https://hf-mirror.com`

### 3) 放入 PDF 文档
把你的 PDF 放到 `user_data/documents/`（仓库默认带空目录占位，不会提交你的文档）。

### 4) 运行
```bash
streamlit run streamlit_app.py
```

首次运行如果需要建库，会下载 embedding 模型并生成 `chroma_langchain_db/`（本地持久化向量库）。
