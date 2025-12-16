from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool

from rag.pipeline import RAGPipeline


def _format_docs_for_context(docs: List[Document], max_chars: int = 6000) -> str:
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "")
        page = meta.get("page", "")
        content = (d.page_content or "").strip()
        parts.append(f"[{i}] source={source} page={page}\n{content}")
    text = "\n\n".join(parts)
    return text[:max_chars]


def make_retrieval_tool(pipeline: RAGPipeline, *, k: int = 5, max_chars: int = 6000):
    """
    生成一个可被 Agent 调用的检索 Tool。
    Tool 的返回值是可直接塞进 prompt 的“带引用上下文”字符串。
    """

    @tool("rag_retrieve", return_direct=False)
    def rag_retrieve(query: str) -> str:
        """从本地 Chroma 向量库检索相关片段，返回带 source/page 的上下文。"""
        docs = pipeline.retrieve_with_synonyms(query, k=k)
        if not docs:
            return "（未检索到相关片段）"
        return _format_docs_for_context(docs, max_chars=max_chars)

    return rag_retrieve

