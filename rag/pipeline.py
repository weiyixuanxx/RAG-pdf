from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import json

# 允许两种运行方式：
# 1) 推荐：在项目根目录执行 `python -m rag.pipeline`
# 2) 兼容：直接执行 `python rag/pipeline.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag.data_preparation import DocumentIngestor
from rag.embeddings import EmbeddingProvider
from rag.vector_store import VectorStoreManager

from core.config import Config
from core.llm import Model

class RAGPipeline:
    """
    负责串联数据准备、向量库、检索与回答。
    当前 answer() 仅返回检索结果；在 TODO 中接入 LLM 生成。
    """

    def __init__(
        self,
        ingestor: Optional[DocumentIngestor] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStoreManager] = None,
        llm: Optional[Model] = None,
    ) -> None:
        self.ingestor = ingestor or DocumentIngestor()
        self.embedding_provider = embedding_provider or EmbeddingProvider()
        self.vector_store = vector_store or VectorStoreManager(
            embedding_function=self.embedding_provider.get()
        )
        if llm is not None:
            self.llm = llm
        else:
            config = Config()
            model = Model(api_key=config.get_api_key(),base_url=config.get_base_url())
            model.set_model_name(config.get_chat_model())
            self.llm = model.create()
        
    def build_index(
        self,
        paths: Iterable[str],
        reset: bool = False,
        skip_if_exists: bool = True,
    ) -> None:
        """加载 -> 切分 -> 写入向量库。"""
        if reset:
            self.vector_store.reset_collection()
        elif skip_if_exists and self.vector_store.count() > 0:
            print(f"检测到向量库已有 {self.vector_store.count()} 条记录，跳过重建。")
            return
        chunks = self.ingestor.ingest(paths)
        if not chunks:
            print("没有可写入的文档。")
            return
        ids = self.vector_store.add_documents(chunks)
        print(f"已写入向量库：{len(ids)} 条。")
        
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """仅做相似度检索。"""
        return self.vector_store.similarity_search(query, k=k)

    def retrieve_with_synonyms(self, query: str, k: int = 3) -> List[Document]:
        """
        针对特定术语做“同义词/别名”扩展检索，并合并去重。
        你目前关心：混淆电路 / Garbled Circuit / 姚氏混淆电路 / Yao
        """
        queries: List[str] = [query]
        if "混淆电路" in query:
            queries.extend(["Garbled Circuit", "姚氏混淆电路", "Yao"])

        seen: set[tuple] = set()
        merged: List[Document] = []
        for q in queries:
            for d in self.retrieve(q, k=k):
                meta = d.metadata or {}
                key = (
                    meta.get("source"),
                    meta.get("page"),
                    meta.get("start_index"),
                    (d.page_content or "")[:80],
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)
        return merged
    
    def answer(self, query: str, k: int = 3) -> str:
        """将检索结果送入 LLM，生成最终答案。"""
        docs = self.retrieve_with_synonyms(query, k=k)
        if not docs:
            return "我没有检索到相关文档，所以无法回答该问题。请看你的初始设置是否正确。"
        context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
        prompt = ChatPromptTemplate.from_template(
            "你是知识助手。请依据检索到的内容作答，中文回答。\n"
            "检索内容：\n{context}\n\n问题：{question}"
        )
        chain = prompt | self.llm
        msg = chain.invoke({"context": context, "question": query})
        return getattr(msg, "content", str(msg))
    
    def answer_iterative(
        self,
        question: str,
        *,
        k: int = 3,
        max_rounds: int = 3,
        max_context_chars: int = 6000,
    ) -> str:
        """
        你描述的“多次检索”：
        1) 先用原问题检索；
        2) LLM 判断是否信息不足；
        3) 若不足，产出 next_query（例如“基于混淆电路的协议”）继续检索；
        4) 直到能回答或达到 max_rounds。
        """

        def _format_docs(docs: List[Document]) -> str:
            parts: List[str] = []
            for i, d in enumerate(docs, start=1):
                meta = d.metadata or {}
                src = meta.get("source", "")
                page = meta.get("page", "")
                snippet = d.page_content[:1200]
                parts.append(f"[{i}] source={src} page={page}\n{snippet}")
            text = "\n\n".join(parts)
            return text[:max_context_chars]

        decide_prompt = ChatPromptTemplate.from_template(
            "你是一个严谨的 RAG 助手，负责决定是否需要继续检索。\n"
            "你必须只输出 JSON（不要 Markdown），格式如下：\n"
            '{{"status":"final"|"need_more","next_query":"...","answer":"..."}}\n\n'
            "规则：\n"
            "- 如果当前检索内容已足够给出清晰定义/解释，则 status=final，并在 answer 给出最终回答。\n"
            "- 如果内容不足，但提到了可能相关的关键词/短语，则 status=need_more，next_query 给出下一次更具体的检索词。\n"
            "- next_query 必须是简短的中文检索短语（<=20字），不要带引号。\n\n"
            "用户问题：{question}\n\n"
            "当前检索内容：\n{context}\n"
        )

        current_query = question
        seen_queries: set[str] = set()
        accumulated_docs: List[Document] = []

        for _round in range(1, max_rounds + 1):
            if current_query in seen_queries:
                break
            seen_queries.add(current_query)

            docs = self.retrieve_with_synonyms(current_query, k=k)
            if docs:
                accumulated_docs.extend(docs)

            context = _format_docs(accumulated_docs)
            decision_msg = (decide_prompt | self.llm).invoke(
                {"question": question, "context": context}
            )
            decision_text = getattr(decision_msg, "content", str(decision_msg)).strip()

            try:
                decision = json.loads(decision_text)
            except Exception:
                # 解析失败则兜底回答一次，避免死循环
                fallback_prompt = ChatPromptTemplate.from_template(
                    "请依据以下检索内容回答问题，中文简洁回答。\n\n"
                    "检索内容：\n{context}\n\n问题：{question}"
                )
                msg = (fallback_prompt | self.llm).invoke(
                    {"context": context, "question": question}
                )
                return getattr(msg, "content", str(msg))

            status = str(decision.get("status", "")).strip()
            if status == "final":
                return str(decision.get("answer", "")).strip() or "（未生成答案）"

            next_query = str(decision.get("next_query", "")).strip()
            if not next_query:
                break
            current_query = next_query

        final_context = _format_docs(accumulated_docs)
        final_prompt = ChatPromptTemplate.from_template(
            "请依据以下检索内容回答问题；若内容不足，请说明缺少哪一类信息。\n\n"
            "检索内容：\n{context}\n\n问题：{question}"
        )
        msg = (final_prompt | self.llm).invoke(
            {"context": final_context, "question": question}
        )
        return getattr(msg, "content", str(msg))
    
    
# def main() -> None:
#     """
#     自检：演示如何串联各组件。
#     默认不真实执行，以免触发模型下载；将 RUN_DEMO 设为 True 进行真实跑通。
#     """
#     RUN_DEMO = True
#     if not RUN_DEMO:
#         print("RAGPipeline 骨架已就绪。将 RUN_DEMO 设为 True 以实际构建索引与检索。")
#         return

#     sample_paths = ["user_data/documents"]
#     pipeline = RAGPipeline()
#     # 默认：如果向量库已有数据就跳过重建；需要强制重建请改为 reset=True
#     pipeline.build_index(sample_paths, reset=False, skip_if_exists=True)
#     results = pipeline.retrieve("什么是混淆电路？", k=10)
#     for i, doc in enumerate(results, start=1):
#         print(f"结果 {i}: {doc.metadata}")
#         print(doc.page_content[:900], "\n")
#     print(pipeline.answer("什么是混淆电路？", k=10))
#     print("---- iterative ----")
#     print(pipeline.answer_iterative("什么是混淆电路？", k=5, max_rounds=3))



if __name__ == "__main__":
    main()
