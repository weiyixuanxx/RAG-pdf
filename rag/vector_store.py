from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_chroma import Chroma

class VectorStoreManager:
    """
    负责创建/持久化向量库
    提供增量写入与检索接口
    """

    def __init__(
        self,
        embedding_function,
        collection_name: str = "rag_collection",
        persist_directory: str | None = "./chroma_langchain_db",
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self._vector_store: Optional["Chroma"] = None

    def reset_collection(self) -> None:
        """
        删除当前 collection（会清空向量库数据）。
        用于避免重复写入或重建索引。
        """
        store = self._get_store()
        # langchain_chroma.Chroma 提供 delete_collection
        store.delete_collection()
        self._vector_store = None

    def _get_store(self) -> "Chroma":
        if self._vector_store is None:
            try:
                from langchain_chroma import Chroma
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "缺少依赖：`langchain-chroma`（以及通常需要 `chromadb`）。\n"
                    "请先安装：`pip install langchain-chroma chromadb`"
                ) from e
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
        return self._vector_store

    def count(self) -> int:
        """返回当前 collection 的记录数（无法获取时返回 0）。"""
        store = self._get_store()
        try:
            if hasattr(store, "_collection") and getattr(store, "_collection") is not None:
                return int(store._collection.count())  # type: ignore[attr-defined]
        except Exception:
            return 0
        return 0

    def _stable_id(self, doc: Document) -> str:
        """
        为每个切片生成稳定 ID，用于避免重复写入。
        基于：source/page/start_index + 内容 hash。
        """
        meta = doc.metadata or {}
        source = str(meta.get("source", ""))
        page = str(meta.get("page", ""))
        start_index = str(meta.get("start_index", ""))
        content_hash = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
        return f"{source}#p{page}#s{start_index}#{content_hash}"

    def add_documents(self, docs: Iterable[Document]) -> List[str]:
        """写入文档，返回生成的 ID。"""
        store = self._get_store()
        docs_list = list(docs)
        ids = [self._stable_id(d) for d in docs_list]

        # 用 upsert 让重复运行可复用同一批 ids，但必须显式传 embeddings，
        # 否则 Chroma 会使用其默认 embedding_function（常见为 384 维），
        # 导致后续用外部 embeddings 查询时报维度不匹配。
        if hasattr(store, "_collection") and getattr(store, "_collection") is not None:
            embeddings = self.embedding_function.embed_documents(
                [d.page_content for d in docs_list]
            )
            store._collection.upsert(  # type: ignore[attr-defined]
                ids=ids,
                embeddings=embeddings,
                documents=[d.page_content for d in docs_list],
                metadatas=[d.metadata for d in docs_list],
            )
            return ids

        return store.add_documents(docs_list, ids=ids)

    def as_retriever(self, k: int = 3):
        """返回检索器接口，供上层链路使用。"""
        store = self._get_store()
        return store.as_retriever(k=k)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        store = self._get_store()
        return store.similarity_search(query, k=k)
    
def main() -> None:
    """
    自检：仅展示初始化信息，不默认写入数据。
    如需完整测试，请传入真实 embedding_function 与文档。
    """
    print("VectorStoreManager 骨架就绪，初始化示例：")
    print("VectorStoreManager(embedding_function=..., collection_name='rag_collection')")


if __name__ == "__main__":
    main()
