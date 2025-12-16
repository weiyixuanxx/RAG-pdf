"""
RAG 向量化骨架（含 TODO）。
- 管理嵌入模型的创建与配置
- 可在此接入镜像源、缓存目录、模型选择

注意：为避免在未安装依赖时一导入就报错，这里采用懒导入。
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingProvider:
    """
    负责提供 embedding_function。
    扩展点：镜像加速、GPU/CPU 选择、模型切换。
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        hf_endpoint: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.hf_endpoint = hf_endpoint
        self._embedding_fn: Optional["HuggingFaceEmbeddings"] = None
        
    def get(self) -> "HuggingFaceEmbeddings":
        """懒加载并返回 Embeddings 对象。"""
        if self._embedding_fn is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "缺少依赖：`langchain-huggingface`。\n"
                    "请先安装：`pip install langchain-huggingface sentence-transformers`"
                ) from e

            # 网络不稳时可用镜像：export HF_ENDPOINT=https://hf-mirror.com
            endpoint = self.hf_endpoint or os.getenv("HF_ENDPOINT")
            if endpoint:
                os.environ["HF_ENDPOINT"] = endpoint

            # TODO: 如需设置缓存目录：export HUGGINGFACE_HUB_CACHE=/path/to/cache
            self._embedding_fn = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._embedding_fn
    
# def main() -> None:
#     """
#     自检：仅打印模型名，避免真实拉取模型。
#     若需要真实测试，可取消注释调用 provider.get() 后再 embed_query。
#     """
#     provider = EmbeddingProvider()
#     print(f"Embedding 模型: {provider.model_name}")
#     demo = provider.get()
#     vec = demo.embed_query("你好，RAG！")
#     print("向量维度:", len(vec))
    
# if __name__ == "__main__":
#     main()
    
