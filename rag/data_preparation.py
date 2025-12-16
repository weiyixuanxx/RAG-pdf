import re
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentIngestor:
    """
    负责 PDF 加载与文本切分。
    可以在此扩展清洗、元数据补充、过滤等逻辑。
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        *,
        loader: str = "auto",
        min_page_chars: int = 20,
        min_chunk_chars: int = 200,
        verbose: bool = False,
    ) -> None:
        # 分词器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
        self.loader = loader
        self.min_page_chars = min_page_chars
        self.min_chunk_chars = min_chunk_chars
        self.verbose = verbose
        
    """ 清洗页面内容的辅助方法 """
    def _clean_page_content(self, content: str) -> str:
        content = content.replace("\xa0", " ").replace("\u200b", " ")
        content = re.sub(r"-\s*\n\s*", "", content)  # 连字符断行
        content = content.replace("\n", " ")
        content = re.sub(r"[ \t]+", " ", content)
        return content.strip()
        
    """ 加载单个 PDF 文件 """
    def _load_single_pdf(self, pdf_path: Path) -> List[Document]:
        # LaTeX/两栏/公式较多的 PDF，pypdf 往往抽不到正文；优先尝试 pdfplumber。
        docs: List[Document] = []
        if self.loader in ("auto", "pdfplumber"):
            try:
                from langchain_community.document_loaders import PDFPlumberLoader

                docs = PDFPlumberLoader(str(pdf_path)).load()
                if self.verbose:
                    print(f"使用 PDFPlumberLoader: {pdf_path}")
            except Exception as e:
                if self.loader == "pdfplumber":
                    raise
                if self.verbose:
                    print(f"PDFPlumberLoader 失败，回退 PyPDFLoader: {pdf_path} ({e})")

        if not docs:
            docs = PyPDFLoader(str(pdf_path)).load()
            if self.verbose:
                print(f"使用 PyPDFLoader: {pdf_path}")

        cleaned_docs = []
        # 针对每页做清洗
        for doc in docs:
            cleaned_content = self._clean_page_content(doc.page_content)
            if len(cleaned_content) < self.min_page_chars:
                continue
            cleaned_doc = Document(
                page_content=cleaned_content,
                metadata={**(doc.metadata or {}), "source": str(pdf_path)},
            )
            cleaned_docs.append(cleaned_doc)
        return cleaned_docs
    
    def load_documents(self, paths: Iterable[str]) -> List[Document]:
        """从给定路径加载 PDF；目录会扫描所有 .pdf。"""
        all_docs: List[Document] = []
        print(f"开始加载文档, 文档个数: {len(paths)}")
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                for pdf_file in sorted(path.glob("*.pdf")):
                    all_docs.extend(self._load_single_pdf(pdf_file))
            elif path.is_file() and path.suffix.lower() == ".pdf":
                all_docs.extend(self._load_single_pdf(path))
            else:
                print(f"跳过不支持的路径: {raw_path}")
        return all_docs
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """将文档切分成小块。"""
        splits = self.text_splitter.split_documents(docs)
        # 过滤掉“标题/目录/空片段”等低信息量内容，减少检索噪声
        filtered: List[Document] = []
        for d in splits:
            content = (d.page_content or "").strip()
            if len(content) < self.min_chunk_chars:
                continue
            if self._is_noise_chunk(content):
                continue
            filtered.append(d)
        if self.verbose:
            print(f"切分片段：{len(splits)}，过滤后：{len(filtered)}")
        return filtered

    def _is_noise_chunk(self, content: str) -> bool:
        """
        过滤“目录样式/章节标题”噪声片段。
        启发式规则（可按你的 PDF 特性继续调整）：
        - 大量连续点号（目录 leader）
        - 含“目录/CHAPTER”且内容偏短（多为标题/目录而非正文）
        """
        text = content.strip()
        if not text:
            return True

        # 目录常见的 leader：". . . . . . ." 或 "......"
        if re.search(r"(\.\s*){20,}", text):
            return True
        if re.search(r"\.{8,}", text):
            return True

        # 标题/目录页：关键词密集但信息量低
        upper = text.upper()
        if ("目录" in text or "CONTENTS" in upper) and len(text) < 1500:
            return True
        if ("CHAPTER" in upper) and len(text) < 400:
            return True

        # 目录页常见：很多行以页码结尾，且点号占比高
        dot_ratio = text.count(".") / max(1, len(text))
        if dot_ratio > 0.18 and re.search(r"\s\d{1,4}\s*$", text):
            return True

        return False
    
    def ingest(self, paths: Iterable[str]) -> List[Document]:
        """高阶流程：加载 -> 切分 -> 返回切片。"""
        docs = self.load_documents(paths)
        if not docs:
            print("未加载到任何文档。")
            return []
        
        print(f",{len(docs)} 页，开始切分...")
        splits = self.split_documents(docs)
        print(f"切分完成，共 {len(splits)} 个片段。")
        return splits
    
# def main() -> None:
#     """本地自检：尝试读取目录并打印首个切片。"""
#     sample_paths = ["user_data/documents"]
#     ingestor = DocumentIngestor(chunk_size=800, chunk_overlap=150)
#     chunks = ingestor.ingest(sample_paths)
    
#     if chunks:
#         print("首个切片预览:")
#         print(chunks[0].page_content[:300])
#         print("元数据:", chunks[0].metadata)
        
# if __name__ == "__main__":
#     main()  
