from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from langchain.agents.factory import create_agent
from langgraph.checkpoint.memory import MemorySaver

from rag.pipeline import RAGPipeline
from rag.tools import make_retrieval_tool


APP_TITLE = "RAG 多方安全计算知识助手"


@st.cache_resource
def _get_pipeline() -> RAGPipeline:
    return RAGPipeline()


@st.cache_resource
def _get_agent_graph() -> object:
    pipeline = _get_pipeline()
    rag_tool = make_retrieval_tool(pipeline, k=5, max_chars=7000)

    system_prompt = (
        "你是一个中文知识助手。你可以调用工具从本地知识库检索内容。\n"
        "当用户的问题需要依据文档回答时：先调用 rag_retrieve 获取上下文，再基于上下文回答。\n"
        "回答要求：\n"
        "- 若上下文中没有答案，明确说明“文档未包含”。\n"
        "- 尽量给出引用（用 [n] 标号即可）。\n"
    )

    # MemorySaver：按 thread_id 维护对话状态
    checkpointer = MemorySaver()
    graph = create_agent(
        model=pipeline.llm,
        tools=[rag_tool],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
    return graph


def _ensure_index_ready(pipeline: RAGPipeline, data_dir: str) -> None:
    db_dir = Path("./chroma_langchain_db")
    if db_dir.exists():
        return
    pipeline.build_index([data_dir], reset=False, skip_if_exists=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    pipeline = _get_pipeline()
    agent_graph = _get_agent_graph()

    with st.sidebar:
        st.header("设置")
        data_dir = st.text_input("PDF 目录", value="user_data/documents")
        auto_index = st.checkbox("启动时自动建库（若不存在）", value=True)
        if st.button("强制重建向量库"):
            pipeline.build_index([data_dir], reset=True, skip_if_exists=False)
            st.success("已重建向量库。")
        st.caption(
            "提示：首次建库会下载 embedding 模型，耗时较长；后续会复用本地缓存与向量库。"
        )

    if auto_index:
        _ensure_index_ready(pipeline, data_dir)

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = os.urandom(8).hex()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("请输入你的问题…")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("思考中…")

        # create_agent 的 graph 走的是 messages state；thread_id 用于记忆
        result = agent_graph.invoke(
            {"messages": [("user", user_text)]},
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        )

        # result 结构：{"messages": [...]}，取最后一条 AI 消息
        final_text = ""
        try:
            msgs = result.get("messages", [])
            if msgs:
                final_text = getattr(msgs[-1], "content", str(msgs[-1]))
        except Exception:
            final_text = str(result)

        placeholder.markdown(final_text or "（未生成内容）")
        st.session_state.messages.append({"role": "assistant", "content": final_text})


if __name__ == "__main__":
    main()

