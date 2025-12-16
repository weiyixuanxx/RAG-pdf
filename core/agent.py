from typing import Any, Callable, Dict, Iterable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

try:
    # When imported as `core.agent`
    from .prompts import DEFAULT_SYSTEM_PROMPT  # type: ignore
except ImportError:
    # When executed directly: `python core/agent.py`
    from prompts import DEFAULT_SYSTEM_PROMPT


class BaseAgent:
    """
    轻量可复用的 Agent 基类，基于最新的 LangChain Runnable API。
    - 传入任意支持 Runnable 的 LLM（如 ChatOpenAI）即可使用。
    - 通过覆写 build_prompt 或 build_chain 增加自定义工具 / 解析。
    - 默认内置 RunnableWithMessageHistory，实现按 session 的对话记忆。
    """

    def __init__(
        self,
        llm: Runnable,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        input_key: str = "input",
        history_messages_key: str = "history",
        history_factory: Callable[[], BaseChatMessageHistory] | None = None,
    ):
        # llm 必须符合 Runnable 协议，至少包含 invoke/stream
        if not hasattr(llm, "invoke"):
            raise TypeError("llm 必须符合 Runnable 协议，至少包含 invoke/stream")
        self.llm = llm
        self.system_prompt = system_prompt.strip()
        self.input_key = input_key
        self.history_messages_key = history_messages_key
        self._history_store: Dict[str, BaseChatMessageHistory] = {}
        self._history_factory = history_factory or ChatMessageHistory

        # 构建提示词与链路，并挂载对话记忆
        self.prompt = self.build_prompt()
        self.chain = self.build_chain()
        self.chain_with_history = self._wrap_with_history(self.chain)

    # ----- Customization hooks -------------------------------------------------
    def build_prompt(self) -> ChatPromptTemplate:
        """可重写：自定义对话模板。"""
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(self.history_messages_key),
                ("human", f"{{{self.input_key}}}"),
            ]
        )

    def build_chain(self) -> Runnable:
        """可重写：插入工具 / 解析器，默认 prompt -> llm。"""
        return self.prompt | self.llm

    # ----- Memory helpers ------------------------------------------------------
    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._history_store:
            self._history_store[session_id] = self._history_factory()
        return self._history_store[session_id]

    def _wrap_with_history(self, chain: Runnable) -> RunnableWithMessageHistory:
        return RunnableWithMessageHistory(
            chain,
            self._get_history,
            input_messages_key=self.input_key,
            history_messages_key=self.history_messages_key,
        )

    def clear_history(self, session_id: str = "default") -> None:
        self._history_store.pop(session_id, None)

    # ----- Public inference APIs ----------------------------------------------
    def invoke(self, user_input: str, session_id: str = "default", **kwargs: Any) -> Any:
        payload = {self.input_key: user_input, **kwargs}
        return self.chain_with_history.invoke(
            payload, config={"configurable": {"session_id": session_id}}
        )

    def stream(
        self, user_input: str, session_id: str = "default", **kwargs: Any
    ) -> Iterable[Any]:
        payload = {self.input_key: user_input, **kwargs}
        return self.chain_with_history.stream(
            payload, config={"configurable": {"session_id": session_id}}
        )

    def update_system_prompt(self, new_prompt: str) -> None:
        self.system_prompt = new_prompt.strip()
        self.prompt = self.build_prompt()
        self.chain = self.build_chain()
        self.chain_with_history = self._wrap_with_history(self.chain)

    def get_messages(self, session_id: str = "default") -> list:
        """Return raw message objects for debugging/UI purposes."""
        return list(self._get_history(session_id).messages)
