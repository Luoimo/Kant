from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Annotated
import sys

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.agents.deepread_agent import DeepReadAgent, deepread_node
from backend.llm.openai_client import get_llm
from backend.rag.chroma.chroma_store import ChromaStore
from backend.security.input_filter import InputSafetyResult, run_input_safety_check


# 依赖注入：不可序列化依赖（子 agent 等）通过 GraphDeps 注入节点，不写入 state，避免 checkpoint 序列化报错。


@dataclass
class GraphDeps:
    """图内依赖容器，仅通过闭包传入节点，不写入 state。"""
    deepread_agent: DeepReadAgent


class GraphState(TypedDict, total=False):
    # 对话历史（多轮）：LangGraph 推荐的 messages 模式
    messages: Annotated[list[AnyMessage], add_messages]

    # 输入辅助字段（兼容旧调用方式）
    user_input: str
    book_source: str | None

    # 识别出的意图
    intent: Literal["recommend", "deepread", "notes", "plan"] | None
    intent_reason: str

    # 安全检查结果
    safety_ok: bool
    safety_reason: str

    # 总控下发给子 Agent 的任务（解耦：子 Agent 只读这些，不直接读 messages）
    deepread_query: str
    deepread_book_source: str | None

    # 输出
    answer: str
    citations: list[Any]
    retrieved_docs_count: int

    # Supervisor 路由字段
    next: Literal["deepread", "finalize", "end"]


# ---------------------------------------------------------------------------
# 意图识别（结构化输出）
# ---------------------------------------------------------------------------


class IntentSchema(BaseModel):
    intent: Literal["recommend", "deepread", "notes", "plan"] = Field(
        description="当前用户请求的主要意图类型。"
    )
    reason: str = Field(
        description="一句话解释为什么这么判断这个意图。"
    )
    book_source: str | None = Field(
        default=None,
        description="如果用户显式提到了某本书/文件路径，这里给出它；否则为 None。",
    )


def classify_intent(user_input: str) -> IntentSchema:
    """
    使用 LLM 做一次结构化意图识别，将请求归类为
    {recommend, deepread, notes, plan} 之一。
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(IntentSchema)

    prompt = (
        "你是一个意图区分器，只负责判断下面这句中文请求的类型，"
        "范围限定在读书相关：找书推荐、针对一本书的精读或问答、"
        "整理读书笔记、制定阅读计划。\n\n"
        "请根据用户文本填写 IntentSchema：\n"
        "- recommend：请求推荐/发现小众书\n"
        "- deepread：围绕某本书的章节/概念进行精读、解释、问答或基于证据的回答（含自由问答）\n"
        "- notes：整理/总结/结构化读书笔记\n"
        "- plan：制定或调整阅读书单/节奏/路线\n\n"
        f"用户输入：{user_input!r}"
    )

    return structured_llm.invoke(prompt)


def supervisor_node(state: GraphState) -> GraphState:
    """
    最小 Supervisor：
    - 第一次必走 deepread
    - deepread 完成后走 finalize
    - finalize 后结束
    """
    print("[Graph] → supervisor_node", file=sys.stdout)

    # 从 messages 中提取最近一条 user 消息内容；兼容 user_input 字段
    user_input = state.get("user_input", "") or ""
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if getattr(m, "type", None) == "user":
            user_input = getattr(m, "content", "") or user_input
            break

    # 1) 安全检查（只做一次）
    if "safety_ok" not in state:
        safety_result: InputSafetyResult = run_input_safety_check(user_input)
        state["safety_ok"] = safety_result.allowed
        state["safety_reason"] = safety_result.reason
        print(
            f"[Supervisor] safety_ok={safety_result.allowed}, "
            f"categories={safety_result.categories}, "
            f"reason={safety_result.reason}",
            file=sys.stdout,
        )

        if not safety_result.allowed:
            state["answer"] = f"当前请求未通过安全检查：{safety_result.reason}"
            state["next"] = "finalize"
            return state

    # 2) 意图识别（只做一次）
    if state.get("intent") is None:
        intent_result = classify_intent(user_input)
        state["intent"] = intent_result.intent
        state["intent_reason"] = intent_result.reason
        # 如果分类器识别出具体书源且当前没有 book_source，用它覆盖
        if not state.get("book_source") and intent_result.book_source:
            state["book_source"] = intent_result.book_source

        print(
            f"[Supervisor] intent={intent_result.intent}, "
            f"reason={intent_result.reason}, "
            f"book_source={state.get('book_source')!r}",
            file=sys.stdout,
        )

    # 3) 若 deepread 已产出 answer，本轮回合结束，走 finalize，避免 deepread → supervisor 死循环
    if state.get("answer"):
        state["next"] = "finalize"
        return state

    # 4) 根据意图路由到对应 agent；路由前写入本轮的“子任务”，子 Agent 只读这些字段
    intent = state.get("intent") or "deepread"
    state["deepread_query"] = user_input
    state["deepread_book_source"] = state.get("book_source")
    state["next"] = "deepread"

    return state


def finalize_node(state: GraphState) -> GraphState:
    print("[Graph] → finalize_node", file=sys.stdout)
    # 这里先保持最小：不做额外格式化，只负责结束。
    state["next"] = "end"
    return state


def build_minimal_supervisor_graph(*, store: ChromaStore | None = None):
    """
    构建最小可跑的 supervisor 编排图：

    START -> supervisor -> deepread -> supervisor -> finalize -> END

    """
    if store is None:
        settings = get_settings()
        store = ChromaStore(collection_name=settings.chroma_database)

    deps = GraphDeps(deepread_agent=DeepReadAgent(store=store))

    def _deepread(state: GraphState) -> GraphState:
        print("[Graph] → deepread_node", file=sys.stdout)
        patch = deepread_node(state, agent=deps.deepread_agent)  # type: ignore[arg-type]
        state.update(patch)  # answer/citations/...
        print(
            f"[Graph]   deepread_node done, "
            f"retrieved_docs_count={state.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return state

    graph = StateGraph(GraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("deepread", _deepread)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s.get("next", "deepread"),
        {
            "deepread": "deepread",
            "finalize": "finalize",
        },
    )
    graph.add_edge("deepread", "supervisor")
    graph.add_conditional_edges(
        "finalize",
        lambda s: s.get("next", "end"),
        {"end": END},
    )

    # 使用内存 checkpointer，使得相同 thread_id 下多轮对话共享 GraphState（尤其是 messages）
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    # agent 通过闭包注入到 _deepread，不放入 state，避免 checkpointer 序列化 DeepReadAgent 报错
    return compiled


_minimal_app = None


def run_minimal_graph(
    query: str,
    *,
    book_source: str | None = None,
    thread_id: str = "default",
) -> GraphState:
    """
    支持多轮对话的入口：
    - 相同 thread_id 下，多次调用会共享同一条对话历史（messages）。
    - 不同 thread_id 互相隔离。
    """
    global _minimal_app
    if _minimal_app is None:
        _minimal_app = build_minimal_supervisor_graph()

    app = _minimal_app
    init: GraphState = {
        "messages": [("user", query)],
        "user_input": query,
        "book_source": book_source,
    }
    return app.invoke(init, config={"thread_id": thread_id})  # type: ignore[return-value]