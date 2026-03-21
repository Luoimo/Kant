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
from backend.agents.note_agent import NoteAgent, notes_node
from backend.agents.reading_plan_agent import ReadingPlanAgent, plan_node
from backend.agents.recommendation_agent import RecommendationAgent, recommend_node
from backend.llm.openai_client import get_llm
from backend.memory.mem0_store import Mem0Store
from backend.rag.chroma.chroma_store import ChromaStore
from backend.security.input_filter import InputSafetyResult, run_input_safety_check


# 依赖注入：不可序列化依赖（子 agent 等）通过 GraphDeps 注入节点，不写入 state，避免 checkpoint 序列化报错。


@dataclass
class GraphDeps:
    """图内依赖容器，仅通过闭包传入节点，不写入 state。"""
    deepread_agent: DeepReadAgent
    notes_agent: NoteAgent
    plan_agent: ReadingPlanAgent
    recommend_agent: RecommendationAgent
    mem0: Mem0Store | None = None


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

    notes_query: str
    notes_book_source: str | None
    notes_raw_text: str | None

    plan_query: str
    plan_book_source: str | None

    recommend_query: str

    # 记忆上下文（由 memory_search 节点写入，子 Agent 只读）
    memory_context: str

    # 输出
    answer: str
    citations: list[Any]
    retrieved_docs_count: int

    # Supervisor 路由字段
    next: Literal["deepread", "notes", "plan", "recommend", "finalize", "end"]


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
    Supervisor：
    - 安全检查 → 意图识别 → 路由到对应 agent
    - agent 完成后返回 supervisor → finalize
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

    # 3) 若 agent 已产出 answer，本轮回合结束，走 finalize
    if state.get("answer"):
        state["next"] = "finalize"
        return state

    # 4) 根据意图路由到对应 agent，写入子任务字段
    intent = state.get("intent") or "deepread"
    if intent == "notes":
        state["notes_query"] = user_input
        state["notes_book_source"] = state.get("book_source")
        state["next"] = "notes"
    elif intent == "plan":
        state["plan_query"] = user_input
        state["plan_book_source"] = state.get("book_source")
        state["next"] = "plan"
    elif intent == "recommend":
        state["recommend_query"] = user_input
        state["next"] = "recommend"
    else:
        # deepread（默认）
        state["deepread_query"] = user_input
        state["deepread_book_source"] = state.get("book_source")
        state["next"] = "deepread"

    return state


def build_minimal_supervisor_graph(*, store: ChromaStore | None = None, enable_memory: bool = True):
    """
    构建 supervisor 编排图：

    START -> supervisor -> [deepread|notes|plan|recommend] -> supervisor -> finalize -> END

    """
    if store is None:
        settings = get_settings()
        store = ChromaStore(collection_name=settings.chroma_database)

    mem0 = Mem0Store() if enable_memory else None

    deps = GraphDeps(
        deepread_agent=DeepReadAgent(store=store),
        notes_agent=NoteAgent(store=store),
        plan_agent=ReadingPlanAgent(store=store),
        recommend_agent=RecommendationAgent(store=store),
        mem0=mem0,
    )

    def _memory_search(state: GraphState) -> GraphState:
        """每次请求开始时，从 Mem0 检索历史记忆写入 state。"""
        print("[Graph] → memory_search_node", file=sys.stdout)
        if deps.mem0 is None:
            state["memory_context"] = ""
            return state
        user_input = state.get("user_input", "") or ""
        past = deps.mem0.search(user_input, top_k=3)
        state["memory_context"] = "\n".join(f"- {m}" for m in past) if past else ""
        if past:
            print(f"[Memory] 找到 {len(past)} 条历史记忆", file=sys.stdout)
        return state

    def _finalize(state: GraphState) -> GraphState:
        """回答完成后，把本次问答存入 Mem0 长期记忆。"""
        print("[Graph] → finalize_node", file=sys.stdout)
        if deps.mem0 and state.get("answer") and state.get("user_input"):
            deps.mem0.add_qa(state["user_input"], state["answer"])
            print("[Memory] 已保存本次问答到 Mem0", file=sys.stdout)
        state["next"] = "end"
        return state

    def _deepread(state: GraphState) -> GraphState:
        print("[Graph] → deepread_node", file=sys.stdout)
        patch = deepread_node(state, agent=deps.deepread_agent)  # type: ignore[arg-type]
        state.update(patch)
        print(
            f"[Graph]   deepread_node done, "
            f"retrieved_docs_count={state.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return state

    def _notes(state: GraphState) -> GraphState:
        print("[Graph] → notes_node", file=sys.stdout)
        patch = notes_node(state, agent=deps.notes_agent)  # type: ignore[arg-type]
        state.update(patch)
        print(
            f"[Graph]   notes_node done, "
            f"retrieved_docs_count={state.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return state

    def _plan(state: GraphState) -> GraphState:
        print("[Graph] → plan_node", file=sys.stdout)
        patch = plan_node(state, agent=deps.plan_agent)  # type: ignore[arg-type]
        state.update(patch)
        print(
            f"[Graph]   plan_node done, "
            f"retrieved_docs_count={state.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return state

    def _recommend(state: GraphState) -> GraphState:
        print("[Graph] → recommend_node", file=sys.stdout)
        patch = recommend_node(state, agent=deps.recommend_agent)  # type: ignore[arg-type]
        state.update(patch)
        print(
            f"[Graph]   recommend_node done, "
            f"retrieved_docs_count={state.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return state

    graph = StateGraph(GraphState)
    graph.add_node("memory_search", _memory_search)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("deepread", _deepread)
    graph.add_node("notes", _notes)
    graph.add_node("plan", _plan)
    graph.add_node("recommend", _recommend)
    graph.add_node("finalize", _finalize)

    graph.add_edge(START, "memory_search")
    graph.add_edge("memory_search", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s.get("next", "deepread"),
        {
            "deepread": "deepread",
            "notes": "notes",
            "plan": "plan",
            "recommend": "recommend",
            "finalize": "finalize",
        },
    )
    graph.add_edge("deepread", "supervisor")
    graph.add_edge("notes", "supervisor")
    graph.add_edge("plan", "supervisor")
    graph.add_edge("recommend", "supervisor")
    graph.add_conditional_edges(
        "finalize",
        lambda s: s.get("next", "end"),
        {"end": END},
    )

    # 使用内存 checkpointer，使得相同 thread_id 下多轮对话共享 GraphState（尤其是 messages）
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

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
    return app.invoke(init, config={"configurable": {"thread_id": thread_id}})  # type: ignore[return-value]
