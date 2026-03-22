from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, Annotated
import sys

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3 as _sqlite3
    _SQLITE_AVAILABLE = True
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as SqliteSaver  # type: ignore[assignment]
    _SQLITE_AVAILABLE = False
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import create_react_agent

from pathlib import Path

from backend.config import get_settings
from backend.agents.deepread_agent import DeepReadAgent
from backend.agents.note_agent import NoteAgent
from backend.agents.plan_editor import PlanEditor
from backend.agents.recommendation_agent import RecommendationAgent
from backend.llm.openai_client import get_llm
from backend.memory.mem0_store import Mem0Store
from backend.rag.chroma.chroma_store import ChromaStore
from backend.security.input_filter import InputSafetyResult, run_input_safety_check
from backend.storage.note_vector_store import make_note_vector_store


# ---------------------------------------------------------------------------
# 书名解析工具函数
# ---------------------------------------------------------------------------


def _title_from_docs(docs) -> str:
    """从检索结果元数据中提取书名（deepread_book 自动笔记 hook 的 fallback）。"""
    for doc in docs:
        title = (doc.metadata or {}).get("book_title", "")
        if title:
            return title
    return ""


# ---------------------------------------------------------------------------
# 依赖容器（不可序列化，通过闭包注入节点）
# ---------------------------------------------------------------------------

@dataclass
class GraphDeps:
    deepread_agent: DeepReadAgent
    notes_agent: NoteAgent
    plan_agent: PlanEditor
    recommend_agent: RecommendationAgent
    mem0: Mem0Store | None = None


# ---------------------------------------------------------------------------
# 请求级上下文（工具调用的副作用收集器）
# ---------------------------------------------------------------------------

@dataclass
class RequestContext:
    """每次 run() 调用重置，工具通过此对象传递 citations 等副作用。"""
    citations: list = field(default_factory=list)
    retrieved_docs_count: int = 0
    intent: str = "deepread"


# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    # 对话历史
    messages: Annotated[list[AnyMessage], add_messages]

    # 输入字段
    user_input: str
    book_id: str | None

    # 安全检查
    safety_ok: bool
    safety_reason: str

    # 记忆上下文
    memory_context: str

    # 输出
    answer: str
    citations: list[Any]
    retrieved_docs_count: int
    intent: str

    # Reader 模式
    selected_text: str | None
    current_chapter: str | None


# ---------------------------------------------------------------------------
# Supervisor 系统提示
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM = """你是智能阅读助手的总协调器，可使用以下工具：

- deepread_book: 对书库中的书籍进行深度精读、解析和问答（每次调用后系统会自动记录笔记）
- modify_plan: 修改或扩展用户已有的阅读计划（在 Reader Mode 中打开书后计划会自动生成）
- recommend_books: 从整个出版世界中推荐值得精读的书籍

工作原则：
1. 分析用户请求，选择合适的工具（可依次调用多个）
2. 修改计划时：直接调用 modify_plan，只需传入用户的修改要求（书籍已自动关联）
3. 将所有工具结果整合为一份连贯完整的 Markdown 回答
4. 如检测到用户报告阅读进度（"XX章节我读完了"），调用 modify_plan 更新计划

重要规则：
- 只使用工具返回的内容作答，不编造不存在的事实
"""

# ---------------------------------------------------------------------------
# 工具构建（每次请求重建，捕获请求级上下文）
# ---------------------------------------------------------------------------

def _build_supervisor_tools(
    deps: GraphDeps,
    ctx: RequestContext,
    memory_context: str,
    thread_id: str,
    book_source: str | None,
    book_title: str,
):
    """构建 3 个子 Agent 工具，通过闭包绑定请求上下文和当前书籍信息。"""

    @tool
    def deepread_book(query: str) -> str:
        """对当前打开的书籍进行深度精读、解析和问答。
        支持多次检索以验证证据充分性。每次调用后系统会自动将问答记录到笔记。
        """
        result = deps.deepread_agent.run(
            query=query,
            book_source=book_source,
            memory_context=memory_context,
        )
        ctx.citations = result.citations
        ctx.retrieved_docs_count = len(result.retrieved_docs)
        ctx.intent = "deepread"
        print(f"[Supervisor.tool] deepread_book done, hits={len(result.retrieved_docs)}", file=sys.stdout)

        # 自动触发笔记 hook：优先使用闭包中的 book_title，fallback 到文档元数据
        resolved_title = book_title or _title_from_docs(result.retrieved_docs)
        if resolved_title:
            try:
                deps.notes_agent.process_qa(query, result.answer, resolved_title)
            except Exception as e:
                print(f"[Supervisor.tool] note hook failed: {e}", file=sys.stderr)

        return result.answer

    @tool
    def modify_plan(query: str, action: str = "edit") -> str:
        """修改或扩展当前书籍的阅读计划。
        action: 'edit'（修改内容）或 'extend'（增加内容）
        """
        if not book_title:
            return "当前未打开任何书籍，无法修改计划。"
        safe_action: Literal["edit", "extend"] = "extend" if action == "extend" else "edit"
        result = deps.plan_agent.run(
            query=query,
            book_title=book_title,
            action=safe_action,
            memory_context=memory_context,
        )
        ctx.citations = result.citations
        ctx.retrieved_docs_count += len(result.retrieved_docs)  # PlanEditor 始终返回空列表，+= 为一致性
        ctx.intent = "plan"
        print("[Supervisor.tool] modify_plan done", file=sys.stdout)
        return result.answer

    @tool
    def recommend_books(query: str, recommend_type: str = "discover") -> str:
        """从整个出版世界中推荐值得精读的书籍（不限于本地书库）。
        recommend_type: discover(发现新书) / similar(相似书) / next(下一本) / theme(主题推荐)
        会自动标注书籍是否已在本地书库（✅已在库 / 📥可上传）。
        """
        result = deps.recommend_agent.run(
            query=query,
            current_book=book_title or "",
            memory_context=memory_context,
            recommend_type=recommend_type,  # type: ignore[arg-type]
        )
        ctx.intent = "recommend"
        print("[Supervisor.tool] recommend_books done", file=sys.stdout)
        return result.answer

    return [deepread_book, modify_plan, recommend_books]


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

def build_minimal_supervisor_graph(
    *,
    store: ChromaStore | None = None,
    enable_memory: bool = True,
    _return_deps: bool = False,
):
    """
    构建简化的 ReAct Supervisor 编排图：

    START → memory_search → react_supervisor → finalize → END

    Supervisor 是一个 ReAct Agent，子 Agent 作为其工具。
    """
    if store is None:
        store = ChromaStore()

    settings = get_settings()
    mem0 = Mem0Store() if enable_memory else None

    note_vector_store = make_note_vector_store(settings)
    deps = GraphDeps(
        deepread_agent=DeepReadAgent(store=store),
        notes_agent=NoteAgent(note_vector_store=note_vector_store),
        plan_agent=PlanEditor(store=store),
        recommend_agent=RecommendationAgent(store=store),
        mem0=mem0,
    )

    llm = get_llm(temperature=0.1)

    def _memory_search(state: GraphState) -> dict:
        print("[Graph] → memory_search_node", file=sys.stdout)
        if deps.mem0 is None:
            return {"memory_context": ""}
        user_input = state.get("user_input", "") or ""
        past = deps.mem0.search(user_input, top_k=3)
        ctx = "\n".join(f"- {m}" for m in past) if past else ""
        if past:
            print(f"[Memory] 找到 {len(past)} 条历史记忆", file=sys.stdout)
        return {"memory_context": ctx}

    def _react_supervisor(state: GraphState, config) -> dict:
        print("[Graph] → react_supervisor_node", file=sys.stdout)
        thread_id = config["configurable"].get("thread_id", "default")
        user_input = state.get("user_input", "") or ""

        # 1) 安全检查
        safety_result: InputSafetyResult = run_input_safety_check(user_input)
        if not safety_result.allowed:
            print(f"[Supervisor] safety blocked: {safety_result.reason}", file=sys.stdout)
            return {
                "safety_ok": False,
                "safety_reason": safety_result.reason,
                "answer": f"当前请求未通过安全检查：{safety_result.reason}",
                "citations": [],
                "retrieved_docs_count": 0,
                "intent": "deepread",
            }

        memory_context = state.get("memory_context", "") or ""

        ctx = RequestContext()

        # 解析当前书籍信息（book_id → source + book_title）
        book_id = state.get("book_id")
        book_info = store.resolve_book_by_id(book_id) if book_id else None
        book_source = book_info["source"] if book_info else None
        book_title = book_info.get("book_title", "") if book_info else ""

        # 构建 supervisor ReAct agent（每次请求重建以捕获最新上下文）
        tools = _build_supervisor_tools(deps, ctx, memory_context, thread_id, book_source, book_title)
        supervisor_agent = create_react_agent(llm, tools, prompt=SUPERVISOR_SYSTEM)

        # 构造用户消息（注入 reader 模式上下文）
        task_content = user_input
        selected_text = state.get("selected_text") or ""
        current_chapter = state.get("current_chapter") or ""

        if selected_text:
            task_content = f"【用户划选的原文片段】：\n{selected_text}\n\n【用户问题】：\n{task_content}"
        if current_chapter:
            task_content += f"\n\n【当前阅读章节】：{current_chapter}"

        # 传入近期对话历史（多轮上下文）
        recent_messages = list(state.get("messages") or [])
        # 去掉最后一条 human message（它会在 invoke 时重新作为 user 传入）
        if recent_messages and getattr(recent_messages[-1], "type", None) == "human":
            recent_messages = recent_messages[:-1]
        # 保留最近 8 条消息作为上下文
        history_for_agent = recent_messages[-8:] if len(recent_messages) > 8 else recent_messages

        input_messages = list(history_for_agent) + [("user", task_content)]

        print(f"[Supervisor] invoking react agent, user_input={user_input[:80]!r}", file=sys.stdout)

        result = supervisor_agent.invoke(
            {"messages": input_messages},
            config={"recursion_limit": 14},
        )

        answer = result["messages"][-1].content

        return {
            "safety_ok": True,
            "answer": answer,
            "citations": ctx.citations,
            "retrieved_docs_count": ctx.retrieved_docs_count,
            "intent": ctx.intent,
            "messages": [AIMessage(content=answer)],
        }

    def _finalize(state: GraphState) -> dict:
        print("[Graph] → finalize_node", file=sys.stdout)
        answer = state.get("answer") or ""
        if deps.mem0 and answer and state.get("user_input"):
            deps.mem0.add_qa(state["user_input"], answer)
            print("[Memory] 已保存本次问答到 Mem0", file=sys.stdout)
        return {}

    # 图构建
    graph = StateGraph(GraphState)
    graph.add_node("memory_search", _memory_search)
    graph.add_node("react_supervisor", _react_supervisor)
    graph.add_node("finalize", _finalize)

    graph.add_edge(START, "memory_search")
    graph.add_edge("memory_search", "react_supervisor")
    graph.add_edge("react_supervisor", "finalize")
    graph.add_edge("finalize", END)

    # Checkpointer
    if _SQLITE_AVAILABLE:
        import os
        os.makedirs("data", exist_ok=True)
        conn = _sqlite3.connect("data/checkpoints.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    else:
        checkpointer = SqliteSaver()  # type: ignore[call-arg]

    compiled = graph.compile(checkpointer=checkpointer)
    if _return_deps:
        return compiled, deps
    return compiled


# ---------------------------------------------------------------------------
# 全局单例 + 入口
# ---------------------------------------------------------------------------

_minimal_app = None
_minimal_deps: GraphDeps | None = None


def invalidate_bm25_caches() -> None:
    """入库新书后调用，清除 NoteAgent / ReadingPlanAgent 的 BM25 索引缓存。"""
    if _minimal_deps is None:
        return
    for agent in (_minimal_deps.notes_agent, _minimal_deps.plan_agent):
        retriever = getattr(agent, "_retriever", None)
        if retriever is not None:
            retriever.invalidate_bm25()


def run_minimal_graph(
    query: str,
    *,
    book_id: str | None = None,
    thread_id: str = "default",
    selected_text: str | None = None,
    current_chapter: str | None = None,
) -> GraphState:
    """
    支持多轮对话的入口：
    - 相同 thread_id 下，多次调用会共享同一条对话历史。
    - 不同 thread_id 互相隔离。
    """
    global _minimal_app, _minimal_deps
    if _minimal_app is None:
        store = ChromaStore()
        _minimal_app, _minimal_deps = build_minimal_supervisor_graph(store=store, _return_deps=True)

    app = _minimal_app
    init: GraphState = {
        "messages": [("user", query)],
        "user_input": query,
        "book_id": book_id,
        "selected_text": selected_text,    # type: ignore[typeddict-item]
        "current_chapter": current_chapter,# type: ignore[typeddict-item]
        # 每轮重置安全状态（强制重新检查）
        "safety_ok": None,                 # type: ignore[typeddict-item]
        # 每轮重置输出（防止跨轮 checkpoint 污染）
        "answer": "",
        "citations": [],
        "retrieved_docs_count": 0,
        "intent": "",
    }
    return app.invoke(init, config={"configurable": {"thread_id": thread_id}})  # type: ignore[return-value]
