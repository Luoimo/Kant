from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, Annotated
import sys

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3 as _sqlite3
    _SQLITE_AVAILABLE = True
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as SqliteSaver  # type: ignore[assignment]
    _SQLITE_AVAILABLE = False
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field

from pathlib import Path

from backend.config import get_settings
from backend.agents.deepread_agent import DeepReadAgent, deepread_node
from backend.agents.note_agent import NoteAgent, notes_node
from backend.agents.reading_plan_agent import ReadingPlanAgent, plan_node
from backend.agents.recommendation_agent import RecommendationAgent, recommend_node
from backend.llm.openai_client import get_llm
from backend.memory.mem0_store import Mem0Store
from backend.rag.chroma.chroma_store import ChromaStore
from backend.security.input_filter import InputSafetyResult, run_input_safety_check
from backend.storage import make_note_storage, make_plan_storage
from backend.storage.note_storage import LocalNoteStorage, NoteStorage
from backend.storage.plan_storage import LocalPlanStorage, PlanStorage


# 依赖注入：不可序列化依赖（子 agent 等）通过 GraphDeps 注入节点，不写入 state，避免 checkpoint 序列化报错。


@dataclass
class GraphDeps:
    """图内依赖容器，仅通过闭包传入节点，不写入 state。"""
    deepread_agent: DeepReadAgent
    notes_agent: NoteAgent
    plan_agent: ReadingPlanAgent
    recommend_agent: RecommendationAgent
    mem0: Mem0Store | None = None
    note_storage: NoteStorage = field(default_factory=lambda: LocalNoteStorage(Path("data/notes")))
    plan_storage: PlanStorage = field(default_factory=lambda: LocalPlanStorage(Path("data/plans")))


class NoteOutputMeta(TypedDict, total=False):
    note_id: str
    book_title: str
    topics: list
    storage_path: str
    created_at: str  # ISO 8601


class PlanOutputMeta(TypedDict, total=False):
    plan_id: str
    book_titles: list
    plan_type: str
    storage_path: str
    created_at: str  # ISO 8601
    progress_summary: str


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

    # ── New: per-agent message lists ────────────────────────────────
    deepread_messages:   Annotated[list[AnyMessage], add_messages]
    notes_messages:      Annotated[list[AnyMessage], add_messages]
    plan_messages:       Annotated[list[AnyMessage], add_messages]
    recommend_messages:  Annotated[list[AnyMessage], add_messages]

    # ── New: multi-agent pipeline ────────────────────────────────────
    pending_agents:   list[str]
    compound_context: str | None

    # ── New: action type ─────────────────────────────────────────────
    action: Literal["new", "edit", "extend"] | None

    # ── New: structured output metadata pointers ─────────────────────
    notes_last_output: NoteOutputMeta | None
    plan_last_output:  PlanOutputMeta | None

    # ── New: progress tracking ───────────────────────────────────────
    plan_progress: Annotated[list[str], lambda a, b: a + b]

    # ── Reader mode context ──────────────────────────────────────────
    # 前端传入：当前激活的 tab、用户划选的原文、当前阅读章节
    active_tab: Literal["deepread", "notes", "plan", "recommend"] | None
    selected_text: str | None   # 用户在 EPUB 阅读器中划选的原文片段
    current_chapter: str | None # 当前阅读章节标题，供 deepread 注入上下文


# ---------------------------------------------------------------------------
# 意图识别（结构化输出）
# ---------------------------------------------------------------------------


class IntentSchema(BaseModel):
    intent: Literal["recommend", "deepread", "notes", "plan"] = Field(
        description="当前用户请求的主要意图类型。"
    )
    action: Literal["new", "edit", "extend"] = Field(
        default="new",
        description="操作类型：new=全新任务, edit=修改已有输出, extend=扩展已有输出"
    )
    reason: str = Field(description="一句话解释为什么这么判断这个意图。")
    book_source: str | None = Field(
        default=None,
        description="如果用户显式提到了某本书/文件路径，这里给出它；否则为 None。",
    )
    compound_intents: list[str] = Field(
        default_factory=list,
        description="复合请求时按顺序排列的所有 agent，如 ['recommend','plan']；单步请求留空。"
    )
    notes_format: Literal["structured", "summary", "qa", "timeline"] | None = Field(
        default=None, description="笔记格式（仅 notes 意图时有效）"
    )
    recommend_type: Literal["discover", "similar", "next", "theme"] | None = Field(
        default=None, description="推荐子类型（仅 recommend 意图时有效）"
    )
    plan_type: Literal["single_deep", "multi_theme", "research"] | None = Field(
        default=None, description="计划子类型（仅 plan 意图时有效）"
    )
    is_progress_update: bool = Field(
        default=False, description='True 当用户报告阅读进度（"XX我读完了/已读"）'
    )


# Reader 模式短路路由辅助
_COMPOUND_SIGNALS = {"并且", "同时", "另外还", "还要", "并帮我", "也帮我", "以及", "顺便", "同时帮我"}


def _has_compound_signals(text: str) -> bool:
    """检测用户输入是否含跨 Agent 复合意图信号（如「分析并做笔记」）。"""
    return any(kw in text for kw in _COMPOUND_SIGNALS)


def _infer_action_from_text(text: str) -> str:
    """从文本启发式推断 action 类型，避免 reader 模式短路时多一次 LLM call。"""
    if any(kw in text for kw in ("修改", "更新", "调整", "改成", "改为", "把", "重写")):
        return "edit"
    if any(kw in text for kw in ("再加", "继续", "补充", "扩展", "追加", "新增")):
        return "extend"
    return "new"


def _extract_agent_last_turns(state: "GraphState") -> dict[str, str]:
    """Extract the last AIMessage content from each agent's messages list."""
    result: dict[str, str] = {}
    for agent in ("deepread", "notes", "plan", "recommend"):
        msgs = state.get(f"{agent}_messages") or []
        last_ai = ""
        for m in reversed(msgs):
            if getattr(m, "type", None) == "ai":
                last_ai = (getattr(m, "content", "") or "")[:300]
                break
        result[agent] = last_ai
    return result


def classify_intent(
    user_input: str,
    agent_last_turns: dict[str, str] | None = None,
) -> IntentSchema:
    """
    使用 LLM 做一次结构化意图识别，将请求归类为
    {recommend, deepread, notes, plan} 之一。
    支持复合意图检测和 action 类型识别。
    """
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(IntentSchema)
    turns = agent_last_turns or {}

    prompt = (
        "你是一个意图区分器，只负责判断下面这句中文请求的类型，"
        "范围限定在读书相关：找书推荐、针对一本书的精读或问答、"
        "整理读书笔记、制定阅读计划。\n\n"
        "请根据用户文本填写 IntentSchema：\n"
        "- recommend：请求推荐/发现小众书\n"
        "- deepread：围绕某本书的章节/概念进行精读、解释、问答或基于证据的回答（含自由问答）\n"
        "- notes：整理/总结/结构化读书笔记\n"
        "- plan：制定或调整阅读书单/节奏/路线\n\n"
        f"可参考的子 Agent 最近输出摘要（判断用户是否在引用上一轮结果）：\n"
        f"- deepread 最近回复：{turns.get('deepread', '')}\n"
        f"- notes 最近回复：{turns.get('notes', '')}\n"
        f"- plan 最近回复：{turns.get('plan', '')}\n"
        f"- recommend 最近回复：{turns.get('recommend', '')}\n\n"
        "额外判断规则：\n"
        '- 如果用户一句话要求多件事（如【推荐...并制定计划】），compound_intents 填完整链路，如 ["recommend","plan"]\n'
        "- 如果用户说【修改/更新/调整/把...改成...】，action=edit\n"
        "- 如果用户说【再加/继续/补充...】，action=extend\n"
        "- 如果用户说【XX章节我读完了/已读】，is_progress_update=true，intent=plan\n"
        "- 不支持动态扇出（如【每本书都做笔记】），compound_intents 留空，在回复中说明需要分步操作\n\n"
        f"用户输入：{user_input!r}"
    )

    return structured_llm.invoke(prompt)


def supervisor_node(state: GraphState) -> dict:
    """
    Supervisor（返回 delta dict，不修改完整 state）：
    - 安全检查 → 意图识别 → compound pipeline 路由 → agent 分发
    - agent 完成后返回 supervisor → finalize
    """
    print("[Graph] → supervisor_node", file=sys.stdout)

    patch: dict = {}

    # 从 messages 中提取最近一条 human 消息内容；兼容 user_input 字段
    user_input = state.get("user_input", "") or ""
    msgs = state.get("messages") or []
    for m in reversed(msgs):
        if getattr(m, "type", None) == "human":
            user_input = getattr(m, "content", "") or user_input
            break

    # 1) 安全检查：每轮新请求均重新检查（safety_ok=None 表示本轮尚未检查）
    #    同一轮内 supervisor 被多次调用时（agent 跑完回来），safety_ok 已是 True/False，跳过
    if state.get("safety_ok") is None:
        safety_result: InputSafetyResult = run_input_safety_check(user_input)
        patch["safety_ok"] = safety_result.allowed
        patch["safety_reason"] = safety_result.reason
        print(
            f"[Supervisor] safety_ok={safety_result.allowed}, "
            f"categories={safety_result.categories}, "
            f"reason={safety_result.reason}",
            file=sys.stdout,
        )
        if not safety_result.allowed:
            patch["answer"] = f"当前请求未通过安全检查：{safety_result.reason}"
            patch["next"] = "finalize"
            return patch

    # 2) 意图识别（只做一次）
    intent_result: IntentSchema | None = None
    if state.get("intent") is None:
        active_tab = state.get("active_tab")
        # Reader 模式短路：active_tab 已明确且无复合意图信号 → 跳过 LLM 分类
        if active_tab and not _has_compound_signals(user_input):
            _TAB_INTENT = {"deepread": "deepread", "notes": "notes",
                           "plan": "plan", "recommend": "recommend"}
            tab_intent = _TAB_INTENT.get(active_tab, "deepread")
            patch["intent"] = tab_intent
            patch["action"] = _infer_action_from_text(user_input)
            patch["intent_reason"] = f"reader mode: active_tab={active_tab} → direct route"
            print(
                f"[Supervisor] short-circuit route: active_tab={active_tab} → intent={tab_intent}",
                file=sys.stdout,
            )
        else:
            # Library 模式或含复合意图信号 → LLM 意图分类
            agent_last_turns = _extract_agent_last_turns(state)
            intent_result = classify_intent(user_input, agent_last_turns)
            patch["intent"] = intent_result.intent
            patch["action"] = intent_result.action
            patch["intent_reason"] = intent_result.reason
            if not state.get("book_source") and intent_result.book_source:
                patch["book_source"] = intent_result.book_source
            print(
                f"[Supervisor] intent={intent_result.intent}, action={intent_result.action}, "
                f"compound_intents={intent_result.compound_intents}, "
                f"reason={intent_result.reason}",
                file=sys.stdout,
            )

    # 3) agent 已产出 answer 且没有剩余待执行的 agent → finalize
    #    注意：必须先检查 pending_agents，否则复合流水线第一个 agent 跑完后就会
    #    提前结束，后续 agent 永远不会被调度。
    if state.get("answer") and not (state.get("pending_agents") or []):
        patch["next"] = "finalize"
        return patch

    # 4) 确定本次分发意图
    effective_intent: str = patch.get("intent") or state.get("intent") or "deepread"

    # 5) Compound pipeline：首次识别时若有复合意图，填充 pending_agents
    pending = list(state.get("pending_agents") or [])
    if intent_result is not None and intent_result.compound_intents and not pending:
        pending = list(intent_result.compound_intents)
        patch["compound_context"] = None

    # 6) 决定本次分发给哪个 agent
    if pending:
        target = pending.pop(0)
        patch["pending_agents"] = pending   # 消费一个
    else:
        target = effective_intent

    # 7) 构造 task message（注入 compound_context / selected_text / current_chapter）
    task_content = user_input
    selected_text = state.get("selected_text") or ""
    current_chapter = state.get("current_chapter") or ""
    if selected_text:
        task_content = f"【用户划选的原文片段】：\n{selected_text}\n\n【用户问题】：\n{task_content}"
    if current_chapter:
        task_content += f"\n\n【当前阅读章节】：{current_chapter}"
    ctx = state.get("compound_context")
    if ctx:
        task_content += f"\n\n【前序步骤结果，供参考】：\n{ctx}"
    task_msg = HumanMessage(content=task_content)

    # 8) 分发到目标 agent
    book_source = patch.get("book_source") or state.get("book_source")

    ctx_so_far = state.get("compound_context") or ""

    if target == "notes":
        if not book_source:
            book_source = _extract_recommended_book(ctx_so_far)
        patch["notes_messages"] = [task_msg]
        patch["notes_query"] = user_input
        patch["notes_book_source"] = book_source
        patch["next"] = "notes"
    elif target == "plan":
        if not book_source:
            book_source = _extract_recommended_book(ctx_so_far)
        patch["plan_messages"] = [task_msg]
        patch["plan_query"] = user_input
        patch["plan_book_source"] = book_source
        patch["next"] = "plan"
    elif target == "recommend":
        patch["recommend_messages"] = [task_msg]
        patch["recommend_query"] = user_input
        patch["next"] = "recommend"
    else:
        patch["deepread_messages"] = [task_msg]
        patch["deepread_query"] = user_input
        patch["deepread_book_source"] = book_source
        patch["next"] = "deepread"

    # 9) 进度更新 override：仅当当前轮没有待执行的复合 agent 时才覆盖路由
    #    否则会打断正在进行的复合流水线（如 ["deepread","notes","plan"] 的中间环节）
    if intent_result is not None and intent_result.is_progress_update and not pending:
        patch["plan_messages"] = [task_msg]
        patch["plan_query"] = user_input
        patch["next"] = "plan"

    return patch


_AGENT_SECTION_RE = r'\[(?:推荐|计划|笔记|精读)结果\]'


def _extract_recommended_book(ctx: str) -> str | None:
    """从 compound_context 中提取前序 recommend 推荐的第一本书名。"""
    if not ctx or "[推荐结果]" not in ctx:
        return None
    m = re.search(r"###\s+《(.+?)》", ctx) or re.search(r"《(.+?)》", ctx)
    return m.group(1) if m else None


def _synthesize_compound_answer(user_input: str, compound_ctx: str, last_answer: str) -> str:
    """当多个 agent 依次运行时，将各步结果整合为一份完整回答。"""
    llm = get_llm(temperature=0.3)
    prompt = (
        f"用户请求：{user_input}\n\n"
        f"多个 Agent 已依次完成任务，结果如下：\n{compound_ctx}\n\n"
        "请将以上多步结果整合为一份完整、连贯的 Markdown 回答。"
    )
    msg = llm.invoke([{"role": "user", "content": prompt}])
    return getattr(msg, "content", str(msg)) or last_answer


def build_minimal_supervisor_graph(*, store: ChromaStore | None = None, enable_memory: bool = True, _return_deps: bool = False):
    """
    构建 supervisor 编排图：

    START -> memory_search -> supervisor -> [deepread|notes|plan|recommend] -> supervisor -> finalize -> END

    所有节点函数返回 delta dict（不修改完整 state），由 LangGraph 合并。
    """
    if store is None:
        settings = get_settings()
        store = ChromaStore()

    settings = get_settings()
    mem0 = Mem0Store() if enable_memory else None

    deps = GraphDeps(
        deepread_agent=DeepReadAgent(store=store),
        notes_agent=NoteAgent(store=store),
        plan_agent=ReadingPlanAgent(store=store),
        recommend_agent=RecommendationAgent(store=store),
        mem0=mem0,
        note_storage=make_note_storage(settings),
        plan_storage=make_plan_storage(settings),
    )

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

    def _finalize(state: GraphState) -> dict:
        print("[Graph] → finalize_node", file=sys.stdout)
        patch: dict = {"next": "end"}
        answer = state.get("answer") or ""
        ctx = state.get("compound_context") or ""
        # 多 agent 运行时（compound_context 含多个 agent 输出标记）→ 合成统一回答
        if ctx and len(re.findall(_AGENT_SECTION_RE, ctx)) > 1:
            answer = _synthesize_compound_answer(
                state.get("user_input", ""), ctx, answer
            )
            patch["answer"] = answer
        if answer:
            patch["messages"] = [AIMessage(content=answer)]
        if deps.mem0 and answer and state.get("user_input"):
            deps.mem0.add_qa(state["user_input"], answer)
            print("[Memory] 已保存本次问答到 Mem0", file=sys.stdout)
        return patch

    def _deepread(state: GraphState) -> dict:
        print("[Graph] → deepread_node", file=sys.stdout)
        patch = deepread_node(state, agent=deps.deepread_agent)  # type: ignore[arg-type]
        print(
            f"[Graph]   deepread_node done, retrieved_docs_count={patch.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return patch

    def _notes(state: GraphState, config) -> dict:
        print("[Graph] → notes_node", file=sys.stdout)
        thread_id = config["configurable"].get("thread_id", "default")
        patch = notes_node(state, agent=deps.notes_agent, deps=deps, thread_id=thread_id)  # type: ignore[arg-type]
        print(
            f"[Graph]   notes_node done, retrieved_docs_count={patch.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return patch

    def _plan(state: GraphState, config) -> dict:
        print("[Graph] → plan_node", file=sys.stdout)
        thread_id = config["configurable"].get("thread_id", "default")
        patch = plan_node(state, agent=deps.plan_agent, deps=deps, thread_id=thread_id)  # type: ignore[arg-type]
        print(
            f"[Graph]   plan_node done, retrieved_docs_count={patch.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return patch

    def _recommend(state: GraphState) -> dict:
        print("[Graph] → recommend_node", file=sys.stdout)
        patch = recommend_node(state, agent=deps.recommend_agent)  # type: ignore[arg-type]
        print(
            f"[Graph]   recommend_node done, retrieved_docs_count={patch.get('retrieved_docs_count', 0)}",
            file=sys.stdout,
        )
        return patch

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

    # SqliteSaver 持久化 checkpointer（降级到 MemorySaver 若包未安装）
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


_minimal_app = None
_minimal_deps: GraphDeps | None = None


def invalidate_bm25_caches() -> None:
    """入库新书后调用，清除 NoteAgent / ReadingPlanAgent 的 BM25 索引缓存。
    下次查询时会从 ChromaDB 重新拉取全量文档（含新书）构建索引。"""
    if _minimal_deps is None:
        return
    for agent in (_minimal_deps.notes_agent, _minimal_deps.plan_agent):
        retriever = getattr(agent, "_retriever", None)
        if retriever is not None:
            retriever.invalidate_bm25()


def run_minimal_graph(
    query: str,
    *,
    book_source: str | None = None,
    thread_id: str = "default",
    active_tab: str | None = None,
    selected_text: str | None = None,
    current_chapter: str | None = None,
) -> GraphState:
    """
    支持多轮对话的入口：
    - 相同 thread_id 下，多次调用会共享同一条对话历史（messages）。
    - 不同 thread_id 互相隔离。

    Reader 模式专用参数：
    - active_tab: 前端当前激活的 tab（deepread/notes/plan/recommend），
      有值时跳过 LLM 意图分类，直接路由到对应 Agent。
    - selected_text: 用户在 EPUB 阅读器中划选的原文片段，注入为问题上下文。
    - current_chapter: 当前阅读章节，注入为问题上下文。
    """
    global _minimal_app, _minimal_deps
    if _minimal_app is None:
        store = ChromaStore()
        _minimal_app, _minimal_deps = build_minimal_supervisor_graph(store=store, _return_deps=True)

    app = _minimal_app
    init: GraphState = {
        "messages": [("user", query)],
        "user_input": query,
        "book_source": book_source,
        "active_tab": active_tab,        # type: ignore[typeddict-item]
        "selected_text": selected_text,  # type: ignore[typeddict-item]
        "current_chapter": current_chapter,  # type: ignore[typeddict-item]
        # ── 每轮重置瞬态字段，防止跨轮 checkpoint 污染 ──────────────────
        # intent/action：强制重新分类，避免旧 action（如 edit）污染新轮意图
        "intent": None,
        "action": None,
        # answer：清空上轮结果，让 supervisor 正确判断当前轮是否已完成
        "answer": "",
        # pending_agents：清空上轮复合流水线队列，防止中断后的残余队列
        "pending_agents": [],
        # compound_context：每轮独立积累，不携带上轮内容
        "compound_context": None,
        # safety_ok：None 表示本轮尚未做安全检查，supervisor 会重新执行
        "safety_ok": None,  # type: ignore[typeddict-item]
    }
    return app.invoke(init, config={"configurable": {"thread_id": thread_id}})  # type: ignore[return-value]
