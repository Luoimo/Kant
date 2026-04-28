"""Typed container classes that group prompts by agent/module."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouterPrompts:
    system: str
    # For router's status line streamed to the frontend.
    intent_status: str
    # Label rendered in user-visible "tool is running" hints.
    tool_status_book: str
    tool_status_notes: str
    tool_status_default: str


@dataclass(frozen=True)
class DeepReadPrompts:
    system_base: str
    graph_aware_appendix: str
    # Labels injected into the system message's dynamic context block.
    ctx_current_book: str      # "Currently reading: {title}" with {title} placeholder
    ctx_book_source: str       # "Book source: {source}"
    ctx_current_chapter: str   # "Current chapter: {chapter}"
    ctx_selected_text: str     # "Selected excerpt: ..."
    ctx_memory: str            # "User reading history (reference only): ..."
    # Evidence block rendering used inside the search_book_content tool.
    evidence_header: str       # "[Evidence{i}] Book: {title}  Chapter: {location}"
    evidence_unknown_book: str
    graph_block_title: str     # "[Graph subgraph]"
    graph_seeds: str           # "Seed nodes: "
    graph_expanded: str        # "Expanded nodes: "
    graph_chapters: str        # "Related chapters: "
    graph_paths: str           # "Relation paths: "
    # Tool descriptions used by LangGraph / ReAct.
    tool_search_book_desc: str
    tool_search_notes_desc: str
    # Fallback string when retrieval is empty.
    no_results: str


@dataclass(frozen=True)
class CriticPrompts:
    system: str
    input_header_query: str    # "[User query]:"
    input_header_evidence: str # "[Retrieved evidence]:"
    input_header_answer: str   # "[AI answer]:"
    critic_note_title: str     # Heading prefix appended to user-visible review notes.
    critic_failed: str         # "Review failed: {err}"


@dataclass(frozen=True)
class FollowupPrompts:
    system: str
    input_template: str        # Uses {question} and {answer}


@dataclass(frozen=True)
class NoteExtractPrompts:
    extract_system: str
    extract_template: str          # Uses {question} and {answer}
    agent_system_template: str     # Uses {book_title} {question} {answer} {summary} {concepts}
    agent_task_suffix: str


@dataclass(frozen=True)
class ObsidianToolPrompts:
    read_past_desc: str
    search_vault_desc: str
    append_note_desc: str


@dataclass(frozen=True)
class RetrieverPrompts:
    query_rewriter_system: str
    llm_reranker_prompt_template: str   # uses {query} and {numbered}


@dataclass(frozen=True)
class PromptBundle:
    locale: str
    router: RouterPrompts
    deepread: DeepReadPrompts
    critic: CriticPrompts
    followup: FollowupPrompts
    note: NoteExtractPrompts
    obsidian: ObsidianToolPrompts
    retriever: RetrieverPrompts


__all__ = [
    "PromptBundle",
    "RouterPrompts",
    "DeepReadPrompts",
    "CriticPrompts",
    "FollowupPrompts",
    "NoteExtractPrompts",
    "ObsidianToolPrompts",
    "RetrieverPrompts",
]
