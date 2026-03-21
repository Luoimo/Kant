from backend.agents.orchestrator_agent import (
    GraphState,
    IntentSchema,
    NoteOutputMeta,
    PlanOutputMeta,
    _extract_agent_last_turns,
)
from langchain_core.messages import AIMessage


def test_graphstate_has_per_agent_message_fields():
    annotations = GraphState.__annotations__
    for field in ("deepread_messages", "notes_messages", "plan_messages", "recommend_messages"):
        assert field in annotations, f"Missing field: {field}"


def test_graphstate_has_pipeline_fields():
    annotations = GraphState.__annotations__
    assert "pending_agents" in annotations
    assert "compound_context" in annotations
    assert "action" in annotations
    assert "notes_last_output" in annotations
    assert "plan_last_output" in annotations
    assert "plan_progress" in annotations


def test_intent_schema_defaults():
    schema = IntentSchema(intent="deepread", reason="test")
    assert schema.action == "new"
    assert schema.compound_intents == []
    assert schema.is_progress_update is False
    assert schema.notes_format is None
    assert schema.recommend_type is None
    assert schema.plan_type is None


def test_intent_schema_compound():
    schema = IntentSchema(
        intent="recommend",
        reason="compound test",
        compound_intents=["recommend", "plan"],
    )
    assert schema.compound_intents == ["recommend", "plan"]


def test_extract_agent_last_turns_empty_state():
    state = {}
    turns = _extract_agent_last_turns(state)
    assert turns == {"deepread": "", "notes": "", "plan": "", "recommend": ""}


def test_extract_agent_last_turns_with_messages():
    state = {
        "deepread_messages": [AIMessage(content="This is a deep read answer")],
        "notes_messages": [],
    }
    turns = _extract_agent_last_turns(state)
    assert turns["deepread"] == "This is a deep read answer"
    assert turns["notes"] == ""
    assert turns["plan"] == ""


def test_extract_agent_last_turns_truncates_to_300():
    state = {
        "recommend_messages": [AIMessage(content="x" * 500)],
    }
    turns = _extract_agent_last_turns(state)
    assert len(turns["recommend"]) == 300
