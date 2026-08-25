from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graph.graph_extractor import LLMGraphExtractor
from graph.hanlp_ner_llm_re_extractor import HanLPNerLLMReExtractor


class Doc:
    def __init__(self, text: str) -> None:
        self.page_content = text


def test_graph_normalization_helpers_cover_invalid_duplicates_and_aliases() -> None:
    extractor = LLMGraphExtractor()

    assert extractor._normalize_pair_rows("bad", left_key="from", right_key="to") == []
    pairs = extractor._normalize_pair_rows(
        [
            None,
            {"from": "", "to": "乙"},
            {"from": "甲", "to": "甲"},
            {"from": "甲", "to": "乙", "relation": "合作", "weight": 0},
        ],
        left_key="from",
        right_key="to",
        relation_key="relation",
    )
    assert pairs == [{"left": "甲", "right": "乙", "relation_type": "合作", "weight": 1}]

    assert extractor._normalize_name_list("bad", max_len=2, max_char=8) == []
    assert extractor._normalize_name_list(
        ["", "A", "English", "康德", "康德", "先验感性论", "超过长度的中文实体"],
        max_len=2,
        max_char=6,
    ) == ["康德", "先验感性论"]

    assert extractor._normalize_character_list(
        ["达西", "达西先生", "伊丽莎白"], max_len=4
    ) == ["达西先生", "伊丽莎白"]
    assert extractor._normalize_character_list(
        ["达西", "达西先生", "达西小姐"], max_len=4
    ) == ["达西", "达西先生", "达西小姐"]

    original = [{"left": "达西", "right": "伊丽莎白", "weight": 2}]
    assert extractor._apply_character_alias_to_pairs(original, alias_map={}) is original
    assert extractor._apply_character_alias_to_pairs(
        original + [{"left": "达西", "right": "达西先生"}],
        alias_map={"达西": "达西先生"},
    ) == [{"left": "达西先生", "right": "伊丽莎白", "weight": 2}]


@pytest.mark.parametrize(
    "content, expected",
    [
        ('{"concepts":["自由"]}', {"concepts": ["自由"]}),
        ('```json\n{"characters":["康德"]}\n```', {"characters": ["康德"]}),
        ('prefix {"events":["相遇"]} suffix', {"events": ["相遇"]}),
        ([{"text": '{"ok":true}'}], {"ok": True}),
    ],
)
def test_invoke_json_llm_parses_supported_response_shapes(monkeypatch, content, expected) -> None:
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(content=content)
    monkeypatch.setattr("llm.openai_client.get_llm", lambda **_: model)

    assert LLMGraphExtractor()._invoke_json_llm(model="test", prompt="prompt") == expected


def test_invoke_json_llm_retries_transient_errors_and_stops_on_invalid_output(monkeypatch) -> None:
    retrying = MagicMock()
    retrying.invoke.side_effect = [TimeoutError("timeout"), SimpleNamespace(content='{"ok":1}')]
    monkeypatch.setattr("llm.openai_client.get_llm", lambda **_: retrying)
    sleep = MagicMock()
    monkeypatch.setattr("graph.graph_extractor.time.sleep", sleep)

    assert LLMGraphExtractor()._invoke_json_llm(model="test", prompt="prompt") == {"ok": 1}
    sleep.assert_called_once()

    invalid = MagicMock()
    invalid.invoke.return_value = SimpleNamespace(content="not json")
    monkeypatch.setattr("llm.openai_client.get_llm", lambda **_: invalid)
    assert LLMGraphExtractor()._invoke_json_llm(model="test", prompt="prompt") == {}
    assert invalid.invoke.call_count == 1

    failing = MagicMock()
    failing.invoke.side_effect = RuntimeError("bad request")
    monkeypatch.setattr("llm.openai_client.get_llm", lambda **_: failing)
    assert LLMGraphExtractor()._invoke_json_llm(model="test", prompt="prompt") == {}


def test_graph_prompt_helpers_and_chapter_rows(monkeypatch) -> None:
    extractor = LLMGraphExtractor()
    invoke = MagicMock(
        side_effect=[
            {"concepts": ["自由"]},
            {"characters": ["康德", "English"]},
            {"dependencies": []},
        ]
    )
    monkeypatch.setattr(extractor, "_invoke_json_llm", invoke)

    assert extractor._extract_entities_with_ner_llm(chapter_text="", schema={}) == {
        "concepts": [], "characters": [], "events": []
    }
    assert extractor._extract_entities_with_ner_llm(
        chapter_text="自由是一种理念", schema={"mode": "learning", "use_concept": True}
    ) == {"concepts": ["自由"]}
    assert extractor._extract_characters_only_with_ner_llm(
        chapter_text="康德在写作", max_characters=3
    ) == ["康德"]
    assert extractor._extract_relations_with_re_llm(
        chapter_text="自由依赖理性",
        schema={"mode": "learning"},
        entities={"concepts": ["自由", "Reason"], "characters": [], "events": []},
    ) == {"dependencies": []}
    assert extractor._extract_characters_only_with_ner_llm(chapter_text=" ", max_characters=2) == []
    assert extractor._extract_relations_with_re_llm(chapter_text=" ", schema={}, entities={}) == {}

    rows, total = extractor._build_chapter_rows(
        [{"docs": [Doc("第一段"), Doc(""), Doc("第二段")]}, {"docs": []}]
    )
    assert total == 3
    assert rows == [
        {"chapter_idx": 0, "text": "第一段\n第二段"},
        {"chapter_idx": 1, "text": ""},
    ]


def test_parallel_ner_re_and_character_fallback(monkeypatch) -> None:
    extractor = LLMGraphExtractor()
    monkeypatch.setattr(
        extractor,
        "_extract_entities_with_ner_llm",
        lambda *, chapter_text, schema: {"characters": [chapter_text]} if chapter_text else {},
    )
    rows = [{"text": "张三"}, {"text": "李四"}]
    ner = extractor._run_ner_for_chapters(chapter_rows=rows, schema={})
    assert ner == [{"characters": ["张三"]}, {"characters": ["李四"]}]

    assert extractor._apply_character_fallback(
        chapter_rows=rows,
        ner_results=ner,
        schema={"use_character": False},
        max_concepts_per_chapter=3,
    ) == 0
    monkeypatch.setattr(
        extractor,
        "_extract_characters_only_with_ner_llm",
        lambda *, chapter_text, max_characters: [f"{chapter_text}先生"],
    )
    empty_ner = [{"characters": []}, {"characters": ["李四"]}]
    assert extractor._apply_character_fallback(
        chapter_rows=rows,
        ner_results=empty_ner,
        schema={"use_character": True},
        max_concepts_per_chapter=3,
    ) == 1
    assert empty_ner[0]["characters"] == ["张三先生"]

    monkeypatch.setattr(
        extractor,
        "_extract_relations_with_re_llm",
        lambda *, chapter_text, schema, entities: {"text": chapter_text, "entities": entities},
    )
    rel = extractor._run_re_for_chapters(chapter_rows=rows, schema={}, ner_results=ner)
    assert [item["text"] for item in rel] == ["张三", "李四"]


def test_build_graph_payloads_normalizes_complete_llm_result(monkeypatch) -> None:
    extractor = LLMGraphExtractor()
    chapters = [
        {"title": "导言", "order": 1, "docs": [Doc("康德讨论自由和理性")]} ,
        {"title": "第二章", "order": 2, "docs": [Doc("达西先生遇见伊丽莎白")]} ,
    ]
    monkeypatch.setattr(
        extractor,
        "_run_ner_for_chapters",
        lambda **_: [
            {"concepts": ["自由", "理性"], "characters": ["康德"], "events": ["讨论"]},
            {"concepts": [], "characters": ["达西", "达西先生", "伊丽莎白"], "events": ["相遇"]},
        ],
    )
    monkeypatch.setattr(extractor, "_apply_character_fallback", lambda **_: 0)
    monkeypatch.setattr(
        extractor,
        "_run_re_for_chapters",
        lambda **_: [
            {
                "concept_relations": [{"from": "自由", "to": "理性", "weight": 2}],
                "dependencies": [{"from": "自由", "to": "理性"}],
                "hierarchies": [{"child": "自由", "parent": "理性"}],
                "character_relations": [{"from": "康德", "to": "康德"}],
                "event_relations": [],
            },
            {
                "character_relations": [
                    {"from": "达西", "to": "伊丽莎白", "relation": "相识", "weight": 3}
                ],
                "event_relations": [{"prev": "相遇", "next": "交谈"}],
            },
        ],
    )

    payload = extractor.build_graph_payloads(
        chapters=chapters,
        schema={"mode": "hybrid", "use_concept": True, "use_character": True},
        max_concepts_per_chapter=8,
        max_events_per_chapter=5,
    )

    assert payload[0]["llm_dependency_pairs"][0]["relation_type"] == "depends_on"
    assert payload[1]["characters"] == ["达西先生", "伊丽莎白"]
    assert payload[1]["llm_character_pairs"] == [
        {"left": "达西先生", "right": "伊丽莎白", "relation_type": "相识", "weight": 3}
    ]
    assert extractor.build_graph_payloads(
        chapters=[], schema={}, max_concepts_per_chapter=1, max_events_per_chapter=1
    ) == []


def test_hanlp_client_initialization_and_cache(monkeypatch) -> None:
    fake_client = MagicMock()
    constructor = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "hanlp_restful", SimpleNamespace(HanLPClient=constructor))
    extractor = HanLPNerLLMReExtractor(api_url="", api_key="", language="", ner_task="")

    assert extractor._get_hanlp_client() is fake_client
    assert extractor._get_hanlp_client() is fake_client
    constructor.assert_called_once()


def test_hanlp_parsing_split_and_error_helpers() -> None:
    extractor = HanLPNerLLMReExtractor(
        api_url="https://hanlp", api_key="key", language="zh", ner_task="ner/msra"
    )
    assert extractor._split_text_for_hanlp("", max_chars=4) == []
    assert extractor._split_text_for_hanlp("短文本", max_chars=10) == ["短文本"]
    chunks = extractor._split_text_for_hanlp("甲乙丙。丁戊己。庚辛壬", max_chars=7)
    assert "".join(chunks) == "甲乙丙。丁戊己。庚辛壬"

    entries = extractor._iter_ner_entries(
        {
            "a": {"text": "张三", "label": "B-PER"},
            "b": [["北京", "LOC", 0, 2], ["invalid", "not label!"]],
        }
    )
    assert ("张三", "PER") in entries
    assert ("北京", "LOC") in entries

    parsed = extractor._parse_hanlp_ner(
        doc={"ner/msra": [["张三", "PERSON"], ["北京", "LOC"], ["A", "ORG"], ["张三", "PER"]]},
        schema={"use_character": True, "use_concept": True},
    )
    assert parsed == {"concepts": ["北京"], "characters": ["张三"], "events": []}
    assert extractor._is_error_doc({"msg": "auth missing", "code": 500}) is True
    assert extractor._is_error_doc({"msg": ""}) is False
    assert extractor._is_error_doc("ok") is False
    assert extractor._extract_error_msg({"error": "bad"}) == "bad"
    assert extractor._extract_error_msg("bad") == "bad"
    extractor._log_empty_ner_debug(chapter_idx=1, stage="test", raw_doc={"x": object()})


def test_hanlp_entity_extraction_primary_fallback_error_and_exception(monkeypatch) -> None:
    extractor = HanLPNerLLMReExtractor(
        api_url="https://hanlp", api_key="key", language="zh", ner_task="ner/msra"
    )

    primary = MagicMock(return_value={"ner/msra": [["张三", "PERSON"], ["北京", "LOC"]]})
    monkeypatch.setattr(extractor, "_get_hanlp_client", lambda: primary)
    assert extractor._extract_entities_with_ner_llm(
        chapter_text="张三在北京", schema={"use_character": True, "use_concept": True}, chapter_idx=0
    ) == {"concepts": ["北京"], "characters": ["张三"], "events": [], "__hanlp_ok": True}

    def fallback_client(text, *, tasks, language):
        if tasks == "ner/msra":
            return {"ner/msra": []}
        return {"ner*": [["李四", "PERSON"], ["上海", "LOC"]]}

    monkeypatch.setattr(extractor, "_get_hanlp_client", lambda: fallback_client)
    fallback = extractor._extract_entities_with_ner_llm(
        chapter_text="李四在上海", schema={"use_character": True, "use_concept": True}
    )
    assert fallback["characters"] == ["李四"]
    assert fallback["concepts"] == ["上海"]

    monkeypatch.setattr(
        extractor,
        "_get_hanlp_client",
        lambda: lambda *args, **kwargs: {"msg": "auth required", "code": 500},
    )
    assert extractor._extract_entities_with_ner_llm(
        chapter_text="文本", schema={}, chapter_idx=2
    )["__hanlp_ok"] is False

    monkeypatch.setattr(extractor, "_get_hanlp_client", MagicMock(side_effect=RuntimeError("down")))
    assert extractor._extract_entities_with_ner_llm(
        chapter_text="文本", schema={}, chapter_idx=3
    )["__hanlp_ok"] is False
    assert extractor._extract_entities_with_ner_llm(chapter_text=" ", schema={})["__hanlp_ok"] is True


def test_hanlp_parallel_runner_and_disabled_llm_fallback(monkeypatch) -> None:
    extractor = HanLPNerLLMReExtractor(
        api_url="https://hanlp", api_key="key", language="zh", ner_task="ner/msra"
    )
    monkeypatch.setattr(
        extractor,
        "_extract_entities_with_ner_llm",
        lambda *, chapter_text, schema, chapter_idx: {
            "concepts": [chapter_text] if chapter_idx == 0 else [],
            "characters": [],
            "events": [],
            "__hanlp_ok": chapter_idx == 0,
        },
    )
    rows = extractor._run_ner_for_chapters(
        chapter_rows=[{"text": "自由"}, {"text": "理性"}], schema={}
    )
    assert rows[0]["concepts"] == ["自由"]
    assert rows[1]["__hanlp_ok"] is False
    assert extractor._apply_character_fallback(
        chapter_rows=[], ner_results=[], schema={}, max_concepts_per_chapter=3
    ) == 0
