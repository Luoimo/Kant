from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import graph.neo4j_store as neo4j_module
from graph.neo4j_store import Neo4jStore


class FakeResult(list):
    def __init__(self, rows=()) -> None:
        super().__init__(rows)
        self.consumed = 0

    def consume(self):
        self.consumed += 1
        return self


class FakeSession:
    def __init__(self, responses=None, fail: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.fail = fail
        self.calls: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, params=None):
        self.calls.append((str(query), params))
        if self.fail:
            raise self.fail
        if self.responses:
            response = self.responses.pop(0)
            return response if isinstance(response, FakeResult) else FakeResult(response)
        return FakeResult()


class FakeDriver:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    def session(self):
        return self._session


class Doc:
    def __init__(self, text: str, metadata: dict | None = None) -> None:
        self.page_content = text
        self.metadata = metadata or {}


def make_store(*, enabled: bool = True, session: FakeSession | None = None) -> Neo4jStore:
    store = object.__new__(Neo4jStore)
    store._database = "neo4j"
    store._enabled = enabled
    store._driver = FakeDriver(session or FakeSession()) if enabled else None
    store._llm_graph_extractor = MagicMock()
    return store


def test_init_handles_disabled_and_connected_configurations(monkeypatch) -> None:
    settings = SimpleNamespace(
        neo4j_database="neo4j",
        graph_extractor_backend="llm",
        hanlp_api_url="",
        hanlp_api_key="",
        hanlp_language="zh",
        hanlp_ner_task="ner/msra",
        neo4j_uri="",
        neo4j_user="",
        neo4j_password="",
    )
    monkeypatch.setattr("config.get_settings", lambda: settings)
    disabled = Neo4jStore()
    assert disabled._enabled is False

    driver = MagicMock()
    graph_database = MagicMock()
    graph_database.driver.return_value = driver
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))
    settings.graph_extractor_backend = "hanlp_ner_llm_re"
    settings.neo4j_uri = "neo4j+s://example"
    settings.neo4j_user = "user"
    settings.neo4j_password = "secret"
    connected = Neo4jStore()
    assert connected._enabled is True
    driver.verify_connectivity.assert_called_once()

    graph_database.driver.side_effect = RuntimeError("offline")
    failed = Neo4jStore()
    assert failed._enabled is False
    assert failed._driver is None


def test_book_write_delete_and_graph_orchestration(monkeypatch) -> None:
    session = FakeSession()
    store = make_store(session=session)
    store.upsert_book(
        book_id="b1", title="Kant", author="康德", source="book.epub", total_chunks=4
    )
    assert len(session.calls) == 2
    assert "MERGE (b:Book" in session.calls[0][0]
    assert "MERGE (a:Author" in session.calls[1][0]

    no_author = FakeSession()
    store._driver = FakeDriver(no_author)
    store.upsert_book(
        book_id="b2", title="Untitled", author=" ", source="x", total_chunks=0
    )
    assert len(no_author.calls) == 1

    store._enabled = False
    store.upsert_book(book_id="b3", title="x", author="", source="x", total_chunks=0)
    store.delete_book(book_id="b3")

    graph_session = FakeSession()
    store = make_store(session=graph_session)
    payload = (
        {"use_concept": True, "use_character": True},
        [{"title": "导言", "concepts": ["自由"], "characters": ["康德"]}],
        [{"left": "自由", "right": "理性"}],
        [{"left": "康德", "right": "黑格尔"}],
        [],
        [],
        [],
    )
    monkeypatch.setattr(store, "_prepare_graph_payloads", MagicMock(return_value=payload))
    monkeypatch.setattr(store, "_clear_book_graph_scope", MagicMock())
    monkeypatch.setattr(store, "_upsert_chapter_layer", MagicMock())
    monkeypatch.setattr(store, "_upsert_entity_embeddings_layer", MagicMock())
    monkeypatch.setattr(store, "_upsert_related_pairs_layer", MagicMock())
    store.upsert_book_graph(book_id="b1", documents=[Doc("text")])
    store._upsert_chapter_layer.assert_called_once()
    store._upsert_related_pairs_layer.assert_called_once()

    monkeypatch.setattr(
        store,
        "_prepare_graph_payloads",
        MagicMock(return_value=({"use_concept": True}, [], [], [], [], [], [])),
    )
    store.upsert_book_graph(book_id="b1", documents=[])

    delete_session = FakeSession()
    store._driver = FakeDriver(delete_session)
    monkeypatch.setattr(store, "_clear_book_graph_scope", MagicMock())
    store.delete_book(book_id="b1")
    assert len(delete_session.calls) == 5


def test_prepare_payloads_collects_and_merges_relationships(monkeypatch) -> None:
    store = make_store()
    schema = {"use_concept": True, "use_character": True}
    chapters = [
        {
            "llm_concept_pairs": [
                {"left": "自由", "right": "理性", "relation_type": "related", "weight": 2},
                {"left": "自由", "right": "理性", "relation_type": "related", "weight": 1},
                {"left": "", "right": "理性"},
            ],
            "llm_character_pairs": [{"left": "甲", "right": "乙", "weight": 1}],
            "llm_event_pairs": [],
            "llm_dependency_pairs": [],
            "llm_hierarchy_pairs": [],
        }
    ]
    monkeypatch.setattr(store, "_select_graph_schema", lambda documents: schema)
    monkeypatch.setattr(store, "_build_graph_payloads", lambda *args, **kwargs: chapters)

    result = store._prepare_graph_payloads(
        documents=[Doc("text")], max_concepts_per_chapter=5, max_events_per_chapter=3
    )
    assert result[0] == schema
    assert result[2] == [
        {"left": "自由", "right": "理性", "relation_type": "related", "weight": 3}
    ]
    assert store._collect_chapter_pairs([], key="x") == []
    assert store._merge_pair_rows([]) == []


def test_cypher_layers_execute_expected_queries(monkeypatch) -> None:
    session = FakeSession()
    store = make_store(session=session)
    chapters = [{"title": "导言", "concepts": ["自由"], "characters": ["康德"], "events": ["写作"]}]
    schema = {"use_concept": True, "use_character": True}

    store._clear_book_graph_scope(session=session, book_id="b1")
    assert len(session.calls) == 6
    store._upsert_chapter_layer(session=session, book_id="b1", chapters=[], schema=schema)
    store._upsert_chapter_layer(session=session, book_id="b1", chapters=chapters, schema=schema)
    assert len(session.calls) == 11

    monkeypatch.setattr(
        store,
        "_build_entity_embedding_rows",
        lambda names: [{"name": name, "embedding": [0.1]} for name in names],
    )
    store._upsert_entity_embeddings_layer(
        session=session, book_id="b1", chapters=chapters, schema=schema
    )
    assert any("MATCH (n:Concept" in query for query, _ in session.calls)
    assert any("MATCH (e:Event" in query for query, _ in session.calls)

    Neo4jStore._upsert_related_pairs_layer(
        session=session,
        book_id="b1",
        concept_pairs=[{"left": "自由", "right": "理性", "weight": 1}],
        character_pairs=[{"left": "康德", "right": "黑格尔", "weight": 1}],
        event_pairs=[{"left": "写作", "right": "出版", "weight": 1}],
        dependency_pairs=[{"left": "自由", "right": "理性", "weight": 1}],
        hierarchy_pairs=[{"left": "范畴", "right": "概念", "weight": 1}],
        schema=schema,
    )
    assert any("DEPENDS_ON" in query for query, _ in session.calls)
    assert any("SUBCONCEPT_OF" in query for query, _ in session.calls)
    assert any("NEXT_EVENT" in query for query, _ in session.calls)


def test_schema_text_pair_and_title_helpers() -> None:
    store = make_store()
    learning = store._select_graph_schema([Doc("因为定义因此得到结论")])
    fiction = store._select_graph_schema([Doc("“你好，”张三说道。他问，李四回答。")])
    hybrid = store._select_graph_schema([Doc("普通中文文本")])
    assert learning["mode"] == "learning"
    assert fiction["mode"] == "fiction"
    assert hybrid["mode"] == "hybrid"
    assert store._analyze_text_features("“中文”") == {"dialogue_marks": 2, "chinese_chars": 2}
    assert store._analyze_relation_distribution(["因为所以", "说道并回答"]) == {
        "logic_rel": 2, "dialog_rel": 2
    }

    rows = store._collect_chapter_pairs(
        [{"pairs": [{"left": "甲", "right": "乙", "weight": 0}, {"left": "甲", "right": "甲"}]}],
        key="pairs",
    )
    assert rows == [{"left": "甲", "right": "乙", "weight": 1, "relation_type": "related"}]
    assert store._merge_pair_rows(rows + rows)[0]["weight"] == 2

    assert store._is_noise_text("") is True
    assert store._is_noise_text("作者") is True
    assert store._is_noise_text("真的吗") is True
    assert store._is_noise_text("因此得到") is True
    assert store._is_noise_text("123") is True
    assert store._is_noise_text("参考文献") is True
    assert store._is_noise_text("先验感性论") is False
    assert store._normalize_title("第一章：先验感性论") == "先验感性论"
    assert store._normalize_title("第一章") == "第一章"
    assert store._normalize_title("目录") == ""


def test_graph_retrieval_and_multihop_helpers(monkeypatch) -> None:
    store = make_store()
    empty = {"seed_entities": [], "expanded_entities": [], "chapter_titles": [], "reasoning_paths": []}
    store._enabled = False
    assert store.graph_retrieve_chunks(book_id="b1", query_terms=["自由"]) == empty

    session = FakeSession(
        responses=[
            [{"name": "理性", "relation": "RELATED_TO", "weight": 2}],
            [{"name": "范畴", "relation": "DEPENDS_ON", "weight": 3}],
            [{"chapter_title": "导言"}, {"chapter_title": ""}],
        ]
    )
    store = make_store(session=session)
    monkeypatch.setattr(store, "_resolve_seed_entities", lambda **_: ["自由"])
    result = store.graph_retrieve_chunks(book_id="b1", query_terms=[" 自由 "])
    assert result["seed_entities"] == ["自由"]
    assert result["expanded_entities"] == ["范畴", "理性"]
    assert result["chapter_titles"] == ["导言"]
    assert result["reasoning_paths"] == ["hop1:RELATED_TO:理性", "hop2:DEPENDS_ON:范畴"]

    seed_store = make_store()
    monkeypatch.setattr(seed_store, "_match_entities_vector", MagicMock(side_effect=[["自由"], ["康德", "自由"]]))
    monkeypatch.setattr(seed_store, "_match_events_vector", MagicMock(return_value=["写作", "康德"]))
    assert seed_store._resolve_seed_entities(
        session=session, terms=["x"], book_id="b1", seed_top_k=3
    ) == ["自由", "康德", "写作"]

    assert store._expand_entities_once(
        session=session, source_entities=[], book_id="b1", expand_top_k=2
    ) == []
    weighted = store._build_weighted_entity_rows(
        seed_entities=["自由"],
        expanded_pairs=[{"name": "自由", "weight": 5}, {"name": "理性", "weight": 0}],
    )
    assert weighted == [{"name": "自由", "weight": 5.0}, {"name": "理性", "weight": 1.0}]


def test_vector_embedding_and_entity_helpers(monkeypatch) -> None:
    store = make_store()
    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2]
    embeddings.embed_documents.side_effect = lambda names: [[float(i)] for i, _ in enumerate(names, 1)]
    monkeypatch.setattr("llm.openai_client.get_embeddings", lambda: embeddings)

    for label, book_id in (("Concept", "b1"), ("Concept", ""), ("Character", "b1"), ("Character", "")):
        session = FakeSession(responses=[[{"name": label}, {"name": ""}]])
        assert store._match_entities_vector(
            session=session, label=label, terms=["自由"], book_id=book_id, limit=2
        ) == [label]
    assert store._match_entities_vector(
        session=FakeSession(), label="Concept", terms=[], book_id="b1", limit=2
    ) == []

    event_session = FakeSession(responses=[[{"name": "写作"}]])
    assert store._match_events_vector(
        session=event_session, terms=["写作"], book_id="b1", limit=2
    ) == ["写作"]
    assert store._match_events_vector(
        session=event_session, terms=[], book_id="b1", limit=2
    ) == []

    assert store._collect_entity_names(
        [{"concepts": ["自由", "自由", ""]}, {"concepts": ["理性"]}], key="concepts"
    ) == ["自由", "理性"]
    assert store._build_entity_embedding_rows([]) == []
    assert store._build_entity_embedding_rows(["自由", "理性"]) == [
        {"name": "自由", "embedding": [1.0]},
        {"name": "理性", "embedding": [2.0]},
    ]

    session = FakeSession()
    store._upsert_entity_embeddings(session=session, label="Concept", rows=[])
    store._upsert_entity_embeddings(
        session=session, label="Concept", rows=[{"name": "自由", "embedding": [1]}]
    )
    store._upsert_event_embeddings(session=session, book_id="b1", rows=[])
    store._upsert_event_embeddings(
        session=session, book_id="b1", rows=[{"name": "写作", "embedding": [1]}]
    )
    assert len(session.calls) == 2


def test_chapter_grouping_build_payloads_and_singleton(monkeypatch) -> None:
    store = make_store()
    docs = [
        Doc("A", {"chapter_title": "第一章：自由论", "section_indices": "3,4"}),
        Doc("B", {"chapter_title": "第一章：自由论", "section_indices": "2"}),
        Doc("C", {"section_title": "第二章 理性", "section_indices": "bad"}),
        Doc("D", {}),
    ]
    grouped = store._group_docs_by_chapter(docs)
    assert grouped["自由论"]["order"] == 2
    assert len(grouped["自由论"]["docs"]) == 2
    assert "未命名章节" in grouped

    store._llm_graph_extractor.build_graph_payloads.return_value = [{"title": "导言"}]
    assert store._build_graph_payloads(
        docs,
        schema={"mode": "hybrid"},
        max_concepts_per_chapter=5,
        max_events_per_chapter=3,
    ) == [{"title": "导言"}]

    monkeypatch.setattr(neo4j_module, "_neo4j_store", store)
    assert neo4j_module.get_neo4j_store() is store
