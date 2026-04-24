from __future__ import annotations
import logging
import re
from typing import Any
from .graph_extractor import LLMGraphExtractor
from .hanlp_ner_llm_re_extractor import HanLPNerLLMReExtractor

logger = logging.getLogger(__name__)

_CHAPTER_PREFIX_RE = re.compile(
    r"^\s*(?:第\s*[一二三四五六七八九十百千0-9]+\s*[章节卷回部篇]|序章|楔子)\s*[:：.\-\s]*"
)
_VECTOR_MIN_SCORE = 0.35
_GRAPH_EXPAND_MIN_WEIGHT = 2.0

_ZH_STOP_TERMS = {
    # 代词
    "这个", "那个", "这里", "那里", "这些", "那些", "他们", "我们", "你们",
    "自己", "别人", "大家", "某些", "某个", "某种", "什么", "为何", "为什么",
    "如何", "怎么", "怎样", "哪些", "哪个", "哪里", "多少", "几个", "是否",
    "这样", "那样", "这么", "那么", "任何", "所有", "一切", "每个", "其他",
    # 泛化名词（无区分度）
    "作者", "书里", "书中", "内容", "部分", "问题", "区别", "关系", "意思",
    "解释", "概念", "章节", "目录", "前言", "导言", "版权", "声明", "方面",
    "情况", "方式", "方法", "过程", "结果", "原因", "影响", "作用", "特点",
    "特征", "特性", "条件", "因素", "基础", "结构", "模型", "研究", "分析",
    "理解", "认为", "表示", "说明", "指出", "发现", "提出", "包括", "通过",
    "时候", "地方", "东西", "事情", "现象", "情形", "状态", "文章", "文本",
    "例子", "情节", "描述", "观点", "理论", "知识", "信息", "数据", "背景",
    "现实", "生活", "世界", "社会", "文化", "历史", "语言", "思想", "价值",
    # 虚词 / 助词
    "一种", "一个", "一些", "可以", "需要", "应该", "必须", "能够", "已经",
    "非常", "十分", "比较", "相当", "可能", "一定", "确实", "当然",
}
_NOISE_TITLE_PATTERNS = (
    re.compile(r"目录|版权|声明|前言|致谢|附录|参考文献"),
    re.compile(r"^\d+[\.\s]"),  # 纯编号开头
)
_NARRATIVE_VERBS = re.compile(r"说道|说|问|答|想|看着|告诉|回应|争论|决定|发现|意识到")
_LOGIC_CUES = re.compile(r"因此|所以|因为|由此|证明|定义|命题|定理|推论|假设|原理|结论")


class Neo4jStore:
    """
    Neo4j 元数据写入封装。

    - 未配置连接信息时自动禁用；
    - 依赖缺失或运行时报错时降级，不影响主流程。
    """

    def __init__(self) -> None:
        from config import get_settings

        settings = get_settings()
        self._database = settings.neo4j_database
        self._enabled = False
        self._driver = None

        backend = (settings.graph_extractor_backend or "llm").strip().lower()
        if backend == "hanlp_ner_llm_re":
            self._llm_graph_extractor = HanLPNerLLMReExtractor(
                api_url=settings.hanlp_api_url,
                api_key=settings.hanlp_api_key,
                language=settings.hanlp_language,
                ner_task=settings.hanlp_ner_task,
            )
            logger.info("知识图谱抽取后端: HanLP NER（RESTful）+ LLM RE")
        else:
            self._llm_graph_extractor = LLMGraphExtractor()
            logger.info("知识图谱抽取后端: LLM（gpt-4o-mini NER + RE）")

        uri = settings.neo4j_uri.strip()
        user = settings.neo4j_user.strip()
        password = settings.neo4j_password.strip()
        if not uri or not user or not password:
            logger.info("Neo4j 未配置（NEO4J_URI/USER/PASSWORD），已跳过图数据库写入")
            return

        try:
            from neo4j import GraphDatabase
        except ImportError:
            logger.warning("neo4j 驱动未安装，已跳过图数据库写入（pip install neo4j）")
            return

        try:
            self._driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_timeout=30,
                keep_alive=True,
            )
            self._driver.verify_connectivity()
            self._enabled = True
            logger.info("Neo4j 连接成功（uri=%s, database=%s）", uri, self._database)
        except Exception as exc:
            logger.warning(
                "Neo4j 初始化失败（uri=%s, err=%s）——所有图写入将被跳过。"
                "请确认：① Aura 实例已 Resume  ② 网络可访问端口 7687",
                uri,
                exc,
            )
            self._driver = None
            self._enabled = False

    def upsert_book(
        self,
        *,
        book_id: str,
        title: str,
        author: str,
        source: str,
        total_chunks: int,
        cover_path: str = "",
        status: str = "unread",
        progress: float = 0.0,
    ) -> None:
        """在 Neo4j 中创建/更新 Book 节点和 Author 关系。"""
        if not self._enabled or self._driver is None:
            logger.warning("Neo4j 未连接，跳过 upsert_book（book_id=%s）——请检查 Aura 实例是否已 Resume", book_id)
            return

        book_payload: dict[str, Any] = {
            "book_id": book_id,
            "title": title,
            "author": author,
            "source": source,
            "total_chunks": total_chunks,
            "cover_path": cover_path,
            "status": status,
            "progress": progress,
        }
        query = """
        MERGE (b:Book {book_id: $book_id})
        SET b.title = $title,
            b.author = $author,
            b.source = $source,
            b.total_chunks = $total_chunks,
            b.cover_path = $cover_path,
            b.status = $status,
            b.progress = $progress,
            b.updated_at = datetime()
        RETURN b.book_id AS book_id
        """
        try:
            with self._driver.session() as session:
                session.run(query, book_payload).consume()
                if author.strip():
                    session.run(
                        """
                        MATCH (b:Book {book_id: $book_id})
                        MERGE (a:Author {name: $author})
                        MERGE (a)-[:WROTE]->(b)
                        """,
                        {"book_id": book_id, "author": author},
                    ).consume()
        except Exception as exc:
            logger.warning("Neo4j upsert_book 失败（book_id=%s, err=%s）", book_id, exc)

    def upsert_book_graph(
        self,
        *,
        book_id: str,
        documents: list[Any],
        max_concepts_per_chapter: int = 12,
        max_events_per_chapter: int = 6,
    ) -> None:
        """
        从文档构建结构化知识图谱（不落 Chunk 节点）：
        - Book -[:HAS_CHAPTER]-> Chapter
        - 学习类：Concept + DEPENDS_ON + SUBCONCEPT_OF
        - 小说类：Character + Event + INVOLVED_IN
        """
        if not self._enabled or self._driver is None:
            logger.warning("Neo4j 未连接，跳过 upsert_book_graph（book_id=%s）——请检查 Aura 实例是否已 Resume", book_id)
            return

        (
            schema,
            chapters,
            concept_pairs,
            character_pairs,
            event_pairs,
            dependency_pairs,
            hierarchy_pairs,
        ) = self._prepare_graph_payloads(
            documents=documents,
            max_concepts_per_chapter=max_concepts_per_chapter,
            max_events_per_chapter=max_events_per_chapter,
        )

        total_chars_extracted = sum(len(ch.get("characters", [])) for ch in chapters)
        total_concepts_extracted = sum(len(ch.get("concepts", [])) for ch in chapters)
        total_char_pairs = len(character_pairs)
        logger.info(
            "Neo4j 准备写入 book_id=%s: chapters=%d concepts=%d characters=%d character_pairs=%d concept_pairs=%d",
            book_id, len(chapters), total_concepts_extracted, total_chars_extracted,
            total_char_pairs, len(concept_pairs),
        )
        if not chapters:
            logger.warning("Neo4j upsert_book_graph: 章节为空，跳过写入（book_id=%s）", book_id)
            return

        try:
            with self._driver.session() as session:
                self._clear_book_graph_scope(session=session, book_id=book_id)
                self._upsert_chapter_layer(
                    session=session, book_id=book_id, chapters=chapters, schema=schema
                )
                logger.info("Neo4j 章节层写入完成（book_id=%s, chapters=%d）", book_id, len(chapters))
                self._upsert_entity_embeddings_layer(
                    session=session, book_id=book_id, chapters=chapters, schema=schema
                )
                self._upsert_related_pairs_layer(
                    session=session,
                    book_id=book_id,
                    concept_pairs=concept_pairs,
                    character_pairs=character_pairs,
                    event_pairs=event_pairs,
                    dependency_pairs=dependency_pairs,
                    hierarchy_pairs=hierarchy_pairs,
                    schema=schema,
                )
                logger.info(
                    "Neo4j 关系层写入完成（book_id=%s, char_pairs=%d, concept_pairs=%d）",
                    book_id, total_char_pairs, len(concept_pairs),
                )
        except Exception as exc:
            logger.warning("Neo4j upsert_book_graph 失败（book_id=%s, err=%s）", book_id, exc)

    def delete_book(self, *, book_id: str) -> None:
        """从 Neo4j 中移除 Book 节点及其图谱数据。

        清理顺序：
        1. 复用 _clear_book_graph_scope 删除章节/关系；
        2. 清除 book_id 作用域内的 Character/Event/Concept 悬挂实体；
        3. DETACH DELETE Book 节点本身；
        4. 清理变成孤儿的 Author 节点。
        """
        if not self._enabled or self._driver is None or not book_id:
            return
        try:
            with self._driver.session() as session:
                self._clear_book_graph_scope(session=session, book_id=book_id)
                session.run(
                    """
                    MATCH (e:Event {book_id: $book_id})
                    DETACH DELETE e
                    """,
                    {"book_id": book_id},
                ).consume()
                session.run(
                    """
                    MATCH (b:Book {book_id: $book_id})-[:HAS_CHARACTER]->(p:Character)
                    WHERE NOT EXISTS {
                        MATCH (other:Book)-[:HAS_CHARACTER]->(p)
                        WHERE other.book_id <> $book_id
                    }
                    DETACH DELETE p
                    """,
                    {"book_id": book_id},
                ).consume()
                session.run(
                    """
                    MATCH (b:Book {book_id: $book_id})
                    DETACH DELETE b
                    """,
                    {"book_id": book_id},
                ).consume()
                session.run(
                    """
                    MATCH (a:Author)
                    WHERE NOT (a)-[:WROTE]->(:Book)
                    DELETE a
                    """
                ).consume()
                session.run(
                    """
                    MATCH (c:Concept)
                    WHERE NOT (c)--()
                    DELETE c
                    """
                ).consume()
            logger.info("Neo4j 已删除 Book 节点及相关图谱（book_id=%s）", book_id)
        except Exception as exc:
            logger.warning("Neo4j delete_book 失败（book_id=%s, err=%s）", book_id, exc)

    def _prepare_graph_payloads(
        self,
        *,
        documents: list[Any],
        max_concepts_per_chapter: int,
        max_events_per_chapter: int,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        schema = self._select_graph_schema(documents)
        chapters = self._build_graph_payloads(
            documents,
            schema=schema,
            max_concepts_per_chapter=max_concepts_per_chapter,
            max_events_per_chapter=max_events_per_chapter,
        )
        # 纯 LLM NER/RE 抽取：不再使用规则共现/章节邻接关系构造。
        concept_pairs = self._merge_pair_rows(
            self._collect_chapter_pairs(chapters, key="llm_concept_pairs")
        )
        character_pairs = self._merge_pair_rows(
            self._collect_chapter_pairs(chapters, key="llm_character_pairs")
        )
        event_pairs = self._merge_pair_rows(
            self._collect_chapter_pairs(chapters, key="llm_event_pairs")
        )
        dependency_pairs = self._merge_pair_rows(
            self._collect_chapter_pairs(chapters, key="llm_dependency_pairs")
        )
        hierarchy_pairs = self._merge_pair_rows(
            self._collect_chapter_pairs(chapters, key="llm_hierarchy_pairs")
        )
        return (
            schema,
            chapters,
            concept_pairs,
            character_pairs,
            event_pairs,
            dependency_pairs,
            hierarchy_pairs,
        )

    @staticmethod
    def _clear_book_graph_scope(*, session, book_id: str) -> None:
        session.run(
            """
            MATCH (b:Book {book_id: $book_id})-[:HAS_CHAPTER]->(c:Chapter)
            DETACH DELETE c
            """,
            {"book_id": book_id},
        ).consume()
        session.run(
            """
            MATCH (c1:Concept)-[r:RELATED_TO {source_book_id: $book_id}]->(c2:Concept)
            DELETE r
            """,
            {"book_id": book_id},
        ).consume()
        session.run(
            """
            MATCH (c1:Character)-[r:RELATED_TO {source_book_id: $book_id}]->(c2:Character)
            DELETE r
            """,
            {"book_id": book_id},
        ).consume()
        session.run(
            """
            MATCH (c1:Concept)-[r:DEPENDS_ON {source_book_id: $book_id}]->(c2:Concept)
            DELETE r
            """,
            {"book_id": book_id},
        ).consume()
        session.run(
            """
            MATCH (c1:Concept)-[r:SUBCONCEPT_OF {source_book_id: $book_id}]->(c2:Concept)
            DELETE r
            """,
            {"book_id": book_id},
        ).consume()
        session.run(
            """
            MATCH (e1:Event)-[r:NEXT_EVENT {source_book_id: $book_id}]->(e2:Event)
            DELETE r
            """,
            {"book_id": book_id},
        ).consume()

    def _upsert_chapter_layer(
        self,
        *,
        session,
        book_id: str,
        chapters: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> None:
        if not chapters:
            return
        session.run(
            """
            MATCH (b:Book {book_id: $book_id})
            UNWIND $chapters AS chapter
            MERGE (ch:Chapter {book_id: $book_id, title: chapter.title})
            SET ch.order = chapter.order,
                ch.updated_at = datetime()
            MERGE (b)-[:HAS_CHAPTER]->(ch)
            """,
            {"book_id": book_id, "chapters": chapters},
        ).consume()
        if schema.get("use_concept"):
            session.run(
                """
                UNWIND $chapters AS chapter
                MATCH (ch:Chapter {book_id: $book_id, title: chapter.title})
                UNWIND chapter.concepts AS concept_name
                MERGE (c:Concept {name: concept_name})
                MERGE (ch)-[:COVERS]->(c)
                """,
                {"book_id": book_id, "chapters": chapters},
            ).consume()
        if schema.get("use_character"):
            session.run(
                """
                MATCH (b:Book {book_id: $book_id})
                UNWIND $chapters AS chapter
                MATCH (ch:Chapter {book_id: $book_id, title: chapter.title})
                UNWIND chapter.characters AS name
                MERGE (p:Character {name: name})
                MERGE (b)-[:HAS_CHARACTER]->(p)
                MERGE (ch)-[:HAS_CHARACTER]->(p)
                """,
                {"book_id": book_id, "chapters": chapters},
            ).consume()
        if schema.get("use_character"):
            session.run(
                """
                UNWIND $chapters AS chapter
                MATCH (ch:Chapter {book_id: $book_id, title: chapter.title})
                UNWIND chapter.events AS event_name
                MERGE (e:Event {book_id: $book_id, name: event_name})
                SET e.updated_at = datetime()
                MERGE (ch)-[:HAS_EVENT]->(e)
                """,
                {"book_id": book_id, "chapters": chapters},
            ).consume()
            session.run(
                """
                UNWIND $chapters AS chapter
                MATCH (ch:Chapter {book_id: $book_id, title: chapter.title})
                UNWIND chapter.events AS event_name
                MATCH (e:Event {book_id: $book_id, name: event_name})
                UNWIND chapter.characters AS name
                MATCH (p:Character {name: name})
                MERGE (p)-[:INVOLVED_IN]->(e)
                """,
                {"book_id": book_id, "chapters": chapters},
            ).consume()

    def _upsert_entity_embeddings_layer(
        self,
        *,
        session,
        book_id: str,
        chapters: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> None:
        if schema.get("use_concept"):
            concept_names = self._collect_entity_names(chapters, key="concepts")
            concept_rows = self._build_entity_embedding_rows(concept_names)
            if concept_rows:
                self._upsert_entity_embeddings(
                    session=session, label="Concept", rows=concept_rows
                )
        if schema.get("use_character"):
            character_names = self._collect_entity_names(chapters, key="characters")
            character_rows = self._build_entity_embedding_rows(character_names)
            if character_rows:
                self._upsert_entity_embeddings(
                    session=session, label="Character", rows=character_rows
                )
            event_names = self._collect_entity_names(chapters, key="events")
            event_rows = self._build_entity_embedding_rows(event_names)
            if event_rows:
                self._upsert_event_embeddings(session=session, book_id=book_id, rows=event_rows)

    @staticmethod
    def _upsert_related_pairs_layer(
        *,
        session,
        book_id: str,
        concept_pairs: list[dict[str, Any]],
        character_pairs: list[dict[str, Any]],
        event_pairs: list[dict[str, Any]],
        dependency_pairs: list[dict[str, Any]],
        hierarchy_pairs: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> None:
        if character_pairs and schema.get("use_character"):
            session.run(
                """
                UNWIND $pairs AS pair
                MERGE (p1:Character {name: pair.left})
                MERGE (p2:Character {name: pair.right})
                MERGE (p1)-[r:RELATED_TO {source_book_id: $book_id}]->(p2)
                SET r.relation_type = pair.relation_type,
                    r.weight = pair.weight,
                    r.updated_at = datetime()
                """,
                {"book_id": book_id, "pairs": character_pairs},
            ).consume()
        if concept_pairs and schema.get("use_concept"):
            session.run(
                """
                UNWIND $pairs AS pair
                MERGE (c1:Concept {name: pair.left})
                MERGE (c2:Concept {name: pair.right})
                MERGE (c1)-[r:RELATED_TO {source_book_id: $book_id}]->(c2)
                SET r.relation_type = pair.relation_type,
                    r.weight = pair.weight,
                    r.updated_at = datetime()
                """,
                {"book_id": book_id, "pairs": concept_pairs},
            ).consume()
        if dependency_pairs and schema.get("use_concept"):
            session.run(
                """
                UNWIND $pairs AS pair
                MERGE (c1:Concept {name: pair.left})
                MERGE (c2:Concept {name: pair.right})
                MERGE (c1)-[r:DEPENDS_ON {source_book_id: $book_id}]->(c2)
                SET r.weight = pair.weight,
                    r.updated_at = datetime()
                """,
                {"book_id": book_id, "pairs": dependency_pairs},
            ).consume()
        if hierarchy_pairs and schema.get("use_concept"):
            session.run(
                """
                UNWIND $pairs AS pair
                MERGE (c1:Concept {name: pair.left})
                MERGE (c2:Concept {name: pair.right})
                MERGE (c1)-[r:SUBCONCEPT_OF {source_book_id: $book_id}]->(c2)
                SET r.weight = pair.weight,
                    r.updated_at = datetime()
                """,
                {"book_id": book_id, "pairs": hierarchy_pairs},
            ).consume()
        if event_pairs and schema.get("use_character"):
            session.run(
                """
                UNWIND $pairs AS pair
                MERGE (e1:Event {book_id: $book_id, name: pair.left})
                MERGE (e2:Event {book_id: $book_id, name: pair.right})
                MERGE (e1)-[r:NEXT_EVENT {source_book_id: $book_id}]->(e2)
                SET r.weight = pair.weight,
                    r.updated_at = datetime()
                """,
                {"book_id": book_id, "pairs": event_pairs},
            ).consume()

    def _select_graph_schema(self, documents: list[Any]) -> dict[str, Any]:
        """
        上传后动态 schema 选择：
        抽样文本特征 -> 关系分布 -> 选择/组合 schema。
        仅针对中文文本，不依赖规则实体抽取。
        """
        samples = [str(getattr(d, "page_content", "") or "")[:400] for d in documents[:30]]
        sample_text = "\n".join(samples)
        feature_stats = self._analyze_text_features(sample_text)
        relation_stats = self._analyze_relation_distribution(samples)

        concept_score = relation_stats["logic_rel"] * 1.0
        character_score = relation_stats["dialog_rel"] * 1.0 + feature_stats["dialogue_marks"] * 0.2

        # 为避免“人物被 schema 误判关掉”，中文书默认同时开启 concept + character。
        # mode 仅用于 prompt 引导，不再用于是否写入人物节点的硬开关。
        mode = "hybrid"
        if concept_score > character_score * 1.35:
            mode = "learning"
        elif character_score > concept_score * 1.35:
            mode = "fiction"
        use_concept = True
        use_character = True

        schema = {
            "mode": mode,
            "use_concept": use_concept,
            "use_character": use_character,
            "feature_stats": feature_stats,
            "entity_stats": {"concept_count": 0, "character_count": 0},
            "relation_stats": relation_stats,
        }
        logger.info("graph schema selected: %s", schema)
        return schema

    @staticmethod
    def _analyze_text_features(sample_text: str) -> dict[str, Any]:
        text = sample_text or ""
        dialogue_marks = text.count("“") + text.count("”") + text.count("「") + text.count("」")
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        return {
            "dialogue_marks": dialogue_marks,
            "chinese_chars": chinese_chars,
        }

    @staticmethod
    def _analyze_relation_distribution(samples: list[str]) -> dict[str, int]:
        logic_rel = 0
        dialog_rel = 0
        for text in samples:
            logic_rel += len(_LOGIC_CUES.findall(text))
            dialog_rel += len(_NARRATIVE_VERBS.findall(text))
        return {"logic_rel": logic_rel, "dialog_rel": dialog_rel}


    @staticmethod
    def _collect_chapter_pairs(chapters: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for chapter in chapters:
            pairs = chapter.get(key, []) or []
            for row in pairs:
                left = str(row.get("left") or "").strip()
                right = str(row.get("right") or "").strip()
                if not left or not right or left.lower() == right.lower():
                    continue
                rows.append(
                    {
                        "left": left,
                        "right": right,
                        "weight": max(1, int(row.get("weight") or 1)),
                        "relation_type": str(row.get("relation_type") or "related"),
                    }
                )
        return rows

    @staticmethod
    def _merge_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        merged: dict[tuple[str, str, str], int] = {}
        for row in rows:
            left = str(row.get("left") or "").strip()
            right = str(row.get("right") or "").strip()
            relation = str(row.get("relation_type") or "related").strip() or "related"
            if not left or not right or left.lower() == right.lower():
                continue
            key = (left, right, relation)
            merged[key] = merged.get(key, 0) + max(1, int(row.get("weight") or 1))
        return [
            {"left": left, "right": right, "relation_type": relation, "weight": weight}
            for (left, right, relation), weight in merged.items()
        ]

    def graph_retrieve_chunks(
        self,
        *,
        book_id: str,
        query_terms: list[str],
        seed_top_k: int = 6,
        expand_top_k: int = 10,
        chapter_limit: int = 24,
    ) -> dict[str, Any]:
        """
        图检索（结构化图，多跳）：
        向量种子实体 -> 多跳关系扩散 -> 召回关联章节。
        """
        if not self._enabled or self._driver is None or not book_id:
            return {"seed_entities": [], "expanded_entities": [], "chapter_titles": [], "reasoning_paths": []}
        terms = [t.strip().lower() for t in query_terms if t and t.strip()]
        if not terms:
            return {"seed_entities": [], "expanded_entities": [], "chapter_titles": [], "reasoning_paths": []}

        try:
            with self._driver.session() as session:
                seed_entities = self._resolve_seed_entities(
                    session=session, terms=terms, book_id=book_id, seed_top_k=seed_top_k
                )
                if not seed_entities:
                    return {"seed_entities": [], "expanded_entities": [], "chapter_titles": [], "reasoning_paths": []}

                expanded_pairs, reasoning_paths = self._expand_seed_entities_multihop(
                    session=session,
                    seed_entities=seed_entities,
                    book_id=book_id,
                    expand_top_k=expand_top_k,
                )
                expanded_entities = [r["name"] for r in expanded_pairs]

                weighted_rows = self._build_weighted_entity_rows(
                    seed_entities=seed_entities, expanded_pairs=expanded_pairs
                )
                chapter_titles = self._retrieve_chapter_titles_by_entities(
                    session=session,
                    book_id=book_id,
                    weighted_rows=weighted_rows,
                    chapter_limit=chapter_limit,
                )
                return {
                    "seed_entities": seed_entities,
                    "expanded_entities": expanded_entities,
                    "expanded_pairs": expanded_pairs,
                    "chapter_titles": chapter_titles,
                    "reasoning_paths": reasoning_paths[:24],
                }
        except Exception as exc:
            logger.warning("Neo4j graph_retrieve_chunks 失败（book_id=%s, err=%s）", book_id, exc)
            return {"seed_entities": [], "expanded_entities": [], "chapter_titles": [], "reasoning_paths": []}

    def _resolve_seed_entities(
        self,
        *,
        session,
        terms: list[str],
        book_id: str,
        seed_top_k: int,
    ) -> list[str]:
        seed_concepts = self._match_entities_vector(
            session=session, label="Concept", terms=terms, book_id=book_id, limit=seed_top_k
        )
        seed_characters = self._match_entities_vector(
            session=session, label="Character", terms=terms, book_id=book_id, limit=seed_top_k
        )
        seed_events = self._match_events_vector(
            session=session, terms=terms, book_id=book_id, limit=seed_top_k
        )
        merged = seed_concepts + [x for x in seed_characters if x not in seed_concepts]
        return merged + [x for x in seed_events if x not in merged]

    @staticmethod
    def _expand_entities_once(
        *,
        session,
        source_entities: list[str],
        book_id: str,
        expand_top_k: int,
    ) -> list[dict[str, Any]]:
        if not source_entities:
            return []
        expanded_rows = session.run(
            """
            MATCH (n)-[r]->(m)
            WHERE n.name IN $source
              AND (
                (type(r) = 'RELATED_TO' AND coalesce(r.source_book_id, '') = $book_id AND coalesce(r.weight, 0) >= $min_weight) OR
                (type(r) IN ['DEPENDS_ON', 'SUBCONCEPT_OF', 'NEXT_EVENT'] AND coalesce(r.source_book_id, '') = $book_id) OR
                (type(r) = 'INVOLVED_IN' AND m:Event AND coalesce(m.book_id, '') = $book_id)
              )
            RETURN m.name AS name, type(r) AS relation, coalesce(r.weight, 1.0) AS weight
            ORDER BY weight DESC
            LIMIT $expand_top_k
            """,
            {
                "source": source_entities,
                "book_id": book_id,
                "min_weight": _GRAPH_EXPAND_MIN_WEIGHT,
                "expand_top_k": expand_top_k,
            },
        )
        return [
            {
                "name": r["name"],
                "relation": r.get("relation") or "",
                "weight": float(r.get("weight") or 0.0),
            }
            for r in expanded_rows
            if r.get("name")
        ]

    def _expand_seed_entities_multihop(
        self,
        *,
        session,
        seed_entities: list[str],
        book_id: str,
        expand_top_k: int,
        max_hops: int = 2,
    ) -> tuple[list[dict[str, float]], list[str]]:
        expanded: dict[str, float] = {}
        reasoning_paths: list[str] = []
        frontier = list(seed_entities)
        for hop in range(1, max_hops + 1):
            rows = self._expand_entities_once(
                session=session,
                source_entities=frontier,
                book_id=book_id,
                expand_top_k=expand_top_k,
            )
            next_frontier: list[str] = []
            for row in rows:
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                weight = max(float(row.get("weight") or 1.0), 1.0)
                rel = str(row.get("relation") or "")
                prev = expanded.get(name, 0.0)
                if weight > prev:
                    expanded[name] = weight
                if name not in frontier and name not in next_frontier:
                    next_frontier.append(name)
                reasoning_paths.append(f"hop{hop}:{rel}:{name}")
            frontier = next_frontier
            if not frontier:
                break
        expanded_pairs = [
            {"name": name, "weight": weight}
            for name, weight in sorted(expanded.items(), key=lambda x: x[1], reverse=True)
        ]
        return expanded_pairs[: expand_top_k * max_hops], reasoning_paths

    @staticmethod
    def _build_weighted_entity_rows(
        *,
        seed_entities: list[str],
        expanded_pairs: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        weighted_entities: dict[str, float] = {name: 3.0 for name in seed_entities}
        for row in expanded_pairs:
            name = row["name"]
            w = max(float(row["weight"]), 1.0)
            if name in weighted_entities:
                weighted_entities[name] = max(weighted_entities[name], w)
            else:
                weighted_entities[name] = w
        return [{"name": name, "weight": weight} for name, weight in weighted_entities.items()]

    @staticmethod
    def _retrieve_chapter_titles_by_entities(
        *,
        session,
        book_id: str,
        weighted_rows: list[dict[str, float]],
        chapter_limit: int,
    ) -> list[str]:
        chapter_rows = session.run(
            """
            UNWIND $entities AS e
            MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(ch:Chapter)
            WHERE EXISTS { MATCH (ch)-[:COVERS]->(:Concept {name: e.name}) }
               OR EXISTS { MATCH (ch)-[:HAS_CHARACTER]->(:Character {name: e.name}) }
               OR EXISTS { MATCH (ch)-[:HAS_EVENT]->(:Event {book_id: $book_id, name: e.name}) }
            WITH ch, sum(e.weight) AS graph_score
            RETURN ch.title AS chapter_title, graph_score, ch.order AS chapter_order
            ORDER BY graph_score DESC, chapter_order ASC
            LIMIT $chapter_limit
            """,
            {
                "book_id": book_id,
                "entities": weighted_rows,
                "chapter_limit": chapter_limit,
            },
        )
        return [r["chapter_title"] for r in chapter_rows if r.get("chapter_title")]

    def _match_entities_vector(
        self,
        *,
        session,
        label: str,
        terms: list[str],
        book_id: str,
        limit: int,
    ) -> list[str]:
        """
        向量匹配回退：对 query terms 做 embedding，和实体 embedding 做余弦相似度。
        """
        if not terms:
            return []
        query_text = " ".join(terms[:8]).strip()
        if not query_text:
            return []
        try:
            from llm.openai_client import get_embeddings

            embedding = get_embeddings().embed_query(query_text)
            if label == "Concept":
                if book_id:
                    rows = session.run(
                        """
                        MATCH (:Book {book_id: $book_id})-[:HAS_CHAPTER]->(:Chapter)-[:COVERS]->(n:Concept)
                        WHERE n.embedding IS NOT NULL
                        WITH DISTINCT n, vector.similarity.cosine(n.embedding, $embedding) AS score
                        WHERE score >= $min_score
                        RETURN n.name AS name
                        ORDER BY score DESC
                        LIMIT $limit
                        """,
                        {
                            "book_id": book_id,
                            "embedding": embedding,
                            "min_score": _VECTOR_MIN_SCORE,
                            "limit": limit,
                        },
                    )
                else:
                    rows = session.run(
                        """
                        MATCH (n:Concept)
                        WHERE n.embedding IS NOT NULL
                        WITH n, vector.similarity.cosine(n.embedding, $embedding) AS score
                        WHERE score >= $min_score
                        RETURN n.name AS name
                        ORDER BY score DESC
                        LIMIT $limit
                        """,
                        {
                            "embedding": embedding,
                            "min_score": _VECTOR_MIN_SCORE,
                            "limit": limit,
                        },
                    )
            else:
                if book_id:
                    rows = session.run(
                        """
                        MATCH (:Book {book_id: $book_id})-[:HAS_CHARACTER]->(n:Character)
                        WHERE n.embedding IS NOT NULL
                        WITH DISTINCT n, vector.similarity.cosine(n.embedding, $embedding) AS score
                        WHERE score >= $min_score
                        RETURN n.name AS name
                        ORDER BY score DESC
                        LIMIT $limit
                        """,
                        {
                            "book_id": book_id,
                            "embedding": embedding,
                            "min_score": _VECTOR_MIN_SCORE,
                            "limit": limit,
                        },
                    )
                else:
                    rows = session.run(
                        """
                        MATCH (n:Character)
                        WHERE n.embedding IS NOT NULL
                        WITH n, vector.similarity.cosine(n.embedding, $embedding) AS score
                        WHERE score >= $min_score
                        RETURN n.name AS name
                        ORDER BY score DESC
                        LIMIT $limit
                        """,
                        {
                            "embedding": embedding,
                            "min_score": _VECTOR_MIN_SCORE,
                            "limit": limit,
                        },
                    )
            return [r["name"] for r in rows if r.get("name")]
        except Exception:
            return []

    def _match_events_vector(
        self,
        *,
        session,
        terms: list[str],
        book_id: str,
        limit: int,
    ) -> list[str]:
        if not terms or not book_id:
            return []
        query_text = " ".join(terms[:8]).strip()
        if not query_text:
            return []
        try:
            from llm.openai_client import get_embeddings

            embedding = get_embeddings().embed_query(query_text)
            rows = session.run(
                """
                MATCH (e:Event {book_id: $book_id})
                WHERE e.embedding IS NOT NULL
                WITH e, vector.similarity.cosine(e.embedding, $embedding) AS score
                WHERE score >= $min_score
                RETURN e.name AS name
                ORDER BY score DESC
                LIMIT $limit
                """,
                {
                    "book_id": book_id,
                    "embedding": embedding,
                    "min_score": _VECTOR_MIN_SCORE,
                    "limit": limit,
                },
            )
            return [r["name"] for r in rows if r.get("name")]
        except Exception:
            return []

    @staticmethod
    def _collect_entity_names(
        chapters: list[dict[str, Any]],
        *,
        key: str,
    ) -> list[str]:
        names: list[str] = []
        for chapter in chapters:
            names.extend(chapter.get(key, []) or [])
        out: list[str] = []
        seen: set[str] = set()
        for n in names:
            s = str(n).strip()
            if not s:
                continue
            k = s.lower()
            if k not in seen:
                seen.add(k)
                out.append(s)
        return out

    def _build_entity_embedding_rows(self, names: list[str]) -> list[dict[str, Any]]:
        if not names:
            return []
        try:
            from llm.openai_client import get_embeddings

            vectors = get_embeddings().embed_documents(names)
            return [
                {"name": name, "embedding": vec}
                for name, vec in zip(names, vectors)
                if name and vec
            ]
        except Exception as exc:
            logger.warning("entity embedding build failed: %s", exc)
            return []

    @staticmethod
    def _upsert_entity_embeddings(*, session, label: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (n:{label} {{name: row.name}})
            SET n.embedding = row.embedding
            """,
            {"rows": rows},
        ).consume()

    @staticmethod
    def _upsert_event_embeddings(*, session, book_id: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        session.run(
            """
            UNWIND $rows AS row
            MATCH (e:Event {book_id: $book_id, name: row.name})
            SET e.embedding = row.embedding
            """,
            {"book_id": book_id, "rows": rows},
        ).consume()

    @staticmethod
    def _is_noise_text(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return True
        # 中文停用词
        if t in _ZH_STOP_TERMS:
            return True
        # 中文语气词结尾
        if t.endswith(("的吗", "吗", "呢", "呀", "嗯", "啊", "哦")):
            return True
        # 中文虚词开头（否则会把"因此"等连词抓进来）
        if re.match(r"^(因此|所以|因为|但是|然而|如果|虽然|尽管|不过|而且|并且|于是|可是)", t):
            return True
        # 纯数字 / 纯符号
        if re.fullmatch(r"[\d\s\W]+", t):
            return True
        # 噪声标题 pattern
        for p in _NOISE_TITLE_PATTERNS:
            if p.search(t):
                return True
        return False

    @staticmethod
    def _normalize_title(raw: str) -> str:
        title = (raw or "").strip()
        if not title:
            return ""
        # 尝试去掉"第X章"前缀，保留副标题部分
        stripped = _CHAPTER_PREFIX_RE.sub("", title).strip()
        # 若去掉前缀后为空（即标题本身就是"第X章"），保留原始标题作为章节标识
        effective = stripped if stripped else title
        if Neo4jStore._is_noise_text(effective):
            return ""
        return effective

    # ------------------------------------------------------------------
    # 内部：按章节分组文档
    # ------------------------------------------------------------------

    def _group_docs_by_chapter(
        self,
        documents: list[Any],
    ) -> dict[str, dict[str, Any]]:
        chapter_map: dict[str, dict[str, Any]] = {}
        for doc in documents:
            metadata = getattr(doc, "metadata", {}) or {}
            chapter_title = self._normalize_title(str(metadata.get("chapter_title") or ""))
            if not chapter_title:
                chapter_title = self._normalize_title(str(metadata.get("section_title") or ""))
            if not chapter_title:
                chapter_title = "未命名章节"
            section_indices_raw = str(metadata.get("section_indices") or "")
            section_candidates = [
                int(x) for x in section_indices_raw.split(",") if x.strip().isdigit()
            ]
            chapter_order = min(section_candidates) if section_candidates else 10**9
            entry = chapter_map.setdefault(
                chapter_title,
                {"title": chapter_title, "order": chapter_order, "docs": []},
            )
            entry["order"] = min(entry["order"], chapter_order)
            entry["docs"].append(doc)
        return chapter_map

    def _build_graph_payloads(
        self,
        documents: list[Any],
        *,
        schema: dict[str, Any],
        max_concepts_per_chapter: int,
        max_events_per_chapter: int,
    ) -> list[dict[str, Any]]:
        chapter_map = self._group_docs_by_chapter(documents)
        chapters = sorted(chapter_map.values(), key=lambda x: (x["order"], x["title"]))
        return self._llm_graph_extractor.build_graph_payloads(
            chapters=chapters,
            schema=schema,
            max_concepts_per_chapter=max_concepts_per_chapter,
            max_events_per_chapter=max_events_per_chapter,
        )

_neo4j_store: Neo4jStore | None = None


def get_neo4j_store() -> Neo4jStore:
    global _neo4j_store
    if _neo4j_store is None:
        _neo4j_store = Neo4jStore()
    return _neo4j_store

