from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

_CHARACTER_TITLE_SUFFIXES = (
    "先生", "女士", "小姐", "夫人", "太太",
    "老师", "公主", "王子", "国王", "女王",
    "将军", "大人", "总裁",
)
_NER_EXTRACT_MODEL = "gpt-4o-mini"
_RE_EXTRACT_MODEL = "gpt-4o-mini"
_LLM_CHAPTER_MAX_CHARS = 20000
_NER_WORKERS = 4
_RE_WORKERS = 4
_LLM_RETRY_MAX = 3
_NER_PROMPT_TEMPLATE = """
你是中文知识图谱 NER 抽取器。请只针对中文文本抽取实体，不要输出英文实体。

模式:
- mode={mode}
- use_concept={use_concept}
- use_character={use_character}

文本:
{text}

要求:
1) 仅抽取文本中明确出现的实体，禁止猜测；
2) 只保留中文实体（必须包含中文字符）；
3) concepts 输出学习术语/概念，characters 必须输出“明确人名/角色名”（不要输出泛词），events 输出事件短语；
4) 每个列表去重；
5) 若该类不存在返回 []。

返回 JSON（不要额外文字）:
{{
  "concepts": ["..."],
  "characters": ["..."],
  "events": ["..."]
}}
""".strip()

_CHARACTER_FALLBACK_PROMPT_TEMPLATE = """
你是中文人物识别器。只做一件事：从文本中抽取明确人物名/角色名。

文本:
{text}

规则:
1) 仅返回人物姓名或稳定称谓（如“贾宝玉”“林黛玉”“王夫人”）；
2) 禁止返回泛词（作者、读者、主角、女孩、男孩、老师、同学等）；
3) 只保留中文人物名；
4) 去重；
5) 最多 {max_characters} 个。

返回 JSON（不要额外文字）:
{{"characters":["..."]}}
""".strip()

_RE_PROMPT_TEMPLATE = """
你是中文知识图谱 RE 抽取器。请只在给定实体集合内抽取关系，不允许新增实体。
仅处理中文语料，不输出英文关系。

模式:
- mode={mode}

文本:
{text}

实体候选:
- concepts: {concepts_json}
- characters: {characters_json}
- events: {events_json}

关系定义:
- concept_relations: 概念之间一般关联 RELATED_TO
- dependencies: 概念依赖 A depends_on B
- hierarchies: 概念层级 A subconcept_of B
- character_relations: 人物关系（冲突/合作/亲属/师生等）
- event_relations: 事件先后顺序 prev -> next

返回 JSON（不要额外文字）:
{{
  "concept_relations": [{{"from":"A","to":"B","weight":1}}],
  "dependencies": [{{"from":"A","to":"B","weight":1}}],
  "hierarchies": [{{"child":"A","parent":"B","weight":1}}],
  "character_relations": [{{"from":"甲","to":"乙","relation":"关系类型","weight":1}}],
  "event_relations": [{{"prev":"事件A","next":"事件B","weight":1}}]
}}
""".strip()


class LLMGraphExtractor:
    @staticmethod
    def _normalize_pair_rows(
        rows: list[dict[str, Any]] | Any,
        *,
        left_key: str,
        right_key: str,
        relation: str | None = None,
        relation_key: str | None = None,
        default_relation: str = "related",
    ) -> list[dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            left = str(row.get(left_key) or "").strip()
            right = str(row.get(right_key) or "").strip()
            if not left or not right or left.lower() == right.lower():
                continue
            relation_type = relation or default_relation
            if relation_key:
                relation_type = str(row.get(relation_key) or default_relation).strip() or default_relation
            out.append(
                {
                    "left": left,
                    "right": right,
                    "relation_type": relation_type,
                    "weight": max(1, int(row.get("weight") or 1)),
                }
            )
        return out

    @staticmethod
    def _normalize_name_list(items: Any, *, max_len: int, max_char: int) -> list[str]:
        if not isinstance(items, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            s = str(item or "").strip()
            if not s or len(s) < 2 or len(s) > max_char:
                continue
            # 仅保留中文实体，避免英文噪声进入图谱。
            if not re.search(r"[\u4e00-\u9fff]", s):
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
            if len(out) >= max_len:
                break
        return out

    @staticmethod
    def _normalize_character_list(items: list[str], *, max_len: int) -> list[str]:
        """
        人物名列表归一化（纯后处理，不调用 LLM）：
        - 裸名（如"达西"）只有唯一称谓变体（如"达西先生"）-> 合并为称谓名；
        - 裸名对应多个称谓变体（如"达西先生"+"达西小姐"）-> 保留各称谓，裸名不合并。
        """
        cleaned = LLMGraphExtractor._normalize_name_list(items, max_len=max_len * 3, max_char=24)
        # 构建 裸名 -> [称谓变体, ...] 映射
        titled_by_base: dict[str, list[str]] = {}
        for name in cleaned:
            for suffix in _CHARACTER_TITLE_SUFFIXES:
                if name.endswith(suffix):
                    base = name[: -len(suffix)].strip()
                    if len(base) >= 2:
                        titled_by_base.setdefault(base, [])
                        if name not in titled_by_base[base]:
                            titled_by_base[base].append(name)
                    break
        out: list[str] = []
        seen: set[str] = set()
        for name in cleaned:
            variants = titled_by_base.get(name, [])
            if variants:
                # 当前 name 是某个称谓的裸名
                final = variants[0] if len(variants) == 1 else name
            else:
                final = name
            key = final.lower()
            if key not in seen:
                seen.add(key)
                out.append(final)
                if len(out) >= max_len:
                    break
        return out

    @staticmethod
    def _apply_character_alias_to_pairs(
        rows: list[dict[str, Any]],
        *,
        alias_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """将关系对中的裸名替换为归一化后的称谓名（与 _normalize_character_list 保持一致）。"""
        if not rows or not alias_map:
            return rows
        out: list[dict[str, Any]] = []
        for row in rows:
            left = alias_map.get(str(row.get("left") or "").strip(), str(row.get("left") or "").strip())
            right = alias_map.get(str(row.get("right") or "").strip(), str(row.get("right") or "").strip())
            if not left or not right or left.lower() == right.lower():
                continue
            out.append({**row, "left": left, "right": right})
        return out

    def _invoke_json_llm(self, *, model: str, prompt: str) -> dict[str, Any]:
        def _parse_json_object(text: str) -> dict[str, Any]:
            raw = str(text or "").strip()
            if not raw:
                return {}
            # 1) strict parse first
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
            # 2) try fenced block extraction
            fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, re.IGNORECASE)
            if fenced:
                try:
                    data = json.loads(fenced.group(1))
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass
            # 3) try from first '{' to any trailing '}' (backward)
            start = raw.find("{")
            if start == -1:
                return {}
            for end in range(len(raw) - 1, start, -1):
                if raw[end] != "}":
                    continue
                candidate = raw[start : end + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue
            return {}

        from llm.openai_client import get_llm

        last_exc: Exception | None = None
        for attempt in range(1, _LLM_RETRY_MAX + 1):
            try:
                resp = get_llm(model=model, temperature=0.0).invoke(prompt)
                content = resp.content if hasattr(resp, "content") else str(resp)
                if isinstance(content, list):
                    content = "".join(
                        str(x.get("text", "")) if isinstance(x, dict) else str(x)
                        for x in content
                    )
                data = _parse_json_object(str(content or ""))
                if data:
                    return data
                # 解析为空通常是模型输出格式不稳，重试通常收益不高、只会放大调用次数。
                # 因此这里直接返回，避免“每次调用最多打 3 次”的请求膨胀。
                logger.warning("json llm parse empty (model=%s, attempt=%d)", model, attempt)
                return {}
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                retryable = any(k in msg for k in ("rate", "429", "timeout", "tempor", "overloaded"))
                if not retryable:
                    logger.warning("json llm call failed (non-retryable, model=%s): %s", model, exc)
                    return {}
                sleep_s = 0.6 * (2 ** (attempt - 1))
                logger.warning(
                    "json llm retry (model=%s, attempt=%d/%d, sleep=%.1fs, err=%s)",
                    model,
                    attempt,
                    _LLM_RETRY_MAX,
                    sleep_s,
                    exc,
                )
                time.sleep(sleep_s)
        if last_exc:
            logger.warning("json llm failed after retries (model=%s): %s", model, last_exc)
        return {}

    def _extract_entities_with_ner_llm(
        self,
        *,
        chapter_text: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if not chapter_text.strip():
            return {"concepts": [], "characters": [], "events": []}
        mode = str(schema.get("mode") or "learning")
        prompt = _NER_PROMPT_TEMPLATE.format(
            mode=mode,
            use_concept=bool(schema.get("use_concept")),
            use_character=bool(schema.get("use_character")),
            text=chapter_text[:_LLM_CHAPTER_MAX_CHARS],
        )
        return self._invoke_json_llm(model=_NER_EXTRACT_MODEL, prompt=prompt)

    def _extract_characters_only_with_ner_llm(
        self,
        *,
        chapter_text: str,
        max_characters: int,
    ) -> list[str]:
        """人物专用 NER 回退：当通用 NER 未抽到人物时触发。"""
        if not chapter_text.strip():
            return []
        prompt = _CHARACTER_FALLBACK_PROMPT_TEMPLATE.format(
            text=chapter_text[:_LLM_CHAPTER_MAX_CHARS],
            max_characters=max_characters,
        )
        data = self._invoke_json_llm(model=_NER_EXTRACT_MODEL, prompt=prompt)
        return self._normalize_name_list(
            data.get("characters", []),
            max_len=max_characters,
            max_char=24,
        )

    def _extract_relations_with_re_llm(
        self,
        *,
        chapter_text: str,
        schema: dict[str, Any],
        entities: dict[str, Any],
    ) -> dict[str, Any]:
        if not chapter_text.strip():
            return {}
        mode = str(schema.get("mode") or "learning")
        concepts = self._normalize_name_list(entities.get("concepts", []), max_len=20, max_char=48)
        characters = self._normalize_name_list(entities.get("characters", []), max_len=16, max_char=24)
        events = self._normalize_name_list(entities.get("events", []), max_len=12, max_char=80)
        prompt = _RE_PROMPT_TEMPLATE.format(
            mode=mode,
            text=chapter_text[:_LLM_CHAPTER_MAX_CHARS],
            concepts_json=json.dumps(concepts, ensure_ascii=False),
            characters_json=json.dumps(characters, ensure_ascii=False),
            events_json=json.dumps(events, ensure_ascii=False),
        )
        return self._invoke_json_llm(model=_RE_EXTRACT_MODEL, prompt=prompt)

    def _build_chapter_rows(self, chapters: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        chapter_rows: list[dict[str, Any]] = []
        total_chunks = 0
        for chapter_idx, chapter in enumerate(chapters):
            docs = chapter.get("docs", []) or []
            total_chunks += len(docs)
            parts: list[str] = []
            current = 0
            for doc in docs:
                text = str(getattr(doc, "page_content", "") or "").strip()
                if not text:
                    continue
                if current >= _LLM_CHAPTER_MAX_CHARS:
                    break
                remaining = _LLM_CHAPTER_MAX_CHARS - current
                parts.append(text[:remaining])
                current += min(len(text), remaining) + 1
            chapter_rows.append({"chapter_idx": chapter_idx, "text": "\n".join(parts)})
        return chapter_rows, total_chunks

    def _run_ner_for_chapters(
        self,
        *,
        chapter_rows: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        ner_results: list[dict[str, Any]] = [{} for _ in chapter_rows]
        with ThreadPoolExecutor(max_workers=_NER_WORKERS) as pool:
            future_to_idx = {
                pool.submit(
                    self._extract_entities_with_ner_llm,
                    chapter_text=row["text"],
                    schema=schema,
                ): idx
                for idx, row in enumerate(chapter_rows)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ner_results[idx] = future.result() or {}
        return ner_results

    def _apply_character_fallback(
        self,
        *,
        chapter_rows: list[dict[str, Any]],
        ner_results: list[dict[str, Any]],
        schema: dict[str, Any],
        max_concepts_per_chapter: int,
    ) -> int:
        fallback_calls = 0
        if not schema.get("use_character"):
            return fallback_calls
        for idx, row in enumerate(chapter_rows):
            chars = self._normalize_name_list(
                ner_results[idx].get("characters", []),
                max_len=max_concepts_per_chapter,
                max_char=24,
            )
            # 仅在该章人物为空时才触发兜底，避免额外请求放大。
            if not chars:
                fallback_calls += 1
                fallback = self._extract_characters_only_with_ner_llm(
                    chapter_text=row["text"],
                    max_characters=max_concepts_per_chapter,
                )
                chars = self._normalize_name_list(
                    chars + fallback,
                    max_len=max_concepts_per_chapter,
                    max_char=24,
                )
            ner_results[idx]["characters"] = chars
        return fallback_calls

    def _run_re_for_chapters(
        self,
        *,
        chapter_rows: list[dict[str, Any]],
        schema: dict[str, Any],
        ner_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        re_results: list[dict[str, Any]] = [{} for _ in chapter_rows]
        with ThreadPoolExecutor(max_workers=_RE_WORKERS) as pool:
            future_to_idx = {
                pool.submit(
                    self._extract_relations_with_re_llm,
                    chapter_text=chapter_rows[idx]["text"],
                    schema=schema,
                    entities=ner_results[idx],
                ): idx
                for idx in range(len(chapter_rows))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                re_results[idx] = future.result() or {}
        return re_results

    @staticmethod
    def _build_character_alias_map(raw_characters: list[Any], canonical_characters: list[str]) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for raw in raw_characters:
            raw_s = str(raw or "").strip()
            for canon in canonical_characters:
                if raw_s and raw_s != canon and canon.startswith(raw_s):
                    alias_map[raw_s] = canon
                    break
        return alias_map

    def _build_single_chapter_payload(
        self,
        *,
        chapter: dict[str, Any],
        ner: dict[str, Any],
        rel: dict[str, Any],
        schema: dict[str, Any],
        max_concepts_per_chapter: int,
        max_events_per_chapter: int,
    ) -> dict[str, Any]:
        # 人物名归一化：裸名合并到唯一称谓，多称谓变体时保留区分
        characters = self._normalize_character_list(
            ner.get("characters", []),
            max_len=max_concepts_per_chapter,
        ) if schema.get("use_character") else []
        alias_map = self._build_character_alias_map(
            ner.get("characters", []) if schema.get("use_character") else [],
            characters,
        )
        return {
            "title": chapter["title"],
            "order": int(chapter["order"]),
            "concepts": self._normalize_name_list(
                ner.get("concepts", []),
                max_len=max_concepts_per_chapter,
                max_char=48,
            ) if schema.get("use_concept") else [],
            "characters": characters,
            "events": self._normalize_name_list(
                ner.get("events", []),
                max_len=max_events_per_chapter,
                max_char=80,
            ) if schema.get("use_character") else [],
            "llm_concept_pairs": self._normalize_pair_rows(
                rel.get("concept_relations", []),
                left_key="from",
                right_key="to",
                relation="semantic_related",
            ),
            "llm_character_pairs": self._apply_character_alias_to_pairs(
                self._normalize_pair_rows(
                    rel.get("character_relations", []),
                    left_key="from",
                    right_key="to",
                    relation_key="relation",
                    default_relation="related",
                ),
                alias_map=alias_map,
            ),
            "llm_event_pairs": self._normalize_pair_rows(
                rel.get("event_relations", []),
                left_key="prev",
                right_key="next",
                relation="next_event",
            ),
            "llm_dependency_pairs": self._normalize_pair_rows(
                rel.get("dependencies", []),
                left_key="from",
                right_key="to",
                relation="depends_on",
            ),
            "llm_hierarchy_pairs": self._normalize_pair_rows(
                rel.get("hierarchies", []),
                left_key="child",
                right_key="parent",
                relation="subconcept_of",
            ),
        }

    @staticmethod
    def _log_extraction_summary(
        *,
        payload: list[dict[str, Any]],
        total_chunks: int,
        chapter_count: int,
        fallback_calls: int,
    ) -> None:
        total_chars = sum(len(ch.get("characters", [])) for ch in payload)
        total_events = sum(len(ch.get("events", [])) for ch in payload)
        total_concepts = sum(len(ch.get("concepts", [])) for ch in payload)
        logger.info(
            "graph extraction with llm ner/re: chapters=%d chunks=%d ner_calls=%d re_calls=%d fallback_calls=%d concepts=%d characters=%d events=%d",
            len(payload),
            total_chunks,
            chapter_count,
            chapter_count,
            fallback_calls,
            total_concepts,
            total_chars,
            total_events,
        )

    def build_graph_payloads(
        self,
        *,
        chapters: list[dict[str, Any]],
        schema: dict[str, Any],
        max_concepts_per_chapter: int,
        max_events_per_chapter: int,
    ) -> list[dict[str, Any]]:
        if not chapters:
            return []
        chapter_rows, total_chunks = self._build_chapter_rows(chapters)
        ner_results = self._run_ner_for_chapters(chapter_rows=chapter_rows, schema=schema)
        fallback_calls = self._apply_character_fallback(
            chapter_rows=chapter_rows,
            ner_results=ner_results,
            schema=schema,
            max_concepts_per_chapter=max_concepts_per_chapter,
        )
        re_results = self._run_re_for_chapters(
            chapter_rows=chapter_rows,
            schema=schema,
            ner_results=ner_results,
        )
        payload = [
            self._build_single_chapter_payload(
                chapter=chapter,
                ner=ner_results[idx] if idx < len(ner_results) else {},
                rel=re_results[idx] if idx < len(re_results) else {},
                schema=schema,
                max_concepts_per_chapter=max_concepts_per_chapter,
                max_events_per_chapter=max_events_per_chapter,
            )
            for idx, chapter in enumerate(chapters)
        ]
        self._log_extraction_summary(
            payload=payload,
            total_chunks=total_chunks,
            chapter_count=len(chapter_rows),
            fallback_calls=fallback_calls,
        )
        return payload
