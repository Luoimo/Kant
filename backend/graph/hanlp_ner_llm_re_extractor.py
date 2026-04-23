from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any

from .graph_extractor import LLMGraphExtractor

logger = logging.getLogger(__name__)

_HANLP_NER_WORKERS = 4
_NER_TEXT_MAX_CHARS = 14000
_NER_FALLBACK_TASK = "ner*"
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_PERSON_TAGS = {"PERSON", "PER", "NR"}
_CONCEPT_TAGS = {"ORG", "ORGANIZATION", "NT", "LOC", "LOCATION", "NS", "GPE", "FAC"}
_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")


class HanLPNerLLMReExtractor(LLMGraphExtractor):
    """
    混合抽取器：
    - NER: HanLP RESTful（通过 API Key）
    - RE:  复用现有 LLM RE 流程
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        language: str = "zh",
        ner_task: str = "ner/msra",
    ) -> None:
        super().__init__()
        self._api_url = (api_url or "").strip() or "https://www.hanlp.com/hanlp/v21/redirect"
        self._api_key = (api_key or "").strip()
        self._language = (language or "zh").strip() or "zh"
        self._ner_task = (ner_task or "ner/msra").strip() or "ner/msra"
        self._client: Any = None
        self._client_lock = Lock()

    def _get_hanlp_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from hanlp_restful import HanLPClient
            except ImportError as exc:
                raise ImportError(
                    "hanlp_restful 未安装，请执行: pip install hanlp-restful"
                ) from exc
            self._client = HanLPClient(
                self._api_url,
                auth=self._api_key or None,
                language=self._language,
                timeout=120,
            )
            if not self._api_key:
                logger.warning(
                    "HanLP API Key 未配置（HANLP_API_KEY 为空），NER 请求将被服务端拒绝。"
                )
            logger.info(
                "HanLP NER 初始化完成（url=%s, task=%s, lang=%s, key_set=%s）",
                self._api_url,
                self._ner_task,
                self._language,
                bool(self._api_key),
            )
            return self._client

    def _extract_entities_with_ner_llm(
        self,
        *,
        chapter_text: str,
        schema: dict[str, Any],
        chapter_idx: int | None = None,
    ) -> dict[str, Any]:
        if not chapter_text.strip():
            return {"concepts": [], "characters": [], "events": [], "__hanlp_ok": True}
        text_chunks = self._split_text_for_hanlp(chapter_text, max_chars=_NER_TEXT_MAX_CHARS)
        all_concepts: list[str] = []
        all_characters: list[str] = []
        concepts_seen: set[str] = set()
        characters_seen: set[str] = set()
        try:
            client = self._get_hanlp_client()
            for chunk in text_chunks:
                doc = client(
                    chunk,
                    tasks=self._ner_task,
                    language=self._language,
                )
                if self._is_error_doc(doc):
                    logger.warning(
                        "HanLP NER 服务返回错误（chapter=%s, task=%s, err=%s）",
                        chapter_idx if chapter_idx is not None else "-",
                        self._ner_task,
                        self._extract_error_msg(doc),
                    )
                    self._log_empty_ner_debug(
                        chapter_idx=chapter_idx,
                        stage="service_error",
                        raw_doc=doc,
                    )
                    return {"concepts": [], "characters": [], "events": [], "__hanlp_ok": False}
                parsed = self._parse_hanlp_ner(doc=doc, schema=schema)
                for c in parsed.get("concepts", []):
                    k = str(c).strip().lower()
                    if k and k not in concepts_seen:
                        concepts_seen.add(k)
                        all_concepts.append(c)
                for p in parsed.get("characters", []):
                    k = str(p).strip().lower()
                    if k and k not in characters_seen:
                        characters_seen.add(k)
                        all_characters.append(p)
        except Exception as exc:
            logger.warning(
                "HanLP NER 调用失败（chapter=%s），返回空实体（err=%s）",
                chapter_idx if chapter_idx is not None else "-",
                exc,
            )
            return {"concepts": [], "characters": [], "events": [], "__hanlp_ok": False}
        parsed = {
            "concepts": all_concepts,
            "characters": all_characters,
            "events": [],
        }
        # 若指定 NER 任务返回空结果，兜底使用 ner* 再试一次（避免单模型覆盖不足）
        if not parsed.get("concepts") and not parsed.get("characters"):
            try:
                fallback_concepts: list[str] = []
                fallback_characters: list[str] = []
                fallback_concepts_seen: set[str] = set()
                fallback_characters_seen: set[str] = set()
                for chunk in text_chunks:
                    fallback_doc = client(
                        chunk,
                        tasks=_NER_FALLBACK_TASK,
                        language=self._language,
                    )
                    if self._is_error_doc(fallback_doc):
                        logger.warning(
                            "HanLP NER 兜底服务返回错误（chapter=%s, task=%s, err=%s）",
                            chapter_idx if chapter_idx is not None else "-",
                            _NER_FALLBACK_TASK,
                            self._extract_error_msg(fallback_doc),
                        )
                        self._log_empty_ner_debug(
                            chapter_idx=chapter_idx,
                            stage="fallback_service_error",
                            raw_doc=fallback_doc,
                        )
                        parsed["__hanlp_ok"] = False
                        return parsed
                    fallback_parsed = self._parse_hanlp_ner(doc=fallback_doc, schema=schema)
                    for c in fallback_parsed.get("concepts", []):
                        k = str(c).strip().lower()
                        if k and k not in fallback_concepts_seen:
                            fallback_concepts_seen.add(k)
                            fallback_concepts.append(c)
                    for p in fallback_parsed.get("characters", []):
                        k = str(p).strip().lower()
                        if k and k not in fallback_characters_seen:
                            fallback_characters_seen.add(k)
                            fallback_characters.append(p)
                if fallback_concepts or fallback_characters:
                    parsed = {
                        "concepts": fallback_concepts,
                        "characters": fallback_characters,
                        "events": [],
                    }
                    logger.info(
                        "HanLP NER 兜底命中（chapter=%s, fallback_task=%s, concepts=%d, characters=%d）",
                        chapter_idx if chapter_idx is not None else "-",
                        _NER_FALLBACK_TASK,
                        len(parsed.get("concepts", [])),
                        len(parsed.get("characters", [])),
                    )
                else:
                    self._log_empty_ner_debug(
                        chapter_idx=chapter_idx,
                        stage="fallback_empty",
                        raw_doc=fallback_doc,
                    )
            except Exception as fallback_exc:
                logger.warning(
                    "HanLP NER 兜底调用失败（chapter=%s, task=%s, err=%s）",
                    chapter_idx if chapter_idx is not None else "-",
                    _NER_FALLBACK_TASK,
                    fallback_exc,
                )
        if not parsed.get("concepts") and not parsed.get("characters"):
            self._log_empty_ner_debug(
                chapter_idx=chapter_idx,
                stage="primary_empty",
                raw_doc=doc,
            )
        parsed["__hanlp_ok"] = True
        logger.info(
            "HanLP NER 调用成功（chapter=%s, concepts=%d, characters=%d）",
            chapter_idx if chapter_idx is not None else "-",
            len(parsed.get("concepts", [])),
            len(parsed.get("characters", [])),
        )
        return parsed

    @staticmethod
    def _split_text_for_hanlp(text: str, *, max_chars: int) -> list[str]:
        s = (text or "").strip()
        if not s:
            return []
        if len(s) <= max_chars:
            return [s]
        chunks: list[str] = []
        start = 0
        while start < len(s):
            end = min(start + max_chars, len(s))
            if end < len(s):
                # 优先在句号等标点处分段，避免切断实体
                cut = max(
                    s.rfind("。", start, end),
                    s.rfind("！", start, end),
                    s.rfind("？", start, end),
                    s.rfind("\n", start, end),
                )
                if cut > start + max_chars // 2:
                    end = cut + 1
            chunk = s[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        return chunks

    def _run_ner_for_chapters(
        self,
        *,
        chapter_rows: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        ner_results: list[dict[str, Any]] = [{} for _ in chapter_rows]
        success_calls = 0
        failed_calls = 0
        non_empty_chapters = 0
        with ThreadPoolExecutor(max_workers=_HANLP_NER_WORKERS) as pool:
            future_to_idx = {
                pool.submit(
                    self._extract_entities_with_ner_llm,
                    chapter_text=row["text"],
                    schema=schema,
                    chapter_idx=idx,
                ): idx
                for idx, row in enumerate(chapter_rows)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                row = future.result() or {}
                ner_results[idx] = row
                if row.get("__hanlp_ok"):
                    success_calls += 1
                    if row.get("concepts") or row.get("characters"):
                        non_empty_chapters += 1
                else:
                    failed_calls += 1
        logger.info(
            "HanLP NER 汇总：chapters=%d success=%d failed=%d non_empty=%d",
            len(chapter_rows),
            success_calls,
            failed_calls,
            non_empty_chapters,
        )
        return ner_results

    def _apply_character_fallback(
        self,
        *,
        chapter_rows: list[dict[str, Any]],
        ner_results: list[dict[str, Any]],
        schema: dict[str, Any],
        max_concepts_per_chapter: int,
    ) -> int:
        # 该混合模式要求 NER 全由 HanLP 提供，不再触发 LLM 人物兜底。
        return 0

    def _parse_hanlp_ner(self, *, doc: Any, schema: dict[str, Any]) -> dict[str, Any]:
        characters: list[str] = []
        concepts: list[str] = []
        char_seen: set[str] = set()
        concept_seen: set[str] = set()

        ner_payload = None
        if isinstance(doc, dict):
            if self._ner_task in doc:
                ner_payload = doc[self._ner_task]
            else:
                for k, v in doc.items():
                    if str(k).startswith("ner"):
                        ner_payload = v
                        break
        if ner_payload is None:
            ner_payload = doc

        for text, tag in self._iter_ner_entries(ner_payload):
            name = text.strip()
            if len(name) < 2 or not _ZH_RE.search(name):
                continue
            tag_u = tag.upper()
            if schema.get("use_character") and tag_u in _PERSON_TAGS:
                key = name.lower()
                if key not in char_seen:
                    char_seen.add(key)
                    characters.append(name)
            elif schema.get("use_concept") and tag_u in _CONCEPT_TAGS:
                key = name.lower()
                if key not in concept_seen:
                    concept_seen.add(key)
                    concepts.append(name)
        return {
            "concepts": concepts,
            "characters": characters,
            "events": [],
        }

    def _iter_ner_entries(self, payload: Any) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []

        def normalize_label(s: str) -> str:
            return (s or "").strip().upper().replace("B-", "").replace("I-", "").replace("S-", "").replace("E-", "")

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                text = node.get("text") or node.get("entity") or node.get("word")
                label = node.get("label") or node.get("tag") or node.get("type") or node.get("ner")
                if isinstance(text, str) and isinstance(label, str):
                    out.append((text, normalize_label(label)))
                for v in node.values():
                    walk(v)
                return
            if isinstance(node, (list, tuple)):
                # 常见格式: [text, label, start, end]
                if len(node) >= 2 and isinstance(node[0], str) and isinstance(node[1], str):
                    text = node[0].strip()
                    label = normalize_label(node[1])
                    if text and label and _LABEL_RE.match(label):
                        out.append((text, label))
                for item in node:
                    walk(item)

        walk(payload)
        return out

    @staticmethod
    def _is_error_doc(doc: Any) -> bool:
        if not isinstance(doc, dict):
            return False
        if "msg" not in doc:
            return False
        msg = str(doc.get("msg", "")).strip()
        if not msg:
            return False
        code = doc.get("code")
        # HanLP 服务端错误常见格式：{"msg":"请添加参数 auth", "code":500, ...}
        return bool(code) or "auth" in msg.lower()

    @staticmethod
    def _extract_error_msg(doc: Any) -> str:
        if not isinstance(doc, dict):
            return str(doc)
        return str(doc.get("msg") or doc.get("error") or doc)

    def _log_empty_ner_debug(self, *, chapter_idx: int | None, stage: str, raw_doc: Any) -> None:
        preview = ""
        keys: list[str] = []
        if isinstance(raw_doc, dict):
            keys = [str(k) for k in raw_doc.keys()]
        try:
            preview = json.dumps(raw_doc, ensure_ascii=False)[:500]
        except Exception:
            preview = str(raw_doc)[:500]
        logger.info(
            "HanLP NER 空结果诊断（chapter=%s, stage=%s, task=%s, keys=%s, preview=%s）",
            chapter_idx if chapter_idx is not None else "-",
            stage,
            self._ner_task,
            keys,
            preview,
        )
