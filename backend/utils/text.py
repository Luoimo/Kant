import logging
import re

logger = logging.getLogger(__name__)


def safe_id(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "_", text)[:40]


def tokenize(text: str) -> list[str]:
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)
        return list(jieba.cut(text))
    except ImportError:
        logger.warning("jieba 未安装，退回字符级 tokenize")
        return list(text)
