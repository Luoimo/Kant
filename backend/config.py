# _*_ coding:utf-8 _*_
# @Time:2026/3/9
# @Author:Chloe
# @File:config.py
# @Project:Kant

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv

# backend 目录，便于从后端目录加载 .env
_BACKEND_DIR = Path(__file__).resolve().parent
# 加载环境变量
load_dotenv(dotenv_path=_BACKEND_DIR / ".env")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: Optional[str] = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = "data/chroma"
    books_data_dir: str = "data/books"

    # Chroma Cloud（设置后自动切换为 CloudClient，留空则使用本地 PersistentClient）
    chroma_api_key: str = ""
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"

    # Mem0 记忆管理
    mem0_user_id: str = "kant-user"
    # Note and plan storage
    note_storage_dir: str = "data/notes"
    plan_storage_dir: str = "data/plans"

    # Single collection for all book chunks (agents read from here)
    books_collection_name: str = "kant_library"

    # SQLite book catalog and cover image directory
    book_catalog_db: str = "data/books.db"
    covers_dir: str = "data/covers"

    # LangSmith / LLMSecOps
    langchain_tracing_v2: str = "false"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "Kant"

    # Lakera Guard
    lakera_guard_api_key: str = ""


def get_settings() -> Settings:
    settings = Settings()
    
    # 将 LangSmith 配置注入到环境变量，供 LangChain 底层自动读取
    import os
    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    return settings
