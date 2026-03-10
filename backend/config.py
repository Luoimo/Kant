# _*_ coding:utf-8 _*_
# @Time:2026/3/9
# @Author:Chloe
# @File:config.py
# @Project:Kant

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录（Kant），便于从任意 cwd 加载 .env
_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: str | None = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = "data/chroma"
    books_data_dir: str = "data/books"

    # Chroma Cloud（设置后自动切换为 CloudClient，留空则使用本地 PersistentClient）
    chroma_api_key: str = ""
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"


def get_settings() -> Settings:
    return Settings()
