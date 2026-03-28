# _*_ coding:utf-8 _*_
# @Time:2026/3/9
# @Author:Chloe
# @File:openai_client.py
# @Project:Kant

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.config import get_settings


def get_llm(**kwargs):
    """获取 LangChain ChatOpenAI 实例，用于对话/推理。"""
    s = get_settings()
    return ChatOpenAI(
        model=s.openai_model,
        openai_api_key=s.openai_api_key,
        openai_api_base=s.openai_base_url,
        temperature=kwargs.get("temperature", 0.3),
        **{k: v for k, v in kwargs.items() if k != "temperature"},
    )


def get_embeddings():
    """获取 OpenAI 嵌入模型，用于向量化（如 Chroma 检索）。"""
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.openai_embedding_model,
        openai_api_key=s.openai_api_key,
        openai_api_base=s.openai_base_url,
    )
