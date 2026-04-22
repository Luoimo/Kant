# _*_ coding:utf-8 _*_
# @Time:2026/3/9
# @Author:Chloe
# @File:openai_client.py
# @Project:Kant

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import get_settings


def get_llm(**kwargs):
    """获取 LangChain ChatOpenAI 实例，用于对话/推理。"""
    s = get_settings()
    model = kwargs.pop("model", None) or s.openai_model
    temperature = kwargs.pop("temperature", 0.3)
    return ChatOpenAI(
        model=model,
        openai_api_key=s.openai_api_key,
        openai_api_base=s.openai_base_url,
        temperature=temperature,
        **kwargs,
    )


def get_embeddings():
    """获取 OpenAI 嵌入模型，用于向量化（如 Chroma 检索）。"""
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.openai_embedding_model,
        openai_api_key=s.openai_api_key,
        openai_api_base=s.openai_base_url,
    )
