# _*_ coding:utf-8 _*_
# @Time:2026/3/9
# @Author:Chloe
# @File:orchestrator
# @Project:Kant
from langchain.agents import create_agent

orchestrator_agent = create_agent(
    name="orchestrator_agent",
    description="Orchestrator agent is responsible for orchestrating the other agents.",
    tools=[],
    llm=get_llm(),
    verbose=True,
)