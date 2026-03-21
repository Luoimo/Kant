# agents/base_agent.py

from abc import ABC, abstractmethod

class base_agent(ABC):
    """
    所有 Agent 的基类
    """

    def __init__(self, name: str, tools: dict = None):
        self.name = name
        self.tools = tools or {}  # 可注入的工具字典

    @abstractmethod
    def run(self, input_data):
        """
        每个 Agent 都必须实现 run 方法
        """
        pass

    def setup(self):
        """
        可选的初始化方法
        """
        print(f"{self.name} setup")

    def teardown(self):
        """
        可选的清理方法
        """
        print(f"{self.name} teardown")