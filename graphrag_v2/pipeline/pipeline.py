"""Pipeline 类定义。

封装工作流的运行。
"""

from collections.abc import Generator

from graphrag_v2.pipeline.workflow import Workflow


class Pipeline:
    """封装工作流运行的 Pipeline 类。
    
    Pipeline 是一系列工作流的有序集合，按顺序执行。
    
    Attributes:
        workflows: 工作流列表
    """

    def __init__(self, workflows: list[Workflow]):
        """初始化 Pipeline。
        
        Args:
            workflows: 工作流列表，每个工作流是 (名称, 函数) 的元组
        """
        self.workflows = workflows

    def run(self) -> Generator[Workflow, None, None]:
        """返回 Pipeline 工作流的生成器。
        
        Yields:
            工作流元组 (名称, 函数)
        """
        yield from self.workflows

    def names(self) -> list[str]:
        """返回 Pipeline 中工作流的名称列表。
        
        Returns:
            工作流名称列表
        """
        return [name for name, _ in self.workflows]

    def remove(self, name: str) -> None:
        """从 Pipeline 中移除指定名称的工作流。
        
        Args:
            name: 要移除的工作流名称
        """
        self.workflows = [w for w in self.workflows if w[0] != name]

    def add(self, workflow: Workflow) -> None:
        """向 Pipeline 添加工作流。
        
        Args:
            workflow: 要添加的工作流元组 (名称, 函数)
        """
        self.workflows.append(workflow)

    def insert(self, index: int, workflow: Workflow) -> None:
        """在指定位置插入工作流。
        
        Args:
            index: 插入位置
            workflow: 要插入的工作流元组 (名称, 函数)
        """
        self.workflows.insert(index, workflow)

    def __len__(self) -> int:
        """返回 Pipeline 中工作流的数量。
        
        Returns:
            工作流数量
        """
        return len(self.workflows)

    def __repr__(self) -> str:
        """返回 Pipeline 的字符串表示。
        
        Returns:
            字符串表示
        """
        workflow_names = ", ".join(self.names())
        return f"Pipeline({len(self)} workflows: {workflow_names})"

