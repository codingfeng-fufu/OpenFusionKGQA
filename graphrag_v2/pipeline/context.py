"""Pipeline 运行上下文。

提供 Pipeline 运行时所需的所有上下文信息。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineRunStats:
    """Pipeline 运行统计信息。
    
    Attributes:
        total_runtime: 总运行时间（秒）
        num_documents: 处理的文档数量
        num_text_units: 生成的文本单元数量
        num_entities: 提取的实体数量
        num_relationships: 提取的关系数量
        num_communities: 检测的社区数量
    """

    total_runtime: float = 0.0
    """总运行时间（秒）。"""

    num_documents: int = 0
    """处理的文档数量。"""

    num_text_units: int = 0
    """生成的文本单元数量。"""

    num_entities: int = 0
    """提取的实体数量。"""

    num_relationships: int = 0
    """提取的关系数量。"""

    num_communities: int = 0
    """检测的社区数量。"""


@dataclass
class PipelineStorage:
    """Pipeline 存储抽象。
    
    简化版的存储接口，用于读写数据。
    
    Attributes:
        base_dir: 基础目录路径
        data: 内存中的数据存储（用于测试）
    """

    base_dir: str
    """基础目录路径。"""

    data: dict[str, Any] = field(default_factory=dict)
    """内存中的数据存储。"""

    async def get(self, key: str) -> Any | None:
        """获取数据。
        
        Args:
            key: 数据键
            
        Returns:
            数据值，如果不存在则返回 None
        """
        return self.data.get(key)

    async def set(self, key: str, value: Any) -> None:
        """设置数据。
        
        Args:
            key: 数据键
            value: 数据值
        """
        self.data[key] = value

    async def has(self, key: str) -> bool:
        """检查数据是否存在。
        
        Args:
            key: 数据键
            
        Returns:
            是否存在
        """
        return key in self.data


@dataclass
class PipelineCache:
    """Pipeline 缓存抽象。
    
    用于缓存 LLM 响应等数据。
    
    Attributes:
        enabled: 是否启用缓存
        cache_dir: 缓存目录
        data: 内存中的缓存数据
    """

    enabled: bool = True
    """是否启用缓存。"""

    cache_dir: str = ".cache"
    """缓存目录。"""

    data: dict[str, Any] = field(default_factory=dict)
    """内存中的缓存数据。"""

    async def get(self, key: str) -> Any | None:
        """获取缓存数据。
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在则返回 None
        """
        if not self.enabled:
            return None
        return self.data.get(key)

    async def set(self, key: str, value: Any) -> None:
        """设置缓存数据。
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        if self.enabled:
            self.data[key] = value


@dataclass
class WorkflowCallbacks:
    """工作流回调接口。
    
    用于在工作流执行过程中提供进度反馈。
    """

    def on_workflow_start(self, workflow_name: str) -> None:
        """工作流开始时调用。
        
        Args:
            workflow_name: 工作流名称
        """
        pass

    def on_workflow_end(self, workflow_name: str, result: Any) -> None:
        """工作流结束时调用。
        
        Args:
            workflow_name: 工作流名称
            result: 工作流结果
        """
        pass

    def on_error(self, workflow_name: str, error: Exception) -> None:
        """发生错误时调用。
        
        Args:
            workflow_name: 工作流名称
            error: 错误对象
        """
        pass

    def on_progress(self, workflow_name: str, current: int, total: int) -> None:
        """进度更新时调用。
        
        Args:
            workflow_name: 工作流名称
            current: 当前进度
            total: 总数
        """
        pass


PipelineState = dict[str, Any]
"""Pipeline 状态类型。

用于存储运行时状态、预计算数据或实验性功能的任意属性包。
"""


@dataclass
class PipelineRunContext:
    """Pipeline 运行上下文。
    
    提供当前 Pipeline 运行所需的所有上下文信息。
    
    Attributes:
        stats: 运行统计信息
        input_storage: 输入文档存储
        output_storage: 输出数据存储（长期存储）
        previous_storage: 上一次运行的存储（更新模式）
        cache: 缓存实例（用于读取之前的 LLM 响应）
        callbacks: 回调接口（用于进度反馈）
        state: 运行时状态（任意属性包）
    """

    stats: PipelineRunStats
    """运行统计信息。"""

    input_storage: PipelineStorage
    """输入文档存储。"""

    output_storage: PipelineStorage
    """输出数据存储。"""

    previous_storage: PipelineStorage | None = None
    """上一次运行的存储（更新模式）。"""

    cache: PipelineCache = field(default_factory=PipelineCache)
    """缓存实例。"""

    callbacks: WorkflowCallbacks = field(default_factory=WorkflowCallbacks)
    """回调接口。"""

    state: PipelineState = field(default_factory=dict)
    """运行时状态。"""

