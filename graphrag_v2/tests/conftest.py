"""
Pytest 配置文件

定义共享的 fixtures 和测试配置。
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Keep native thread pools small in pytest and inherited CLI subprocesses.
# This avoids intermittent pyarrow/pandas shutdown aborts in full-suite runs.
os.environ["OMP_NUM_THREADS"] = "1"

from graphrag_v2.config import GraphRagConfig, create_default_config
from graphrag_v2.data_model import Document, Entity, Relationship, Community


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """创建临时目录用于测试。"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def default_config() -> GraphRagConfig:
    """创建默认配置。"""
    return create_default_config()


@pytest.fixture
def sample_config(temp_dir: Path) -> GraphRagConfig:
    """创建示例配置，使用临时目录。"""
    config = create_default_config()
    config.storage.base_dir = str(temp_dir)
    return config


@pytest.fixture
def sample_documents() -> list[Document]:
    """创建示例文档。"""
    return [
        Document(
            id="doc1",
            short_id="doc1",
            title="文档1",
            text="GraphRAG 是微软开发的技术。",
        ),
        Document(
            id="doc2",
            short_id="doc2",
            title="文档2",
            text="Leiden 算法用于社区检测。",
        ),
    ]


@pytest.fixture
def sample_entities() -> list[Entity]:
    """创建示例实体。"""
    return [
        Entity(
            id="e1",
            short_id="e1",
            title="GraphRAG",
            type="技术",
            description="一种结合知识图谱和 RAG 的技术",
        ),
        Entity(
            id="e2",
            short_id="e2",
            title="微软",
            type="组织",
            description="GraphRAG 的开发者",
        ),
        Entity(
            id="e3",
            short_id="e3",
            title="Leiden算法",
            type="算法",
            description="用于社区检测的算法",
        ),
    ]


@pytest.fixture
def sample_relationships() -> list[Relationship]:
    """创建示例关系。"""
    return [
        Relationship(
            id="r1",
            short_id="r1",
            source="微软",
            target="GraphRAG",
            description="微软开发了 GraphRAG",
            weight=0.9,
        ),
        Relationship(
            id="r2",
            short_id="r2",
            source="GraphRAG",
            target="Leiden算法",
            description="GraphRAG 使用 Leiden 算法",
            weight=0.8,
        ),
    ]


@pytest.fixture
def sample_communities() -> list[Community]:
    """创建示例社区。"""
    return [
        Community(
            id="c1",
            short_id="c1",
            title="GraphRAG 技术生态",
            level="0",
            parent="root",
            children=[],
            entity_ids=["e1", "e2", "e3"],
            relationship_ids=["r1", "r2"],
        ),
    ]


@pytest.fixture
def sample_text() -> str:
    """创建示例文本。"""
    return """
GraphRAG 是微软研究院开发的一种创新技术。它将知识图谱与检索增强生成（RAG）相结合。
该技术使用 Leiden 算法进行社区检测，能够从大规模文本中提取结构化知识。
GraphRAG 在问答、摘要生成等任务中表现出色。
"""


@pytest.fixture
def chinese_text() -> str:
    """创建中文示例文本。"""
    return """
人工智能（AI）是计算机科学的一个分支。机器学习是人工智能的核心技术之一。
深度学习是机器学习的一个子领域，它使用神经网络来学习数据的表示。
OpenAI 是一家专注于人工智能研究的公司，开发了 GPT 系列模型。
"""
