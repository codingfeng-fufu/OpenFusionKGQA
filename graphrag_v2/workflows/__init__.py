"""工作流模块。

提供各种索引工作流的实现。
"""

from graphrag_v2.workflows.create_base_text_units import run_workflow as create_base_text_units
from graphrag_v2.workflows.create_communities import run_workflow as create_communities
from graphrag_v2.workflows.create_community_reports import run_workflow as create_community_reports
from graphrag_v2.workflows.extract_graph import run_workflow as extract_graph
from graphrag_v2.workflows.generate_embeddings import run_workflow as generate_embeddings
from graphrag_v2.workflows.load_documents import run_workflow as load_documents

__all__ = [
    "create_base_text_units",
    "create_communities",
    "create_community_reports",
    "extract_graph",
    "generate_embeddings",
    "load_documents",
]

