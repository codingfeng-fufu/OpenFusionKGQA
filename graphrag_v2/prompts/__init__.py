"""OpenFusionKGQA prompt module."""

from graphrag_v2.prompts.base import PromptLibrary, PromptTemplate
from graphrag_v2.prompts.community_report import get_community_report_prompt
from graphrag_v2.prompts.entity_extraction import get_entity_extraction_prompt
from graphrag_v2.prompts.query_prompts import (
    get_global_search_map_prompt,
    get_global_search_reduce_prompt,
    get_local_search_prompt,
)

# 别名以保持向后兼容
create_entity_extraction_prompt = get_entity_extraction_prompt
create_community_report_prompt = get_community_report_prompt
create_global_search_map_prompt = get_global_search_map_prompt
create_global_search_reduce_prompt = get_global_search_reduce_prompt
create_local_search_prompt = get_local_search_prompt

__all__ = [
    "PromptTemplate",
    "PromptLibrary",
    "get_entity_extraction_prompt",
    "get_community_report_prompt",
    "get_global_search_map_prompt",
    "get_global_search_reduce_prompt",
    "get_local_search_prompt",
    "create_entity_extraction_prompt",
    "create_community_report_prompt",
    "create_global_search_map_prompt",
    "create_global_search_reduce_prompt",
    "create_local_search_prompt",
]
