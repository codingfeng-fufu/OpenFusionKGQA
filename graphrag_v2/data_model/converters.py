"""数据转换工具。

提供 dataclass 与 Pandas DataFrame 之间的转换功能。
"""

import json
from dataclasses import asdict, fields
from typing import Any, TypeVar

import pandas as pd

from graphrag_v2.data_model.community import Community
from graphrag_v2.data_model.community_report import CommunityReport
from graphrag_v2.data_model.covariate import Covariate
from graphrag_v2.data_model.document import Document
from graphrag_v2.data_model.entity import Entity
from graphrag_v2.data_model.relationship import Relationship
from graphrag_v2.data_model.text_unit import TextUnit

T = TypeVar("T")


def dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """将 dataclass 对象转换为字典。
    
    Args:
        obj: dataclass 对象
        
    Returns:
        dict: 字典表示
    """
    return asdict(obj)


def dataclass_list_to_dataframe(objects: list[Any]) -> pd.DataFrame:
    """将 dataclass 对象列表转换为 DataFrame。
    
    Args:
        objects: dataclass 对象列表
        
    Returns:
        pd.DataFrame: DataFrame 表示
    """
    if not objects:
        return pd.DataFrame()
    
    # 转换为字典列表
    data = [dataclass_to_dict(obj) for obj in objects]
    
    # 创建 DataFrame
    df = pd.DataFrame(data)
    
    # 处理列表和字典类型的列（转换为 JSON 字符串以便存储）
    for col in df.columns:
        if df[col].dtype == object:
            # 检查是否包含列表或字典
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, (list, dict)):
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if x is not None else None
                )
    
    return df


def dataframe_to_dataclass_list(
    df: pd.DataFrame, 
    dataclass_type: type[T]
) -> list[T]:
    """将 DataFrame 转换为 dataclass 对象列表。
    
    Args:
        df: DataFrame
        dataclass_type: dataclass 类型
        
    Returns:
        list: dataclass 对象列表
    """
    if df.empty:
        return []
    
    # 获取 dataclass 的字段名
    field_names = {f.name for f in fields(dataclass_type)}
    
    # 转换为字典列表
    records = df.to_dict("records")
    
    # 处理 JSON 字符串列（转换回列表或字典）
    for record in records:
        for key, value in record.items():
            if isinstance(value, str) and key in field_names:
                # 尝试解析 JSON
                try:
                    if value.startswith("[") or value.startswith("{"):
                        record[key] = json.loads(value)
                except (json.JSONDecodeError, AttributeError):
                    pass
    
    # 创建 dataclass 对象
    objects = []
    for record in records:
        # 只保留 dataclass 中定义的字段
        filtered_record = {k: v for k, v in record.items() if k in field_names}
        # 处理 NaN 值
        for key, value in filtered_record.items():
            if pd.isna(value):
                filtered_record[key] = None
        objects.append(dataclass_type(**filtered_record))
    
    return objects


def entities_to_dataframe(entities: list[Entity]) -> pd.DataFrame:
    """将实体列表转换为 DataFrame。
    
    Args:
        entities: 实体列表
        
    Returns:
        pd.DataFrame: DataFrame 表示
    """
    return dataclass_list_to_dataframe(entities)


def dataframe_to_entities(df: pd.DataFrame) -> list[Entity]:
    """将 DataFrame 转换为实体列表。
    
    Args:
        df: DataFrame
        
    Returns:
        list[Entity]: 实体列表
    """
    return dataframe_to_dataclass_list(df, Entity)


def relationships_to_dataframe(relationships: list[Relationship]) -> pd.DataFrame:
    """将关系列表转换为 DataFrame。
    
    Args:
        relationships: 关系列表
        
    Returns:
        pd.DataFrame: DataFrame 表示
    """
    return dataclass_list_to_dataframe(relationships)


def dataframe_to_relationships(df: pd.DataFrame) -> list[Relationship]:
    """将 DataFrame 转换为关系列表。
    
    Args:
        df: DataFrame
        
    Returns:
        list[Relationship]: 关系列表
    """
    return dataframe_to_dataclass_list(df, Relationship)


def communities_to_dataframe(communities: list[Community]) -> pd.DataFrame:
    """将社区列表转换为 DataFrame。
    
    Args:
        communities: 社区列表
        
    Returns:
        pd.DataFrame: DataFrame 表示
    """
    return dataclass_list_to_dataframe(communities)


def dataframe_to_communities(df: pd.DataFrame) -> list[Community]:
    """将 DataFrame 转换为社区列表。
    
    Args:
        df: DataFrame
        
    Returns:
        list[Community]: 社区列表
    """
    return dataframe_to_dataclass_list(df, Community)


def community_reports_to_dataframe(
    reports: list[CommunityReport]
) -> pd.DataFrame:
    """将社区报告列表转换为 DataFrame。
    
    Args:
        reports: 社区报告列表
        
    Returns:
        pd.DataFrame: DataFrame 表示
    """
    return dataclass_list_to_dataframe(reports)


def dataframe_to_community_reports(df: pd.DataFrame) -> list[CommunityReport]:
    """将 DataFrame 转换为社区报告列表。
    
    Args:
        df: DataFrame
        
    Returns:
        list[CommunityReport]: 社区报告列表
    """
    return dataframe_to_dataclass_list(df, CommunityReport)


def text_units_to_dataframe(text_units: list[TextUnit]) -> pd.DataFrame:
    """将文本单元列表转换为 DataFrame。
    
    Args:
        text_units: 文本单元列表
        
    Returns:
        pd.DataFrame: DataFrame 表示
    """
    return dataclass_list_to_dataframe(text_units)


def dataframe_to_text_units(df: pd.DataFrame) -> list[TextUnit]:
    """将 DataFrame 转换为文本单元列表。
    
    Args:
        df: DataFrame
        
    Returns:
        list[TextUnit]: 文本单元列表
    """
    return dataframe_to_dataclass_list(df, TextUnit)


def documents_to_dataframe(documents: list[Document]) -> pd.DataFrame:
    """将文档列表转换为 DataFrame。
    
    Args:
        documents: 文档列表
        
    Returns:
        pd.DataFrame: DataFrame 表示
    """
    return dataclass_list_to_dataframe(documents)


def dataframe_to_documents(df: pd.DataFrame) -> list[Document]:
    """将 DataFrame 转换为文档列表。
    
    Args:
        df: DataFrame
        
    Returns:
        list[Document]: 文档列表
    """
    return dataframe_to_dataclass_list(df, Document)


def covariates_to_dataframe(covariates: list[Covariate]) -> pd.DataFrame:
    """将协变量列表转换为 DataFrame。
    
    Args:
        covariates: 协变量列表
        
    Returns:
        pd.DataFrame: DataFrame 表示
    """
    return dataclass_list_to_dataframe(covariates)


def dataframe_to_covariates(df: pd.DataFrame) -> list[Covariate]:
    """将 DataFrame 转换为协变量列表。
    
    Args:
        df: DataFrame
        
    Returns:
        list[Covariate]: 协变量列表
    """
    return dataframe_to_dataclass_list(df, Covariate)

