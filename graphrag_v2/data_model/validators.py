"""数据验证工具。

提供数据模型的验证功能，确保数据的完整性和一致性。
"""

from graphrag_v2.data_model.community import Community
from graphrag_v2.data_model.community_report import CommunityReport
from graphrag_v2.data_model.covariate import Covariate
from graphrag_v2.data_model.document import Document
from graphrag_v2.data_model.entity import Entity
from graphrag_v2.data_model.relationship import Relationship
from graphrag_v2.data_model.text_unit import TextUnit


def validate_entity(entity: Entity) -> list[str]:
    """验证实体数据。
    
    Args:
        entity: 实体对象
        
    Returns:
        list[str]: 错误消息列表（空列表表示验证通过）
    """
    errors = []
    
    if not entity.id:
        errors.append("实体 ID 不能为空")
    
    if not entity.title:
        errors.append("实体标题不能为空")
    
    if entity.rank is not None and entity.rank < 0:
        errors.append("实体排名不能为负数")
    
    return errors


def validate_relationship(relationship: Relationship) -> list[str]:
    """验证关系数据。
    
    Args:
        relationship: 关系对象
        
    Returns:
        list[str]: 错误消息列表（空列表表示验证通过）
    """
    errors = []
    
    if not relationship.id:
        errors.append("关系 ID 不能为空")
    
    if not relationship.source:
        errors.append("关系源实体不能为空")
    
    if not relationship.target:
        errors.append("关系目标实体不能为空")
    
    if relationship.weight is not None and relationship.weight < 0:
        errors.append("关系权重不能为负数")
    
    if relationship.rank is not None and relationship.rank < 0:
        errors.append("关系排名不能为负数")
    
    return errors


def validate_community(community: Community) -> list[str]:
    """验证社区数据。
    
    Args:
        community: 社区对象
        
    Returns:
        list[str]: 错误消息列表（空列表表示验证通过）
    """
    errors = []
    
    if not community.id:
        errors.append("社区 ID 不能为空")
    
    if not community.title:
        errors.append("社区标题不能为空")
    
    if not community.level:
        errors.append("社区层级不能为空")
    
    if community.size is not None and community.size < 0:
        errors.append("社区大小不能为负数")
    
    return errors


def validate_community_report(report: CommunityReport) -> list[str]:
    """验证社区报告数据。
    
    Args:
        report: 社区报告对象
        
    Returns:
        list[str]: 错误消息列表（空列表表示验证通过）
    """
    errors = []
    
    if not report.id:
        errors.append("报告 ID 不能为空")
    
    if not report.title:
        errors.append("报告标题不能为空")
    
    if not report.community_id:
        errors.append("报告关联的社区 ID 不能为空")
    
    if report.rank is not None and report.rank < 0:
        errors.append("报告排名不能为负数")
    
    if report.size is not None and report.size < 0:
        errors.append("报告大小不能为负数")
    
    return errors


def validate_text_unit(text_unit: TextUnit) -> list[str]:
    """验证文本单元数据。
    
    Args:
        text_unit: 文本单元对象
        
    Returns:
        list[str]: 错误消息列表（空列表表示验证通过）
    """
    errors = []
    
    if not text_unit.id:
        errors.append("文本单元 ID 不能为空")
    
    if not text_unit.text:
        errors.append("文本单元内容不能为空")
    
    if text_unit.n_tokens is not None and text_unit.n_tokens < 0:
        errors.append("Token 数量不能为负数")
    
    return errors


def validate_document(document: Document) -> list[str]:
    """验证文档数据。
    
    Args:
        document: 文档对象
        
    Returns:
        list[str]: 错误消息列表（空列表表示验证通过）
    """
    errors = []
    
    if not document.id:
        errors.append("文档 ID 不能为空")
    
    if not document.title:
        errors.append("文档标题不能为空")
    
    return errors


def validate_covariate(covariate: Covariate) -> list[str]:
    """验证协变量数据。
    
    Args:
        covariate: 协变量对象
        
    Returns:
        list[str]: 错误消息列表（空列表表示验证通过）
    """
    errors = []
    
    if not covariate.id:
        errors.append("协变量 ID 不能为空")
    
    if not covariate.subject_id:
        errors.append("协变量主体 ID 不能为空")
    
    return errors


def validate_entities(entities: list[Entity]) -> dict[str, list[str]]:
    """批量验证实体列表。
    
    Args:
        entities: 实体列表
        
    Returns:
        dict: 实体 ID 到错误消息列表的映射（只包含有错误的实体）
    """
    errors_by_id = {}
    for entity in entities:
        errors = validate_entity(entity)
        if errors:
            errors_by_id[entity.id] = errors
    return errors_by_id


def validate_relationships(
    relationships: list[Relationship]
) -> dict[str, list[str]]:
    """批量验证关系列表。
    
    Args:
        relationships: 关系列表
        
    Returns:
        dict: 关系 ID 到错误消息列表的映射（只包含有错误的关系）
    """
    errors_by_id = {}
    for relationship in relationships:
        errors = validate_relationship(relationship)
        if errors:
            errors_by_id[relationship.id] = errors
    return errors_by_id


def validate_communities(communities: list[Community]) -> dict[str, list[str]]:
    """批量验证社区列表。
    
    Args:
        communities: 社区列表
        
    Returns:
        dict: 社区 ID 到错误消息列表的映射（只包含有错误的社区）
    """
    errors_by_id = {}
    for community in communities:
        errors = validate_community(community)
        if errors:
            errors_by_id[community.id] = errors
    return errors_by_id


def validate_community_reports(
    reports: list[CommunityReport]
) -> dict[str, list[str]]:
    """批量验证社区报告列表。
    
    Args:
        reports: 社区报告列表
        
    Returns:
        dict: 报告 ID 到错误消息列表的映射（只包含有错误的报告）
    """
    errors_by_id = {}
    for report in reports:
        errors = validate_community_report(report)
        if errors:
            errors_by_id[report.id] = errors
    return errors_by_id


def validate_text_units(text_units: list[TextUnit]) -> dict[str, list[str]]:
    """批量验证文本单元列表。
    
    Args:
        text_units: 文本单元列表
        
    Returns:
        dict: 文本单元 ID 到错误消息列表的映射（只包含有错误的文本单元）
    """
    errors_by_id = {}
    for text_unit in text_units:
        errors = validate_text_unit(text_unit)
        if errors:
            errors_by_id[text_unit.id] = errors
    return errors_by_id


def validate_documents(documents: list[Document]) -> dict[str, list[str]]:
    """批量验证文档列表。
    
    Args:
        documents: 文档列表
        
    Returns:
        dict: 文档 ID 到错误消息列表的映射（只包含有错误的文档）
    """
    errors_by_id = {}
    for document in documents:
        errors = validate_document(document)
        if errors:
            errors_by_id[document.id] = errors
    return errors_by_id


def validate_covariates(covariates: list[Covariate]) -> dict[str, list[str]]:
    """批量验证协变量列表。
    
    Args:
        covariates: 协变量列表
        
    Returns:
        dict: 协变量 ID 到错误消息列表的映射（只包含有错误的协变量）
    """
    errors_by_id = {}
    for covariate in covariates:
        errors = validate_covariate(covariate)
        if errors:
            errors_by_id[covariate.id] = errors
    return errors_by_id

