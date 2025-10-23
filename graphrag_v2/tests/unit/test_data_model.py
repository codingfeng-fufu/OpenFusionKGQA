"""
数据模型单元测试
"""
import pytest
import pandas as pd

from graphrag_v2.data_model import (
    Document,
    Entity,
    Relationship,
    Community,
    CommunityReport,
    TextUnit,
    Covariate,
)
from graphrag_v2.data_model.converters import (
    documents_to_dataframe,
    entities_to_dataframe,
    relationships_to_dataframe,
    communities_to_dataframe,
)
from graphrag_v2.data_model.validators import (
    validate_document,
    validate_entity,
    validate_relationship,
    validate_community,
)


class TestDocument:
    """测试 Document 数据类"""
    
    def test_create_document(self):
        """测试创建文档"""
        doc = Document(
            id="doc1",
            short_id="doc1",
            title="测试文档",
            text="这是一个测试文档。",
        )

        assert doc.id == "doc1"
        assert doc.text == "这是一个测试文档。"
        assert doc.title == "测试文档"

    def test_document_with_attributes(self):
        """测试带属性的文档"""
        doc = Document(
            id="doc1",
            short_id="doc1",
            title="测试文档",
            text="测试文本",
            attributes={"source": "test", "date": "2024-01-01"},
        )

        assert doc.attributes is not None
        assert doc.attributes["source"] == "test"
        assert doc.attributes["date"] == "2024-01-01"

    def test_validate_document(self):
        """测试文档验证"""
        doc = Document(id="doc1", short_id="doc1", title="测试", text="测试")
        errors = validate_document(doc)
        assert len(errors) == 0  # 没有错误

        # 测试无效文档
        invalid_doc = Document(id="", short_id="", title="", text="")
        errors = validate_document(invalid_doc)
        assert len(errors) > 0  # 有错误


class TestEntity:
    """测试 Entity 数据类"""
    
    def test_create_entity(self):
        """测试创建实体"""
        entity = Entity(
            id="e1",
            short_id="e1",
            title="GraphRAG",
            type="技术",
            description="一种技术",
        )

        assert entity.id == "e1"
        assert entity.title == "GraphRAG"
        assert entity.type == "技术"
        assert entity.description == "一种技术"

    def test_entity_with_text_unit_ids(self):
        """测试带文本单元 ID 的实体"""
        entity = Entity(
            id="e1",
            short_id="e1",
            title="GraphRAG",
            type="技术",
            text_unit_ids=["tu1", "tu2"],
        )

        assert len(entity.text_unit_ids) == 2
        assert "tu1" in entity.text_unit_ids

    def test_validate_entity(self):
        """测试实体验证"""
        entity = Entity(id="e1", short_id="e1", title="GraphRAG", type="技术")
        errors = validate_entity(entity)
        assert len(errors) == 0  # 没有错误

        # 测试无效实体
        invalid_entity = Entity(id="", short_id="", title="", type="")
        errors = validate_entity(invalid_entity)
        assert len(errors) > 0  # 有错误


class TestRelationship:
    """测试 Relationship 数据类"""
    
    def test_create_relationship(self):
        """测试创建关系"""
        rel = Relationship(
            id="r1",
            short_id="r1",
            source="微软",
            target="GraphRAG",
            description="开发了",
            weight=0.9,
        )

        assert rel.id == "r1"
        assert rel.source == "微软"
        assert rel.target == "GraphRAG"
        assert rel.description == "开发了"
        assert rel.weight == 0.9

    def test_relationship_default_weight(self):
        """测试关系默认权重"""
        rel = Relationship(
            id="r1",
            short_id="r1",
            source="A",
            target="B",
        )

        assert rel.weight == 1.0

    def test_validate_relationship(self):
        """测试关系验证"""
        rel = Relationship(id="r1", short_id="r1", source="A", target="B")
        errors = validate_relationship(rel)
        assert len(errors) == 0  # 没有错误

        # 测试无效关系
        invalid_rel = Relationship(id="", short_id="", source="", target="")
        errors = validate_relationship(invalid_rel)
        assert len(errors) > 0  # 有错误


class TestCommunity:
    """测试 Community 数据类"""
    
    def test_create_community(self):
        """测试创建社区"""
        community = Community(
            id="c1",
            short_id="c1",
            title="测试社区",
            level="0",
            parent="root",
            children=[],
            entity_ids=["e1", "e2"],
            relationship_ids=["r1"],
        )

        assert community.id == "c1"
        assert community.title == "测试社区"
        assert community.level == "0"
        assert len(community.entity_ids) == 2
        assert len(community.relationship_ids) == 1

    def test_community_default_level(self):
        """测试社区默认层级"""
        community = Community(
            id="c1",
            short_id="c1",
            title="测试社区",
            level="0",
            parent="root",
            children=[],
        )
        assert community.level == "0"

    def test_validate_community(self):
        """测试社区验证"""
        community = Community(
            id="c1",
            short_id="c1",
            title="测试",
            level="0",
            parent="root",
            children=[],
        )
        errors = validate_community(community)
        assert len(errors) == 0  # 没有错误

        # 测试无效社区
        invalid_community = Community(
            id="",
            short_id="",
            title="",
            level="0",
            parent="",
            children=[],
        )
        errors = validate_community(invalid_community)
        assert len(errors) > 0  # 有错误


class TestCommunityReport:
    """测试 CommunityReport 数据类"""
    
    def test_create_community_report(self):
        """测试创建社区报告"""
        report = CommunityReport(
            id="cr1",
            short_id="cr1",
            title="社区报告",
            community_id="c1",
            summary="这是一个摘要",
            full_content="这是完整内容",
            rank=7.5,
        )

        assert report.id == "cr1"
        assert report.community_id == "c1"
        assert report.title == "社区报告"
        assert report.rank == 7.5

    def test_community_report_with_attributes(self):
        """测试带属性的社区报告"""
        report = CommunityReport(
            id="cr1",
            short_id="cr1",
            title="社区报告",
            community_id="c1",
            attributes={
                "findings": [
                    {"summary": "发现1", "explanation": "解释1"},
                    {"summary": "发现2", "explanation": "解释2"},
                ],
            },
        )

        assert report.attributes is not None
        assert len(report.attributes["findings"]) == 2
        assert report.attributes["findings"][0]["summary"] == "发现1"


class TestTextUnit:
    """测试 TextUnit 数据类"""
    
    def test_create_text_unit(self):
        """测试创建文本单元"""
        text_unit = TextUnit(
            id="tu1",
            short_id="tu1",
            text="这是一个文本单元",
            document_ids=["doc1"],
            n_tokens=10,
        )

        assert text_unit.id == "tu1"
        assert text_unit.text == "这是一个文本单元"
        assert text_unit.n_tokens == 10
        assert len(text_unit.document_ids) == 1

    def test_text_unit_with_entities(self):
        """测试带实体的文本单元"""
        text_unit = TextUnit(
            id="tu1",
            short_id="tu1",
            text="测试",
            entity_ids=["e1", "e2"],
            relationship_ids=["r1"],
        )

        assert len(text_unit.entity_ids) == 2
        assert len(text_unit.relationship_ids) == 1


class TestCovariate:
    """测试 Covariate 数据类"""
    
    def test_create_covariate(self):
        """测试创建协变量"""
        covariate = Covariate(
            id="cov1",
            short_id="cov1",
            subject_id="e1",
            covariate_type="claim",
            attributes={"claim": "这是一个声明"},
        )

        assert covariate.id == "cov1"
        assert covariate.subject_id == "e1"
        assert covariate.covariate_type == "claim"
        assert covariate.attributes["claim"] == "这是一个声明"


class TestConverters:
    """测试数据转换器"""
    
    def test_documents_to_dataframe(self, sample_documents):
        """测试文档转 DataFrame"""
        df = documents_to_dataframe(sample_documents)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_documents)
        assert "id" in df.columns
        assert "text" in df.columns
        
    def test_entities_to_dataframe(self, sample_entities):
        """测试实体转 DataFrame"""
        df = entities_to_dataframe(sample_entities)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_entities)
        assert "id" in df.columns
        assert "title" in df.columns  # 使用 title 而不是 name
        assert "type" in df.columns
        
    def test_relationships_to_dataframe(self, sample_relationships):
        """测试关系转 DataFrame"""
        df = relationships_to_dataframe(sample_relationships)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_relationships)
        assert "id" in df.columns
        assert "source" in df.columns
        assert "target" in df.columns
        
    def test_communities_to_dataframe(self, sample_communities):
        """测试社区转 DataFrame"""
        df = communities_to_dataframe(sample_communities)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_communities)
        assert "id" in df.columns
        assert "title" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

