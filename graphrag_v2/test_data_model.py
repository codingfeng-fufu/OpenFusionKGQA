"""测试数据模型。

验证数据模型的创建、转换和验证功能。
"""

import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from graphrag_v2.data_model import (
    Community,
    CommunityReport,
    Covariate,
    Document,
    Entity,
    Relationship,
    TextUnit,
)
from graphrag_v2.data_model.converters import (
    communities_to_dataframe,
    dataframe_to_communities,
    dataframe_to_entities,
    dataframe_to_relationships,
    entities_to_dataframe,
    relationships_to_dataframe,
)
from graphrag_v2.data_model.validators import (
    validate_community,
    validate_entity,
    validate_relationship,
)


def print_section(title: str) -> None:
    """打印分节标题。"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_entity_creation():
    """测试实体创建。"""
    print_section("测试 1: 实体创建")
    
    try:
        # 创建实体
        entity = Entity(
            id="entity-1",
            short_id="E1",
            title="张三",
            type="person",
            description="一个测试人物",
            rank=10,
            community_ids=["community-1"],
            text_unit_ids=["text-1", "text-2"],
        )
        
        print("\n✓ 成功创建实体")
        print(f"  - ID: {entity.id}")
        print(f"  - 标题: {entity.title}")
        print(f"  - 类型: {entity.type}")
        print(f"  - 排名: {entity.rank}")
        
        # 验证实体
        errors = validate_entity(entity)
        if errors:
            print(f"\n✗ 实体验证失败: {errors}")
        else:
            print("\n✓ 实体验证通过")
        
        # 测试 from_dict
        entity_dict = {
            "id": "entity-2",
            "title": "李四",
            "type": "person",
            "description": "另一个测试人物",
            "degree": 5,
        }
        entity2 = Entity.from_dict(entity_dict)
        print(f"\n✓ 从字典创建实体: {entity2.title}, 排名: {entity2.rank}")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


def test_relationship_creation():
    """测试关系创建。"""
    print_section("测试 2: 关系创建")
    
    try:
        # 创建关系
        relationship = Relationship(
            id="rel-1",
            short_id="R1",
            source="张三",
            target="李四",
            description="认识",
            weight=0.8,
            rank=5,
        )
        
        print("\n✓ 成功创建关系")
        print(f"  - ID: {relationship.id}")
        print(f"  - 源: {relationship.source}")
        print(f"  - 目标: {relationship.target}")
        print(f"  - 权重: {relationship.weight}")
        
        # 验证关系
        errors = validate_relationship(relationship)
        if errors:
            print(f"\n✗ 关系验证失败: {errors}")
        else:
            print("\n✓ 关系验证通过")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


def test_community_creation():
    """测试社区创建。"""
    print_section("测试 3: 社区创建")
    
    try:
        # 创建社区
        community = Community(
            id="community-1",
            short_id="C1",
            title="人物社区",
            level="0",
            parent="root",
            children=["community-2", "community-3"],
            entity_ids=["entity-1", "entity-2"],
            size=10,
        )
        
        print("\n✓ 成功创建社区")
        print(f"  - ID: {community.id}")
        print(f"  - 标题: {community.title}")
        print(f"  - 层级: {community.level}")
        print(f"  - 大小: {community.size}")
        
        # 验证社区
        errors = validate_community(community)
        if errors:
            print(f"\n✗ 社区验证失败: {errors}")
        else:
            print("\n✓ 社区验证通过")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


def test_dataframe_conversion():
    """测试 DataFrame 转换。"""
    print_section("测试 4: DataFrame 转换")
    
    try:
        # 创建实体列表
        entities = [
            Entity(
                id=f"entity-{i}",
                short_id=f"E{i}",
                title=f"实体{i}",
                type="person",
                rank=i,
            )
            for i in range(3)
        ]
        
        # 转换为 DataFrame
        df = entities_to_dataframe(entities)
        print(f"\n✓ 成功转换为 DataFrame")
        print(f"  - 行数: {len(df)}")
        print(f"  - 列数: {len(df.columns)}")
        print(f"\n前几行:")
        print(df[["id", "title", "type", "rank"]].head())
        
        # 转换回实体列表
        entities_back = dataframe_to_entities(df)
        print(f"\n✓ 成功转换回实体列表")
        print(f"  - 实体数量: {len(entities_back)}")
        print(f"  - 第一个实体: {entities_back[0].title}")
        
        # 验证数据一致性
        if entities[0].title == entities_back[0].title:
            print("\n✓ 数据转换一致性验证通过")
        else:
            print("\n✗ 数据转换一致性验证失败")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_text_unit_and_document():
    """测试文本单元和文档。"""
    print_section("测试 5: 文本单元和文档")
    
    try:
        # 创建文档
        document = Document(
            id="doc-1",
            short_id="D1",
            title="测试文档",
            text="这是一个测试文档的内容。",
            type="text",
        )
        
        print("\n✓ 成功创建文档")
        print(f"  - ID: {document.id}")
        print(f"  - 标题: {document.title}")
        print(f"  - 类型: {document.type}")
        
        # 创建文本单元
        text_unit = TextUnit(
            id="text-1",
            short_id="T1",
            text="这是一个文本单元。",
            n_tokens=10,
            document_ids=["doc-1"],
            entity_ids=["entity-1"],
        )
        
        print("\n✓ 成功创建文本单元")
        print(f"  - ID: {text_unit.id}")
        print(f"  - Token 数: {text_unit.n_tokens}")
        print(f"  - 文档数: {len(text_unit.document_ids)}")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


def test_community_report():
    """测试社区报告。"""
    print_section("测试 6: 社区报告")
    
    try:
        # 创建社区报告
        report = CommunityReport(
            id="report-1",
            short_id="CR1",
            title="人物社区报告",
            community_id="community-1",
            summary="这是一个关于人物的社区摘要。",
            full_content="这是完整的社区报告内容...",
            rank=8.5,
        )
        
        print("\n✓ 成功创建社区报告")
        print(f"  - ID: {report.id}")
        print(f"  - 标题: {report.title}")
        print(f"  - 社区 ID: {report.community_id}")
        print(f"  - 排名: {report.rank}")
        print(f"  - 摘要长度: {len(report.summary)} 字符")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")


def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("GraphRAG v2 数据模型测试")
    print("=" * 60)
    
    test_entity_creation()
    test_relationship_creation()
    test_community_creation()
    test_dataframe_conversion()
    test_text_unit_and_document()
    test_community_report()
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

