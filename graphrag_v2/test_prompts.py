"""测试 Prompt 模板和 GLM 集成。"""

import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from graphrag_v2.llm import GLMClient
from graphrag_v2.prompts import (
    get_community_report_prompt,
    get_entity_extraction_prompt,
    get_global_search_map_prompt,
    get_global_search_reduce_prompt,
    get_local_search_prompt,
)


def test_entity_extraction_prompt():
    """测试实体提取 Prompt。"""
    print("\n" + "=" * 80)
    print("测试 1: 实体提取 Prompt")
    print("=" * 80)
    
    # 测试文本
    input_text = """
    GraphRAG 是微软研究院开发的一种创新技术，它将知识图谱与检索增强生成（RAG）相结合。
    该技术由微软的研究团队在2024年发布，旨在提升大语言模型在复杂问答任务中的表现。
    GraphRAG 使用 Leiden 算法进行社区检测，并生成详细的社区报告。
    """
    
    # 生成 Prompt
    prompt = get_entity_extraction_prompt(
        entity_types=["组织", "技术", "算法", "人物"],
        input_text=input_text,
        include_examples=False,  # 不包含示例以节省空间
    )
    
    print(f"\n生成的 Prompt 长度: {len(prompt)} 字符")
    print(f"\n前 500 字符:\n{prompt[:500]}...")
    
    # 使用 GLM 客户端调用
    print("\n调用 GLM API...")
    client = GLMClient()
    
    messages = [
        {"role": "system", "content": "你是一个专业的信息提取助手。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages, temperature=0.3)
    
    print(f"\nGLM 响应:\n{response}")
    
    # 解析响应
    print("\n解析实体和关系...")
    lines = response.split("<|>")
    entities = [line for line in lines if line.strip().startswith('("entity"')]
    relationships = [line for line in lines if line.strip().startswith('("relationship"')]
    
    print(f"[OK] 提取到 {len(entities)} 个实体")
    print(f"[OK] 提取到 {len(relationships)} 个关系")


def test_community_report_prompt():
    """测试社区报告 Prompt。"""
    print("\n" + "=" * 80)
    print("测试 2: 社区报告 Prompt")
    print("=" * 80)
    
    # 测试数据
    input_text = """
实体

id,entity,description
1,GraphRAG,一种结合知识图谱和检索增强生成的技术
2,微软,GraphRAG 的开发者
3,Leiden算法,用于社区检测的算法

关系

id,source,target,description
1,微软,GraphRAG,微软开发了 GraphRAG 技术
2,GraphRAG,Leiden算法,GraphRAG 使用 Leiden 算法进行社区检测
"""
    
    # 生成 Prompt
    prompt = get_community_report_prompt(
        input_text=input_text,
        role="技术分析师",
        report_length="300-500字",
    )
    
    print(f"\n生成的 Prompt 长度: {len(prompt)} 字符")
    
    # 使用 GLM 客户端调用
    print("\n调用 GLM API...")
    client = GLMClient()
    
    messages = [
        {"role": "system", "content": "你是一个专业的数据分析师。"},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(messages, temperature=0.5)
    
    print(f"\nGLM 响应:\n{response}")
    
    # 解析 JSON 响应
    try:
        report = json.loads(response)
        print("\n解析的报告:")
        print(f"  标题: {report.get('title', 'N/A')}")
        print(f"  评分: {report.get('rating', 'N/A')}")
        print(f"  发现数量: {len(report.get('findings', []))}")
        print("[OK] JSON 格式正确")
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON 解析失败: {e}")


def test_global_search_prompts():
    """测试 Global Search Prompt。"""
    print("\n" + "=" * 80)
    print("测试 3: Global Search Prompt (Map + Reduce)")
    print("=" * 80)
    
    # 测试 Map 阶段
    print("\n--- Map 阶段 ---")
    context_data = """
报告 ID: 1
标题: GraphRAG 技术社区
摘要: GraphRAG 是一种结合知识图谱和 RAG 的技术，由微软开发。

报告 ID: 2
标题: 社区检测算法
摘要: Leiden 算法用于检测社区结构，在 GraphRAG 中发挥重要作用。
"""
    
    map_prompt = get_global_search_map_prompt(
        context_data=context_data,
        max_length=200,
    )
    
    client = GLMClient()
    
    messages = [
        {"role": "system", "content": "你是一个专业的数据分析师。"},
        {"role": "user", "content": map_prompt + "\n\n问题: GraphRAG 的主要特点是什么？"}
    ]
    
    map_response = client.chat_completion(messages, temperature=0.3)
    print(f"\nMap 响应:\n{map_response}")
    
    # 测试 Reduce 阶段
    print("\n--- Reduce 阶段 ---")
    reduce_prompt = get_global_search_reduce_prompt(
        report_data=f"分析师报告 1:\n{map_response}\n\n分析师报告 2:\n{map_response}",
        response_type="简短段落",
        max_length=300,
    )
    
    messages = [
        {"role": "system", "content": "你是一个专业的数据分析师。"},
        {"role": "user", "content": reduce_prompt + "\n\n问题: GraphRAG 的主要特点是什么？"}
    ]
    
    reduce_response = client.chat_completion(messages, temperature=0.5)
    print(f"\nReduce 响应:\n{reduce_response}")


def test_local_search_prompt():
    """测试 Local Search Prompt。"""
    print("\n" + "=" * 80)
    print("测试 4: Local Search Prompt")
    print("=" * 80)
    
    # 测试数据
    context_data = """
实体表:
id,entity,type,description
1,GraphRAG,技术,结合知识图谱和 RAG 的技术
2,微软,组织,GraphRAG 的开发者
3,Leiden算法,算法,用于社区检测

关系表:
id,source,target,description
1,微软,GraphRAG,微软开发了 GraphRAG
2,GraphRAG,Leiden算法,GraphRAG 使用 Leiden 算法

社区表:
id,title,summary
1,GraphRAG 技术生态,包含 GraphRAG 及其相关技术和组织
"""
    
    # 生成 Prompt
    prompt = get_local_search_prompt(
        context_data=context_data,
        response_type="简短段落",
    )
    
    print(f"\n生成的 Prompt 长度: {len(prompt)} 字符")
    
    # 使用 GLM 客户端调用
    print("\n调用 GLM API...")
    client = GLMClient()
    
    messages = [
        {"role": "system", "content": "你是一个专业的问答助手。"},
        {"role": "user", "content": prompt + "\n\n问题: GraphRAG 是什么？"}
    ]
    
    response = client.chat_completion(messages, temperature=0.5)
    
    print(f"\nGLM 响应:\n{response}")


def test_glm_stats():
    """测试 GLM 统计信息。"""
    print("\n" + "=" * 80)
    print("GLM 客户端统计信息")
    print("=" * 80)
    
    client = GLMClient()
    stats = client.get_stats()
    
    print(f"\n总调用次数: {stats['total_calls']}")
    print(f"总 Token 数: {stats['total_tokens']}")
    print(f"总错误次数: {stats['total_errors']}")


def main():
    """运行所有测试。"""
    print("\n" + "=" * 80)
    print("GraphRAG v2 - Prompt 模板和 GLM 集成测试")
    print("=" * 80)
    
    try:
        # 测试 1: 实体提取
        test_entity_extraction_prompt()
        
        # 测试 2: 社区报告
        test_community_report_prompt()
        
        # 测试 3: Global Search
        test_global_search_prompts()
        
        # 测试 4: Local Search
        test_local_search_prompt()
        
        # 测试 5: 统计信息
        test_glm_stats()
        
        print("\n" + "=" * 80)
        print("[OK] 所有测试完成！")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

