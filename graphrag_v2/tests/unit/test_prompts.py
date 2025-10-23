"""
Prompt 模块单元测试
"""
import pytest

from graphrag_v2.prompts.base import PromptTemplate, PromptLibrary
from graphrag_v2.prompts import (
    get_entity_extraction_prompt,
    get_community_report_prompt,
    get_global_search_map_prompt,
    get_global_search_reduce_prompt,
    get_local_search_prompt,
)


class TestPromptTemplate:
    """测试 PromptTemplate 类"""
    
    def test_simple_variable_substitution(self):
        """测试简单变量替换"""
        template = PromptTemplate("Hello {name}!")
        result = template.format(name="Alice")
        assert result == "Hello Alice!"
        
    def test_default_value(self):
        """测试默认值"""
        template = PromptTemplate("Hello {name:World}!")
        
        # 提供值
        result = template.format(name="Alice")
        assert result == "Hello Alice!"
        
        # 使用默认值
        result = template.format()
        assert result == "Hello World!"
        
    def test_multiple_variables(self):
        """测试多个变量"""
        template = PromptTemplate("Hello {name}, you are {age} years old.")
        result = template.format(name="Alice", age=25)
        assert result == "Hello Alice, you are 25 years old."
        
    def test_conditional_rendering_true(self):
        """测试条件渲染（真）"""
        template = PromptTemplate(
            "Hello!{?premium} You are a premium user.{/premium}"
        )
        result = template.format(premium=True)
        assert "premium user" in result
        
    def test_conditional_rendering_false(self):
        """测试条件渲染（假）"""
        template = PromptTemplate(
            "Hello!{?premium} You are a premium user.{/premium}"
        )
        result = template.format(premium=False)
        assert "premium user" not in result
        
    def test_list_to_string_conversion(self):
        """测试列表转字符串"""
        template = PromptTemplate("Types: {types}")
        result = template.format(types=["A", "B", "C"])
        assert result == "Types: A, B, C"
        
    def test_missing_variable(self):
        """测试缺失变量"""
        template = PromptTemplate("Hello {name}!")
        # 缺失变量应该保持原样
        result = template.format()
        assert "{name}" in result


class TestPromptLibrary:
    """测试 PromptLibrary 类"""
    
    def test_register_and_get(self):
        """测试注册和获取模板"""
        library = PromptLibrary()
        library.register("greeting", "Hello {name}!")
        
        template = library.get("greeting")
        assert template is not None
        assert isinstance(template, PromptTemplate)
        
    def test_format_template(self):
        """测试格式化模板"""
        library = PromptLibrary()
        library.register("greeting", "Hello {name}!")

        result = library.format("greeting", name="Alice")
        assert result == "Hello Alice!"

    def test_list_templates(self):
        """测试列出所有模板"""
        library = PromptLibrary()
        library.register("greeting", "Hello!")
        library.register("farewell", "Goodbye!")

        templates = library.list_prompts()  # 修正方法名
        assert len(templates) == 2
        assert "greeting" in templates
        assert "farewell" in templates

    def test_get_nonexistent_template(self):
        """测试获取不存在的模板"""
        library = PromptLibrary()

        # 应该抛出 KeyError
        with pytest.raises(KeyError):
            library.get("nonexistent")


class TestEntityExtractionPrompt:
    """测试实体提取 Prompt"""
    
    def test_basic_prompt(self):
        """测试基础 Prompt"""
        prompt = get_entity_extraction_prompt(
            entity_types=["组织", "技术"],
            input_text="GraphRAG 是微软开发的技术。",
        )
        
        assert prompt is not None
        assert "组织" in prompt
        assert "技术" in prompt
        assert "GraphRAG" in prompt
        
    def test_with_examples(self):
        """测试包含示例的 Prompt"""
        prompt = get_entity_extraction_prompt(
            entity_types=["组织"],
            input_text="测试文本",
            include_examples=True,
        )
        
        assert prompt is not None
        assert len(prompt) > 500  # 包含示例应该更长
        
    def test_without_examples(self):
        """测试不包含示例的 Prompt"""
        prompt = get_entity_extraction_prompt(
            entity_types=["组织"],
            input_text="测试文本",
            include_examples=False,
        )
        
        assert prompt is not None
        assert len(prompt) < 2000  # 不包含示例应该较短
        
    def test_custom_delimiters(self):
        """测试自定义分隔符"""
        prompt = get_entity_extraction_prompt(
            entity_types=["组织"],
            input_text="测试",
            tuple_delimiter="||",
            record_delimiter="##",
            completion_delimiter="##DONE##",
        )
        
        assert "||" in prompt
        assert "##" in prompt
        assert "##DONE##" in prompt


class TestCommunityReportPrompt:
    """测试社区报告 Prompt"""
    
    def test_basic_prompt(self):
        """测试基础 Prompt"""
        prompt = get_community_report_prompt(
            input_text="实体和关系数据",
        )
        
        assert prompt is not None
        assert "实体和关系数据" in prompt
        assert "JSON" in prompt or "json" in prompt
        
    def test_with_role(self):
        """测试带角色的 Prompt"""
        prompt = get_community_report_prompt(
            input_text="数据",
            role="技术分析师",
        )
        
        assert "技术分析师" in prompt
        
    def test_with_report_length(self):
        """测试带报告长度的 Prompt"""
        prompt = get_community_report_prompt(
            input_text="数据",
            report_length="500-1000字",
        )
        
        assert "500-1000字" in prompt


class TestGlobalSearchPrompts:
    """测试 Global Search Prompt"""
    
    def test_map_prompt(self):
        """测试 Map 阶段 Prompt"""
        prompt = get_global_search_map_prompt(
            context_data="社区报告数据",
        )
        
        assert prompt is not None
        assert "社区报告数据" in prompt
        
    def test_map_prompt_with_max_length(self):
        """测试带最大长度的 Map Prompt"""
        prompt = get_global_search_map_prompt(
            context_data="数据",
            max_length=200,
        )
        
        assert "200" in prompt
        
    def test_reduce_prompt(self):
        """测试 Reduce 阶段 Prompt"""
        prompt = get_global_search_reduce_prompt(
            report_data="分析师报告",
        )
        
        assert prompt is not None
        assert "分析师报告" in prompt
        
    def test_reduce_prompt_with_response_type(self):
        """测试带响应类型的 Reduce Prompt"""
        prompt = get_global_search_reduce_prompt(
            report_data="数据",
            response_type="简短段落",
        )
        
        assert "简短段落" in prompt


class TestLocalSearchPrompt:
    """测试 Local Search Prompt"""
    
    def test_basic_prompt(self):
        """测试基础 Prompt"""
        prompt = get_local_search_prompt(
            context_data="实体和关系数据",
        )
        
        assert prompt is not None
        assert "实体和关系数据" in prompt
        
    def test_with_response_type(self):
        """测试带响应类型的 Prompt"""
        prompt = get_local_search_prompt(
            context_data="数据",
            response_type="详细段落",
        )
        
        assert "详细段落" in prompt


class TestPromptIntegration:
    """测试 Prompt 集成"""
    
    def test_all_prompts_generate_valid_output(self):
        """测试所有 Prompt 都能生成有效输出"""
        # 实体提取
        entity_prompt = get_entity_extraction_prompt(
            entity_types=["组织"],
            input_text="测试",
        )
        assert len(entity_prompt) > 0
        
        # 社区报告
        report_prompt = get_community_report_prompt(
            input_text="测试",
        )
        assert len(report_prompt) > 0
        
        # Global Search Map
        map_prompt = get_global_search_map_prompt(
            context_data="测试",
        )
        assert len(map_prompt) > 0
        
        # Global Search Reduce
        reduce_prompt = get_global_search_reduce_prompt(
            report_data="测试",
        )
        assert len(reduce_prompt) > 0
        
        # Local Search
        local_prompt = get_local_search_prompt(
            context_data="测试",
        )
        assert len(local_prompt) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

