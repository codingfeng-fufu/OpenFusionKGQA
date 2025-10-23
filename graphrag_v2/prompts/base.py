"""Prompt 模板基础类。"""

import re
from typing import Any


class PromptTemplate:
    """Prompt 模板类。
    
    支持：
    - 变量替换：{variable_name}
    - 默认值：{variable_name:default_value}
    - 条件渲染：{?variable_name}content{/variable_name}
    """
    
    def __init__(self, template: str):
        """初始化 Prompt 模板。
        
        Args:
            template: 模板字符串
        """
        self.template = template
    
    def format(self, **kwargs: Any) -> str:
        """格式化模板。
        
        Args:
            **kwargs: 模板变量
            
        Returns:
            str: 格式化后的字符串
        """
        result = self.template
        
        # 处理条件渲染 {?variable}content{/variable}
        result = self._process_conditionals(result, kwargs)
        
        # 处理变量替换 {variable} 或 {variable:default}
        result = self._process_variables(result, kwargs)
        
        return result
    
    def _process_conditionals(self, text: str, variables: dict[str, Any]) -> str:
        """处理条件渲染。
        
        Args:
            text: 文本
            variables: 变量字典
            
        Returns:
            str: 处理后的文本
        """
        # 匹配 {?variable}content{/variable}
        pattern = r'\{\?(\w+)\}(.*?)\{/\1\}'
        
        def replace_conditional(match: re.Match) -> str:
            var_name = match.group(1)
            content = match.group(2)
            
            # 如果变量存在且为真值，保留内容
            if var_name in variables and variables[var_name]:
                return content
            else:
                return ""
        
        return re.sub(pattern, replace_conditional, text, flags=re.DOTALL)
    
    def _process_variables(self, text: str, variables: dict[str, Any]) -> str:
        """处理变量替换。
        
        Args:
            text: 文本
            variables: 变量字典
            
        Returns:
            str: 处理后的文本
        """
        # 匹配 {variable} 或 {variable:default}
        pattern = r'\{(\w+)(?::([^}]*))?\}'
        
        def replace_variable(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)
            
            # 如果变量存在，使用变量值
            if var_name in variables:
                value = variables[var_name]
                # 如果是列表，转换为字符串
                if isinstance(value, list):
                    return ", ".join(str(v) for v in value)
                return str(value)
            # 否则使用默认值
            elif default_value is not None:
                return default_value
            # 如果没有默认值，保留原样
            else:
                return match.group(0)
        
        return re.sub(pattern, replace_variable, text)
    
    def __str__(self) -> str:
        """返回模板字符串。"""
        return self.template


class PromptLibrary:
    """Prompt 库，用于管理多个 Prompt 模板。"""
    
    def __init__(self):
        """初始化 Prompt 库。"""
        self._prompts: dict[str, PromptTemplate] = {}
    
    def register(self, name: str, template: str | PromptTemplate) -> None:
        """注册 Prompt 模板。
        
        Args:
            name: 模板名称
            template: 模板字符串或 PromptTemplate 对象
        """
        if isinstance(template, str):
            template = PromptTemplate(template)
        self._prompts[name] = template
    
    def get(self, name: str) -> PromptTemplate:
        """获取 Prompt 模板。
        
        Args:
            name: 模板名称
            
        Returns:
            PromptTemplate: Prompt 模板
            
        Raises:
            KeyError: 如果模板不存在
        """
        if name not in self._prompts:
            raise KeyError(f"Prompt template '{name}' not found")
        return self._prompts[name]
    
    def format(self, template_name: str, **kwargs: Any) -> str:
        """格式化 Prompt 模板。

        Args:
            template_name: 模板名称
            **kwargs: 模板变量

        Returns:
            str: 格式化后的字符串
        """
        template = self.get(template_name)
        return template.format(**kwargs)
    
    def list_prompts(self) -> list[str]:
        """列出所有 Prompt 模板名称。
        
        Returns:
            list[str]: 模板名称列表
        """
        return list(self._prompts.keys())

