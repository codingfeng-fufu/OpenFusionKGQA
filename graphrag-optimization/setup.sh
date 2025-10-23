#!/bin/bash
# 项目初始化脚本
# 用于快速搭建开发环境

echo "========================================="
echo "GraphRAG优化项目 - 环境初始化"
echo "========================================="

# 检查Python版本
echo ""
echo "检查Python版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $python_version"

# 检查是否满足最低要求 (3.8+)
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python版本过低，需要3.8或更高版本"
    exit 1
fi
echo "✅ Python版本符合要求"

# 创建虚拟环境
echo ""
echo "创建虚拟环境..."
if [ -d "venv" ]; then
    echo "⚠️  虚拟环境已存在"
    read -p "是否重新创建? (y/n): " recreate
    if [ "$recreate" = "y" ]; then
        rm -rf venv
        python -m venv venv
        echo "✅ 虚拟环境已重新创建"
    fi
else
    python -m venv venv
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo ""
echo "激活虚拟环境..."
source venv/bin/activate || . venv/Scripts/activate
echo "✅ 虚拟环境已激活"

# 安装依赖
echo ""
echo "安装依赖包..."
read -p "使用国内镜像源? (推荐) (y/n): " use_mirror

if [ "$use_mirror" = "y" ]; then
    echo "使用清华源安装..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
else
    pip install -r requirements.txt
fi

# 检查安装结果
if [ $? -eq 0 ]; then
    echo "✅ 依赖包安装成功"
else
    echo "❌ 依赖包安装失败"
    exit 1
fi

# 创建必要的目录
echo ""
echo "创建项目目录..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p results/baseline
mkdir -p results/optimized
mkdir -p results/bipartite
mkdir -p results/figures
mkdir -p logs
echo "✅ 目录创建完成"

# 创建.env文件（如果不存在）
if [ ! -f ".env" ]; then
    echo ""
    echo "创建环境配置文件..."
    cat > .env << EOF
# OpenAI API配置
OPENAI_API_KEY=your_api_key_here

# 项目配置
PROJECT_ROOT=$(pwd)
DATA_PATH=./data
RESULTS_PATH=./results

# 日志配置
LOG_LEVEL=INFO
EOF
    echo "✅ .env文件已创建，请编辑并填入你的API密钥"
fi

# 验证安装
echo ""
echo "验证安装..."
python -c "
import sys
sys.path.append('.')
try:
    import numpy as np
    import pandas as pd
    import sklearn
    import networkx as nx
    import jieba
    print('✅ 核心库导入成功')
    
    # 尝试导入接口
    from src.interfaces import BipartiteGraphInterface
    print('✅ 接口定义加载成功')
    
    print()
    print('========================================')
    print('🎉 环境配置完成！')
    print('========================================')
    print()
    print('下一步:')
    print('1. 成员A: 阅读 docs/member_A_tasks.md')
    print('2. 成员B: 阅读 docs/member_B_tasks.md')
    print('3. 开始开发!')
    print()
    print('运行测试:')
    print('  pytest tests/ -v')
    print()
    print('查看项目结构:')
    print('  tree -L 2 -I venv')
    print()
    
except ImportError as e:
    print(f'❌ 导入失败: {e}')
    print('请检查依赖安装是否成功')
    sys.exit(1)
"

# 显示帮助信息
echo ""
echo "========================================="
echo "常用命令:"
echo "========================================="
echo ""
echo "激活虚拟环境:"
echo "  source venv/bin/activate  # Linux/Mac"
echo "  venv\\Scripts\\activate     # Windows"
echo ""
echo "运行测试:"
echo "  pytest tests/ -v"
echo ""
echo "查看接口定义:"
echo "  cat src/interfaces.py"
echo ""
echo "运行演示:"
echo "  python src/interfaces.py"
echo ""
echo "========================================="
