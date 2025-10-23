@echo off
REM Git上传脚本 - 只上传给同学B需要的文件

echo ========================================
echo 准备上传到 GitHub
echo ========================================

REM 检查是否已经是Git仓库
if not exist .git (
    echo 初始化Git仓库...
    git init
    git remote add origin https://github.com/codingfeng-fufu/KETGraphRAG.git
) else (
    echo Git仓库已存在
)

echo.
echo ========================================
echo 添加文件到Git
echo ========================================

REM 核心文档（给同学B的）
echo 添加核心文档...
git add "给同学B的任务说明.md"
git add "实现步骤_成员B.md"
git add README.md
git add COLLABORATION.md
git add QUICK_START.md

REM 代码和配置
echo 添加代码和配置...
git add src/interfaces.py
git add config/ketrag_config.yaml
git add requirements.txt
git add setup.sh

REM 文档
echo 添加文档...
git add docs/member_B_tasks.md
git add docs/weekly_progress.md

REM 创建.gitignore
echo 创建.gitignore...
echo __pycache__/ > .gitignore
echo *.pyc >> .gitignore
echo *.pyo >> .gitignore
echo *.pyd >> .gitignore
echo .Python >> .gitignore
echo venv/ >> .gitignore
echo env/ >> .gitignore
echo .vscode/ >> .gitignore
echo .idea/ >> .gitignore
echo *.log >> .gitignore
echo .DS_Store >> .gitignore
echo results/ >> .gitignore
echo data/ >> .gitignore
echo *.pkl >> .gitignore
echo *.pickle >> .gitignore

git add .gitignore

echo.
echo ========================================
echo 查看将要提交的文件
echo ========================================
git status

echo.
echo ========================================
echo 提交到本地仓库
echo ========================================
git commit -m "初始化KET-RAG项目 - 成员B任务文档和接口定义"

echo.
echo ========================================
echo 推送到GitHub
echo ========================================
echo 注意：如果是第一次推送，可能需要输入GitHub用户名和密码
echo 或者使用Personal Access Token
echo.

REM 设置主分支为main
git branch -M main

REM 推送到远程仓库
git push -u origin main

echo.
echo ========================================
echo 完成！
echo ========================================
echo 项目已上传到: https://github.com/codingfeng-fufu/KETGraphRAG.git
echo.
echo 你可以把这个链接发给同学B：
echo https://github.com/codingfeng-fufu/KETGraphRAG.git
echo.
pause

