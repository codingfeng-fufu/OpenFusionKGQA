# 如何上传到GitHub

## 📦 已准备好的文件

以下文件将会上传到GitHub，供同学B使用：

### 核心文档
- ✅ `给同学B的任务说明.md` - 快速了解项目
- ✅ `实现步骤_成员B.md` - 详细实现步骤
- ✅ `README.md` - 项目概览
- ✅ `COLLABORATION.md` - 协作规范
- ✅ `QUICK_START.md` - 快速开始

### 代码和配置
- ✅ `src/interfaces.py` - 接口定义
- ✅ `config/ketrag_config.yaml` - 配置文件
- ✅ `requirements.txt` - 依赖包
- ✅ `setup.sh` - 环境搭建脚本

### 文档
- ✅ `docs/member_B_tasks.md` - 详细任务说明
- ✅ `docs/weekly_progress.md` - 进度跟踪

---

## 🚀 上传步骤

### 方法1: 使用自动脚本（推荐）

#### Windows系统
```bash
# 在graphrag-optimization目录下运行
.\upload_to_git.bat
```

#### Linux/Mac系统
```bash
# 在graphrag-optimization目录下运行
chmod +x upload_to_git.sh
./upload_to_git.sh
```

### 方法2: 手动上传

```bash
# 1. 进入项目目录
cd graphrag-optimization

# 2. 初始化Git仓库（如果还没有）
git init

# 3. 添加远程仓库
git remote add origin https://github.com/codingfeng-fufu/KETGraphRAG.git

# 4. 添加文件
git add "给同学B的任务说明.md"
git add "实现步骤_成员B.md"
git add README.md
git add COLLABORATION.md
git add QUICK_START.md
git add src/interfaces.py
git add config/ketrag_config.yaml
git add requirements.txt
git add setup.sh
git add docs/member_B_tasks.md
git add docs/weekly_progress.md

# 5. 创建.gitignore
cat > .gitignore << EOF
__pycache__/
*.pyc
venv/
.idea/
results/
data/
*.pkl
EOF

git add .gitignore

# 6. 提交
git commit -m "初始化KET-RAG项目 - 成员B任务文档和接口定义"

# 7. 设置主分支
git branch -M main

# 8. 推送到GitHub
git push -u origin main
```

---

## 🔑 GitHub认证

### 如果提示需要认证

#### 方法1: 使用Personal Access Token（推荐）
1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成token并复制
5. 推送时，用户名输入你的GitHub用户名，密码输入token

#### 方法2: 使用SSH
```bash
# 1. 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 2. 添加到GitHub
# 复制公钥内容
cat ~/.ssh/id_ed25519.pub

# 3. 访问 https://github.com/settings/keys
# 点击 "New SSH key"，粘贴公钥

# 4. 修改远程仓库URL
git remote set-url origin git@github.com:codingfeng-fufu/KETGraphRAG.git

# 5. 推送
git push -u origin main
```

---

## ✅ 验证上传成功

上传成功后，访问：
https://github.com/codingfeng-fufu/KETGraphRAG

你应该能看到以下文件：
- 给同学B的任务说明.md
- 实现步骤_成员B.md
- README.md
- src/interfaces.py
- 等等...

---

## 📞 发送给同学B

上传成功后，发送以下消息给同学B：

```
Hi [同学B]！

项目已经上传到GitHub了：
https://github.com/codingfeng-fufu/KETGraphRAG

克隆项目：
git clone https://github.com/codingfeng-fufu/KETGraphRAG.git

必读文档（按顺序）：
1. 给同学B的任务说明.md ⭐
2. src/interfaces.py ⭐
3. 实现步骤_成员B.md ⭐

有问题随时找我！
```

---

## 🔄 后续更新

如果你需要更新文件：

```bash
# 1. 修改文件后
git add .

# 2. 提交
git commit -m "更新任务说明"

# 3. 推送
git push
```

---

## ⚠️ 注意事项

1. **不要上传敏感信息**
   - API密钥
   - 密码
   - 个人信息

2. **不要上传大文件**
   - 数据集（放在data/目录，已在.gitignore中）
   - 模型文件
   - 结果文件（放在results/目录，已在.gitignore中）

3. **保持仓库整洁**
   - 只上传必要的文件
   - 使用.gitignore排除临时文件

---

**准备好了吗？运行脚本开始上传吧！** 🚀

