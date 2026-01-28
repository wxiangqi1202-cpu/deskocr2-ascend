# GitHub 发布指南

## ✅ 项目准备就绪

项目文件已完整，包含：
- ✅ README.md (完整项目说明)
- ✅ LICENSE (Apache 2.0)
- ✅ requirements.txt (依赖列表)
- ✅ .gitignore (忽略规则)
- ✅ 核心代码文件
- ✅ 文档和示例

---

## 🚀 发布步骤

### 步骤 1: 初始化 Git 仓库

```bash
cd /home/wxq/deskocr-ascend

# 初始化 Git
git init

# 设置用户信息（如果未设置）
git config user.name "你的名字"
git config user.email "你的邮箱@example.com"
```

### 步骤 2: 添加文件到 Git

```bash
# 添加所有文件（.gitignore 会自动排除大文件）
git add .

# 查看将要提交的文件
git status
```

### 步骤 3: 提交代码

```bash
git commit -m "Initial commit: DeepSeek-OCR on Ascend NPU

- Complete NPU deployment with custom Conv2D operator
- 100% success rate in benchmark tests
- Comprehensive documentation and examples
- Performance: 31.9s/image (CANN 8.3.RC1)"
```

### 步骤 4: 在 GitHub 创建仓库

1. 访问: https://github.com/new

2. 填写仓库信息：
   - **Repository name**: `deskocr-ascend`
   - **Description**: `DeepSeek-OCR deployment on Ascend 910B2 NPU with custom operators`
   - **Visibility**: Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"

3. 点击 "Create repository"

### 步骤 5: 关联远程仓库

```bash
# 替换为你的 GitHub 用户名
git remote add origin https://github.com/你的用户名/deskocr-ascend.git

# 验证远程仓库
git remote -v
```

### 步骤 6: 推送到 GitHub

```bash
# 重命名主分支为 main
git branch -M main

# 推送代码
git push -u origin main
```

---

## ⚠️ 重要说明

### 模型文件不会上传

`.gitignore` 已配置排除以下文件：
- `model/*.safetensors` (6.3GB 模型权重)
- `model/*.bin`
- `*.log`
- `__pycache__/`
- 备份目录

用户需要自行从 Hugging Face 下载模型文件。

### 如果需要上传大文件

使用 Git LFS (不推荐，模型太大):

```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "model/*.safetensors"

# 提交 .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

---

## 📝 后续维护

### 更新代码

```bash
# 修改文件后
git add .
git commit -m "描述你的修改"
git push
```

### 创建 Release

1. 在 GitHub 仓库页面点击 "Releases"
2. 点击 "Create a new release"
3. 填写信息：
   - **Tag version**: `v1.0.0`
   - **Release title**: `v1.0.0 - Initial Release`
   - **Description**: 复制从 FINAL_BENCHMARK_REPORT.md
4. 点击 "Publish release"

### 添加 Topics

在 GitHub 仓库页面添加标签：
- `ascend-npu`
- `deepseek`
- `ocr`
- `pytorch`
- `huawei-ascend`
- `npu-acceleration`

---

## ✨ 优化建议

### 添加 README Badges

在 README.md 中更新徒章链接：

```markdown
[![Stars](https://img.shields.io/github/stars/你的用户名/deskocr-ascend)](https://github.com/你的用户名/deskocr-ascend/stargazers)
[![Issues](https://img.shields.io/github/issues/你的用户名/deskocr-ascend)](https://github.com/你的用户名/deskocr-ascend/issues)
[![License](https://img.shields.io/github/license/你的用户名/deskocr-ascend)](./LICENSE)
```

### 创建 GitHub Actions CI/CD

在 `.github/workflows/test.yml` 中添加自动化测试（可选）

### 添加贡献指南

创建 `CONTRIBUTING.md` 文件说明如何贡献代码

---

## 📞 需要帮助？

如果遇到问题：

1. **Git 错误**: 检查网络连接和 GitHub 凭据
2. **文件太大**: 确认 .gitignore 配置正确
3. **推送失败**: 检查远程仓库 URL 是否正确

---

## ✅ 检查清单

发布前请确认：

- [ ] README.md 内容完整
- [ ] LICENSE 文件存在
- [ ] .gitignore 配置正确
- [ ] requirements.txt 列出所有依赖
- [ ] 模型下载说明清晰
- [ ] 示例代码可运行
- [ ] 文档链接正确
- [ ] GitHub 仓库信息已更新

---

**祝发布顺利！🎉**
