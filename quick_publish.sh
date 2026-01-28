#!/bin/bash

echo "======================================"
echo "  GitHub 快速发布脚本"
echo "======================================"
echo ""

cd /home/wxq/deskocr-ascend

# 检查是否已初始化 Git
if [ -d ".git" ]; then
    echo "⚠️  Git 仓库已存在，跳过初始化"
else
    echo "[1/5] 初始化 Git 仓库..."
    git init
    echo "✅ 完成"
fi

echo ""
echo "[2/5] 添加文件..."
git add .
echo "✅ 完成 - $(git diff --cached --numstat | wc -l) 个文件已添加"

echo ""
echo "[3/5] 提交代码..."
git commit -m "Initial commit: DeepSeek-OCR on Ascend NPU

- Complete NPU deployment with custom Conv2D operator
- 100% success rate in benchmark tests
- Comprehensive documentation and examples
- Performance: 31.9s/image (CANN 8.3.RC1)"
echo "✅ 完成"

echo ""
echo "[4/5] 设置主分支..."
git branch -M main
echo "✅ 完成"

echo ""
echo "======================================"
echo "  准备就绪！"
echo "======================================"
echo ""
echo "接下来的步骤："
echo ""
echo "1. 在 GitHub 创建仓库:"
echo "   https://github.com/new"
echo "   仓库名: deskocr-ascend"
echo ""
echo "2. 关联远程仓库（替换为你的用户名）："
echo "   git remote add origin https://github.com/你的用户名/deskocr-ascend.git"
echo ""
echo "3. 推送到 GitHub："
echo "   git push -u origin main"
echo ""
echo "======================================"
echo ""
