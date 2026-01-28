#!/bin/bash

echo "================================="
echo "  DeepSeek-OCR NPU 启动脚本"
echo "================================="
echo ""

# 激活Ascend环境
echo "[1/2] 激活 Ascend 环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "      ✓ 完成"
echo ""

# 启动OCR
echo "[2/2] 启动 OCR 模型 (NPU)..."
cd /home/wxq/dpsk
python3 ocr_interactive.py
