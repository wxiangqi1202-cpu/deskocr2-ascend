# 模型文件下载

## 下载地址

模型权重文件 (6.3GB) 请从以下地址下载：

- **Hugging Face**: https://huggingface.co/deepseek-ai/deepseek-ocr
- **ModelScope**: https://modelscope.cn/models/deepseek-ai/deepseek-ocr

## 下载步骤

```bash
# 方式 1: 使用 huggingface-cli
pip install huggingface-hub
huggingface-cli download deepseek-ai/deepseek-ocr --local-dir ./model

# 方式 2: 使用 git lfs
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-ocr ./model
```

## 文件结构

下载完成后，model 目录应包含：

```
model/
├── model-00001-of-000001.safetensors  (6.3GB)
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## 验证下载

```bash
# 检查文件大小
ls -lh model/*.safetensors

# 应该显示约 6.3GB
```
