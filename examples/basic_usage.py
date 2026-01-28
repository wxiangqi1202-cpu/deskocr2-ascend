#!/usr/bin/env python3
"""基础使用示例"""

import os
import sys
import torch

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../model"))

from transformers import AutoTokenizer
from modeling_deepseekocr import DeepseekOCRForCausalLM
import tempfile

def main():
    print("加载模型...")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "../model", 
        trust_remote_code=True
    )
    
    # 加载模型
    model = DeepseekOCRForCausalLM.from_pretrained(
        "../model",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    
    # 移至 NPU
    device = torch.device("npu:0")
    model = model.to(device)
    model.eval()
    
    print("模型加载完成！")
    
    # 处理图片
    image_path = "test_image.png"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "out")
        os.makedirs(output_path, exist_ok=True)
        
        result = model.infer(
            tokenizer,
            prompt="OCR",
            image_file=image_path,
            output_path=output_path
        )
        
        print("识别结果:")
        print(result)

if __name__ == "__main__":
    main()
