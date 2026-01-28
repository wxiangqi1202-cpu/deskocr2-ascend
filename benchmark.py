#!/usr/bin/env python3
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
import warnings, tempfile, io, json
from contextlib import redirect_stdout
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, "/home/wxq/dpsk/ds-ocr-ascend")
os.chdir("/home/wxq/dpsk/ds-ocr-ascend")

print("="*70)
print("DeepSeek-OCR NPU Benchmark Test")
print("="*70 + "\n")

# Patch masked_scatter_
original_masked_scatter_ = torch.Tensor.masked_scatter_
def patched_masked_scatter_(self, mask, source):
    if self.device.type == "npu":
        orig_dtype = self.dtype
        self_cpu = self.cpu()
        mask_cpu = mask.cpu() if mask.device.type == "npu" else mask
        source_cpu = source.cpu() if source.device.type == "npu" else source
        if source_cpu.dtype != self_cpu.dtype:
            source_cpu = source_cpu.to(self_cpu.dtype)
        result = self_cpu.masked_scatter_(mask_cpu, source_cpu)
        self.copy_(result.to(self.device, dtype=orig_dtype))
        return self
    return original_masked_scatter_(self, mask, source)
torch.Tensor.masked_scatter_ = patched_masked_scatter_

class AscendConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=0, bias=True):
        super().__init__()
        self.k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.s = stride if isinstance(stride, int) else stride[0]
        self.p = padding if isinstance(padding, int) else padding[0]
        self.weight = nn.Parameter(torch.Tensor(out_c, in_c, self.k, self.k))
        self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None: nn.init.zeros_(self.bias)
    def forward(self, x):
        B, C, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.p, stride=self.s)
        w = self.weight.view(self.weight.shape[0], -1)
        out = torch.einsum("oi,bil->bol", w, cols)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1)
        H_out = (H + 2*self.p - self.k) // self.s + 1
        W_out = (W + 2*self.p - self.k) // self.s + 1
        return out.view(B, -1, H_out, W_out)

def patch_conv2d(model):
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                repl = AscendConv2d(child.in_channels, child.out_channels,
                    child.kernel_size, child.stride, child.padding, child.bias is not None)
                with torch.no_grad():
                    repl.weight.copy_(child.weight)
                    if child.bias is not None: repl.bias.copy_(child.bias)
                setattr(module, child_name, repl)
                count += 1
    return count

print("[SETUP] Loading model...")
setup_start = time.time()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)

from modeling_deepseekocr import DeepseekOCRForCausalLM
model = DeepseekOCRForCausalLM.from_pretrained(
    "./model", trust_remote_code=True,
    torch_dtype=torch.float16, attn_implementation="eager")

conv_count = patch_conv2d(model)

device = torch.device("npu:0")
torch.npu.set_device(device)
model = model.to(device)
model.eval()

setup_time = time.time() - setup_start
print(f"[SETUP] Complete in {setup_time:.1f}s\n")
print("="*70)
print("BENCHMARK RESULTS")
print("="*70 + "\n")

test_dir = "test_images"
imgs = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])

results = []
total_chars = 0
total_time = 0

for i, img_name in enumerate(imgs, 1):
    img_path = os.path.join(test_dir, img_name)
    file_size = os.path.getsize(img_path) / 1024  # KB
    
    print(f"[{i}/{len(imgs)}] {img_name} ({file_size:.1f}KB)")
    
    inference_start = time.time()
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out")
            os.makedirs(out_path, exist_ok=True)
            
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                model.infer(tokenizer, prompt="OCR", image_file=img_path, output_path=out_path)
            
            captured = output_buffer.getvalue()
            if "<｜begin▁of▁sentence｜>" in captured:
                text = captured[captured.find("<｜begin▁of▁sentence｜>")+len("<｜begin▁of▁sentence｜>"):].strip()
            else:
                text = captured.strip()
            
            inference_time = time.time() - inference_start
            char_count = len(text)
            
            results.append({
                "image": img_name,
                "size_kb": file_size,
                "time_s": inference_time,
                "chars": char_count,
                "speed_chars_per_s": char_count / inference_time if inference_time > 0 else 0,
                "status": "SUCCESS"
            })
            
            total_chars += char_count
            total_time += inference_time
            
            print(f"  Time: {inference_time:.2f}s")
            print(f"  Output: {char_count} chars")
            print(f"  Speed: {char_count/inference_time:.1f} chars/s\n")
            
    except Exception as e:
        inference_time = time.time() - inference_start
        results.append({
            "image": img_name,
            "size_kb": file_size,
            "time_s": inference_time,
            "chars": 0,
            "speed_chars_per_s": 0,
            "status": "FAILED",
            "error": str(e)[:100]
        })
        print(f"  FAILED: {str(e)[:60]}\n")

print("="*70)
print("SUMMARY")
print("="*70 + "\n")

success = sum(1 for r in results if r["status"] == "SUCCESS")
avg_time = total_time / len(results) if results else 0
avg_speed = total_chars / total_time if total_time > 0 else 0

print(f"Setup Time: {setup_time:.2f}s")
print(f"Total Images: {len(results)}")
print(f"Successful: {success}/{len(results)} ({100*success/len(results):.0f}%)")
print(f"Total Processing Time: {total_time:.2f}s")
print(f"Average Time per Image: {avg_time:.2f}s")
print(f"Total Characters: {total_chars}")
print(f"Average Speed: {avg_speed:.1f} chars/s")
print()

print("Per-Image Details:")
for r in results:
    status_icon = "✓" if r["status"] == "SUCCESS" else "✗"
    print(f"  {status_icon} {r['image']:20s} {r['time_s']:6.2f}s  {r['chars']:5d} chars  {r['speed_chars_per_s']:6.1f} c/s")

print("\n" + "="*70)
print("HARDWARE INFO")
print("="*70 + "\n")
print(f"Device: NPU:0 (Ascend 910B2)")
print(f"Precision: float16")
print(f"Conv2D Implementation: Ascend C (im2col+matmul, {conv_count} layers)")
print(f"masked_scatter_: CPU fallback")
print("="*70)

# Save JSON report
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "setup_time_s": setup_time,
    "total_images": len(results),
    "successful": success,
    "total_time_s": total_time,
    "avg_time_s": avg_time,
    "total_chars": total_chars,
    "avg_speed_chars_per_s": avg_speed,
    "results": results
}

with open("/home/wxq/dpsk/benchmark_results.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\nBenchmark results saved to: /home/wxq/dpsk/benchmark_results.json")
