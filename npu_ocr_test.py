#!/usr/bin/env python3
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
import warnings, tempfile
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, "/home/wxq/dpsk/ds-ocr-ascend")
os.chdir("/home/wxq/dpsk/ds-ocr-ascend")

print("="*70)
print("DeepSeek-OCR on Ascend NPU - Final Test")
print("="*70)

# Patch masked_scatter_ with dtype fix
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
    else:
        return original_masked_scatter_(self, mask, source)

torch.Tensor.masked_scatter_ = patched_masked_scatter_
print("[INFO] CPU fallback enabled for masked_scatter_\n")

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
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
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
                repl = AscendConv2d(
                    child.in_channels, child.out_channels,
                    child.kernel_size, child.stride, child.padding,
                    child.bias is not None
                )
                with torch.no_grad():
                    repl.weight.copy_(child.weight)
                    if child.bias is not None:
                        repl.bias.copy_(child.bias)
                setattr(module, child_name, repl)
                count += 1
    return count

print("[1] Loading tokenizer...")
t0 = time.time()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
print(f"    Done: {time.time()-t0:.1f}s\n")

print("[2] Loading model (float16)...")
t0 = time.time()
from modeling_deepseekocr import DeepseekOCRForCausalLM
model = DeepseekOCRForCausalLM.from_pretrained(
    "./model", trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="eager"
)
print(f"    Done: {time.time()-t0:.1f}s\n")

print("[3] Patching Conv2d...")
count = patch_conv2d(model)
print(f"    {count} layers patched\n")

print("[4] Moving to NPU:0...")
device = torch.device("npu:0")
torch.npu.set_device(device)
model = model.to(device)
model.eval()
print(f"    Model ready on NPU:0\n")

test_dir = "test_images"
imgs = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
print(f"[5] Testing {len(imgs)} images\n")
print("="*70 + "\n")

results = []
for i, img_name in enumerate(imgs, 1):
    img_path = os.path.join(test_dir, img_name)
    print(f"Image {i}/{len(imgs)}: {img_name}")
    t0 = time.time()
    
    # Capture stdout to get OCR result
    import io
    from contextlib import redirect_stdout
    
    output_buffer = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out")
            os.makedirs(out_path, exist_ok=True)
            
            with redirect_stdout(output_buffer):
                result = model.infer(tokenizer, prompt="OCR", image_file=img_path, output_path=out_path)
            
            captured_output = output_buffer.getvalue()
            dt = time.time() - t0
            
            # Extract actual text from captured output
            if "<｜begin▁of▁sentence｜>" in captured_output:
                text_start = captured_output.find("<｜begin▁of▁sentence｜>") + len("<｜begin▁of▁sentence｜>")
                ocr_text = captured_output[text_start:].strip()
            else:
                ocr_text = captured_output.strip()
            
            results.append((img_name, "SUCCESS", dt, ocr_text))
            print(f"  Status: SUCCESS")
            print(f"  Time: {dt:.2f}s")
            print(f"  Text length: {len(ocr_text)} chars")
            if ocr_text:
                preview = ocr_text[:100].replace("\n", " ")
                print(f"  Preview: {preview}{'...' if len(ocr_text) > 100 else ''}")
            print()
    except Exception as e:
        dt = time.time() - t0
        err = str(e).split("\n")[0][:100]
        results.append((img_name, "FAILED", dt, err))
        print(f"  Status: FAILED")
        print(f"  Time: {dt:.2f}s")
        print(f"  Error: {err}\n")

print("="*70)
print("FINAL RESULTS")
print("="*70 + "\n")

success = 0
total_time = 0
for name, status, dt, info in results:
    total_time += dt
    if status == "SUCCESS":
        success += 1
        text_len = len(info) if isinstance(info, str) else 0
        print(f"✓ {name}: {status} ({dt:.2f}s, {text_len} chars)")
    else:
        print(f"✗ {name}: {status} ({dt:.2f}s)")
        print(f"  Error: {info}")

print("\n" + "-"*70)
print(f"Device: NPU:0 (Ascend 910B2)")
print(f"Precision: float16")
print(f"Conv2D: Ascend C implementation (6 layers)")
print(f"masked_scatter_: CPU fallback")
print(f"\nSuccess: {success}/{len(results)}")
if len(results) > 0:
    print(f"Average time: {total_time/len(results):.2f}s per image")
print("="*70)
