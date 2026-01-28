#!/usr/bin/env python3
import os, sys, time, torch, torch.nn as nn, torch.nn.functional as F
import warnings, tempfile, io
from contextlib import redirect_stdout

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
sys.path.insert(0, "/home/wxq/dpsk/ds-ocr-ascend")
os.chdir("/home/wxq/dpsk/ds-ocr-ascend")

print("\033[1;36m" + "="*70 + "\033[0m")
print("\033[1;36mDeepSeek-OCR Interactive Mode (NPU)\033[0m")
print("\033[1;36m" + "="*70 + "\033[0m\n")

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

print("\033[1;33m[1/4] Loading tokenizer...\033[0m")
t0 = time.time()
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./model", trust_remote_code=True)
print(f"      ✓ Done in {time.time()-t0:.1f}s\n")

print("\033[1;33m[2/4] Loading model (float16)...\033[0m")
t0 = time.time()
from modeling_deepseekocr import DeepseekOCRForCausalLM
model = DeepseekOCRForCausalLM.from_pretrained(
    "./model", trust_remote_code=True,
    torch_dtype=torch.float16, attn_implementation="eager")
print(f"      ✓ Done in {time.time()-t0:.1f}s\n")

print("\033[1;33m[3/4] Patching Conv2d...\033[0m")
count = patch_conv2d(model)
print(f"      ✓ {count} layers patched\n")

print("\033[1;33m[4/4] Moving to NPU:0...\033[0m")
device = torch.device("npu:0")
torch.npu.set_device(device)
model = model.to(device)
model.eval()
print(f"      ✓ Model ready\n")

print("\033[1;32m" + "="*70 + "\033[0m")
print("\033[1;32mModel loaded successfully! Ready for OCR.\033[0m")
print("\033[1;32m" + "="*70 + "\033[0m\n")

print("\033[1;37mUsage:\033[0m")
print("  - Enter image path (relative to /home/wxq/dpsk/ds-ocr-ascend/)")
print("  - Type 'test' to process test_images/test_image_1.png")
print("  - Type 'all' to process all test images")
print("  - Press Ctrl+C or type 'quit' to exit\n")

while True:
    try:
        print("\033[1;34m" + "-"*70 + "\033[0m")
        user_input = input("\033[1;36mEnter image path (or command): \033[0m").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n\033[1;33mGoodbye!\033[0m")
            break
        
        if user_input.lower() == "test":
            img_path = "test_images/test_image_1.png"
        elif user_input.lower() == "all":
            test_imgs = sorted([f"test_images/{f}" for f in os.listdir("test_images") if f.endswith(".png")])
            print(f"\n\033[1;33mProcessing {len(test_imgs)} images...\033[0m\n")
            for idx, img in enumerate(test_imgs, 1):
                print(f"\033[1;35m[{idx}/{len(test_imgs)}] {os.path.basename(img)}\033[0m")
                t0 = time.time()
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        out_path = os.path.join(tmpdir, "out")
                        os.makedirs(out_path, exist_ok=True)
                        output_buffer = io.StringIO()
                        with redirect_stdout(output_buffer):
                            model.infer(tokenizer, prompt="OCR", image_file=img, output_path=out_path)
                        captured = output_buffer.getvalue()
                        if "<｜begin▁of▁sentence｜>" in captured:
                            text = captured[captured.find("<｜begin▁of▁sentence｜>")+len("<｜begin▁of▁sentence｜>"):].strip()
                        else:
                            text = captured.strip()
                        print(f"  ✓ {time.time()-t0:.1f}s - {len(text)} chars")
                        if text:
                            preview = text[:80].replace("\n", " ")
                            print(f"  {preview}{'...' if len(text) > 80 else ''}\n")
                except Exception as e:
                    print(f"  ✗ Error: {str(e)[:60]}\n")
            continue
        else:
            img_path = user_input
        
        if not os.path.exists(img_path):
            print(f"\033[1;31m✗ File not found: {img_path}\033[0m\n")
            continue
        
        print(f"\n\033[1;33mProcessing: {img_path}\033[0m")
        t0 = time.time()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out")
            os.makedirs(out_path, exist_ok=True)
            
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                model.infer(tokenizer, prompt="OCR", image_file=img_path, output_path=out_path)
            
            captured = output_buffer.getvalue()
            if "<｜begin▁of▁sentence｜>" in captured:
                ocr_text = captured[captured.find("<｜begin▁of▁sentence｜>")+len("<｜begin▁of▁sentence｜>"):].strip()
            else:
                ocr_text = captured.strip()
            
            dt = time.time() - t0
            print(f"\n\033[1;32m✓ Success in {dt:.2f}s\033[0m")
            print(f"\033[1;37mLength: {len(ocr_text)} characters\033[0m\n")
            print("\033[1;36m" + "-"*70 + "\033[0m")
            print("\033[1;37mOCR Result:\033[0m")
            print("\033[1;36m" + "-"*70 + "\033[0m")
            print(ocr_text if len(ocr_text) < 500 else ocr_text[:500] + "\n...")
            print("\033[1;36m" + "-"*70 + "\033[0m\n")
            
    except KeyboardInterrupt:
        print("\n\n\033[1;33mInterrupted. Goodbye!\033[0m")
        break
    except Exception as e:
        print(f"\n\033[1;31m✗ Error: {str(e)[:100]}\033[0m\n")
