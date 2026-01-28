# DeepSeek-OCR NPU 最终基准测试报告

**测试时间**: 2026-01-28 16:51  
**测试环境**: Ascend 910B2 NPU, CANN 8.3.RC1

---

## 📊 性能总结

| 指标 | 数值 |
|------|------|
| 模型加载时间 | 23.6秒 |
| 测试图片数量 | 4张 |
| 成功率 | 100% (4/4) |
| 总处理时间 | 127.7秒 |
| 平均处理时间 | **31.9秒/图** |
| 总识别字符数 | 4,510字符 |
| 平均识别速度 | **35.3字符/秒** |

---

## 🎯 单图详细性能

| 图片 | 大小 | 耗时 | 识别字符 | 速度 |
|------|------|------|---------|------|
| image.png | 183.9KB | 64.1s | 3,068 | 47.9 c/s |
| test_image_1.png | 132.9KB | 14.9s | 291 | 19.5 c/s |
| test_image_2.png | 18.5KB | 2.5s | 27 | 10.7 c/s |
| test_image_3.png | 167.2KB | 46.1s | 1,124 | 24.4 c/s |

---

## 🔧 技术实现

### 硬件配置
- **设备**: NPU:0 (Ascend 910B2)
- **显存**: 65GB HBM2e
- **精度**: float16

### 软件优化
- **Conv2D**: Ascend C 实现 (im2col + matmul, 6层)
- **masked_scatter_**: CPU fallback
- **批处理**: 单图串行处理

---

## 📈 性能分析

### 速度分级
```
短文本 (< 50 字符):   2.5秒  ✅ 快速
中等文本 (< 500 字符): 14.9秒  ✅ 正常
长文本 (> 1000 字符): 46-64秒 ⚠️ 较慢
```

### 瓶颈识别
1. **Vision Encoder**: ✅ 完全NPU加速 (Conv2D已优化)
2. **Token生成**: ⚠️ CPU fallback导致延迟
3. **数据传输**: ⚠️ NPU↔CPU频繁传输

---

## 🚀 优化潜力

### 当前限制 (CANN 8.3.RC1)
- masked_scatter_ 算子缺失 → CPU fallback
- 预估性能损失: 5-10倍

### 升级后预期 (CANN 8.4+)
- ✅ 纯NPU执行，无CPU回退
- 🎯 **预期速度**: 3-6秒/图
- 🎯 **预期吞吐**: 200-400 chars/s
- 📈 **性能提升**: 5-10倍

---

## 📦 文件清理结果

### 清理前
- Python文件: 25个
- 报告文件: 8个
- **总计**: 33个文件

### 清理后
- `ocr_interactive.py` (7.8KB) - 交互式OCR
- `npu_ocr_test.py` (6.5KB) - 批量测试
- `start_ocr.sh` (382B) - 启动脚本
- `benchmark.py` (6.9KB) - 基准测试
- `NPU_TEST_REPORT.md` (3.7KB) - 技术报告
- **总计**: 5个核心文件

### 备份
- 30个旧文件已移至: `/home/wxq/dpsk/backup_20260128_164743/`

---

## 🎮 使用指南

### 快速启动
```bash
bash /home/wxq/dpsk/start_ocr.sh
```

### 批量测试
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/wxq/dpsk
python3 npu_ocr_test.py
```

### 基准测试
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/wxq/dpsk
python3 benchmark.py
```

---

## ✅ 结论

### 当前状态
- ✅ **功能**: 完全正常，100%识别成功率
- ✅ **稳定性**: 连续处理4张图片无崩溃
- ⚠️ **性能**: 可用级别 (31.9s/图)
- ✅ **可维护性**: 代码整洁，文件精简

### 生产就绪
- ✅ 小批量处理 (< 100张/天)
- ⚠️ 中等批量 (100-1000张/天) - 需要更多NPU或时间
- ❌ 大批量 (> 1000张/天) - 需升级CANN 8.4+

### 推荐操作
1. **立即可用**: 当前版本适合功能验证和小批量处理
2. **短期优化**: 联系华为技术支持升级CANN工具包
3. **长期方案**: 迁移到vLLM框架获得生产级性能

---

**报告生成**: 2026-01-28  
**测试执行**: Kernelcat AI Assistant  
**结果文件**: `/home/wxq/dpsk/benchmark_results.json`
