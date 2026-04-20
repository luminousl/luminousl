# LLM 量化流程图

> 基于 `tensorrt_edgellm/quantization/llm_quantization.py`

---

## 一、整体量化流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         quantize_and_save_llm()                             │
│                         (主入口函数)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: load_hf_model(model_dir, dtype, device)                          │
│  加载 HuggingFace 模型和 Tokenizer                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ is_quantized(model) │
                         │   检查是否已量化     │
                         └─────────────────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                   已量化                     未量化
                   (跳过)                     │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: quantize_llm()                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  2.1 get_llm_calib_dataloader()                                    │   │
│  │      - 加载校准数据集 (cnn_dailymail 或本地数据集)                 │   │
│  │      - Tokenize 处理                                               │   │
│  │      - 返回 DataLoader                                             │   │
│  │                                                                     │   │
│  │  2.2 get_llm_quant_config(quantization, lm_head_quantization)      │   │
│  │      - 获取基础量化配置 (FP8/INT4_AWQ/NVFP4)                       │   │
│  │      - 合并 lm_head 量化配置                                       │   │
│  │      - 禁用视觉模型量化                                            │   │
│  │                                                                     │   │
│  │  2.3 quantize_model(model, quant_config, data_loader)              │   │
│  │      - 执行模型量化                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 3: 保存量化模型                                                      │
│  - model.save_pretrained(output_dir)                                       │
│  - tokenizer.save_pretrained(output_dir)                                  │
│  - 保存 hf_quant_config.json                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、量化配置获取流程 (get_llm_quant_config)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    get_llm_quant_config()                                  │
│  输入: quantization, lm_head_quantization                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  quantization == ? │
                         └─────────────────────┘
                         │         │         │
              ┌──────────┴──┐   ┌───┴───┐  ┌┴────────────┐
              ▼             ▼         ▼  ▼             ▼
           "fp8"       "int4_awq"  "nvfp4"  其他(报错)
              │             │         │
              ▼             ▼         ▼
    mtq.FP8_DEFAULT_CFG  mtq.INT4_AWQ_CFG  mtq.NVFP4_DEFAULT_CFG
              │             │         │
              └─────────────┴─────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  lm_head_quantization  │
              │     是否指定?           │
              └────────────────────────┘
                    │          │
               指定(None)      指定
                    │          │
                    ▼          ▼
           移除原 lm_head    合并 lm_head 配置
           配置              ┌─────────────────┐
              │             │ "fp8" → FP8_LM_HEAD_CONFIG   │
              │             │ "nvfp4" → NVFP4_LM_HEAD_CONFIG │
              │             └─────────────────┘
              │                    │
              └────────┬───────────┘
                       ▼
              ┌────────────────────────┐
              │ DISABLE_VISUAL_CONFIG   │
              │ 禁用视觉模型量化         │
              └────────────────────────┘
                       │
                       ▼
              ┌────────────────────────┐
              │    返回 quant_cfg     │
              └────────────────────────┘
```

---

## 三、模型量化执行流程 (quantize_model)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         quantize_model()                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  calibrate_loop() - 校准循环                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  for batch in calib_dataloader:                                     │   │
│  │      model(batch)  # 前向传播，收集激活值统计                         │   │
│  │                                                                     │   │
│  │  目的:                                                              │   │
│  │  - 收集每层激活的 min/max 值                                         │   │
│  │  - 计算最佳量化 scale                                                │   │
│  │  - 调整权重和量化参数                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  mtq.quantize(model, quant_config, forward_loop=calibrate_loop)           │
│  - 调用 ModelOpt 进行量化                                                  │
│  - 应用量化配置到模型各层                                                  │
│  - 执行校准                                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  mtq.print_quant_summary(model)                                            │
│  - 打印量化摘要 (量化层数、精度等)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 四、量化配置详解

### 4.1 量化方法对比

| 量化方法 | 精度 | 权重量化 | 激活量化 | 校准方式 |
|----------|------|----------|----------|----------|
| FP8 | 8-bit | FP8 | FP8 | 动态 |
| INT4_AWQ | 4-bit | INT4 | FP16 | AWQ |
| NVFP4 | 4-bit | NVFP4 | 动态 | 动态 |

### 4.2 lm_head 特殊配置

```python
# FP8 lm_head 配置
FP8_LM_HEAD_CONFIG = {
    "*lm_head.input_quantizer": {"num_bits": (4, 3), "axis": None},
    "*lm_head.weight_quantizer": {"num_bits": (4, 3), "axis": None},
}

# NVFP4 lm_head 配置
NVFP4_LM_HEAD_CONFIG = {
    "*lm_head.input_quantizer": {
        "num_bits": (2, 1),
        "block_sizes": {"-1": 16, "type": "dynamic", "scale_bits": (4, 3)},
        "enable": True
    },
    ...
}
```

**为什么 lm_head 需要特殊配置？**
- lm_head 输出 logits 直接影响采样质量
- 高精度对数概率对生成质量至关重要
- 通常使用更高精度或特殊量化方案

---

## 五、关键函数调用链

```
quantize_and_save_llm()
    │
    ├── load_hf_model()
    │       │
    │       └── AutoModelForCausalLM.from_pretrained()
    │       └── AutoTokenizer.from_pretrained()
    │
    ├── quantize_llm()
    │       │
    │       ├── get_llm_calib_dataloader()
    │       │       │
    │       │       ├── load_dataset()
    │       │       └── tokenizer.batch_encode_plus()
    │       │
    │       ├── get_llm_quant_config()
    │       │       │
    │       │       ├── mtq.FP8_DEFAULT_CFG / INT4_AWQ_CFG / NVFP4_DEFAULT_CFG
    │       │       └── 合并 lm_head 配置
    │       │
    │       └── quantize_model()
    │               │
    │               ├── calibrate_loop()
    │               └── mtq.quantize()
    │
    ├── model.save_pretrained()
    ├── tokenizer.save_pretrained()
    └── 保存 hf_quant_config.json
```

---

## 六、支持的量化配置组合

| quantization | lm_head_quantization | 说明 |
|--------------|----------------------|------|
| fp8 | None | 仅 FP8 量化 |
| fp8 | fp8 | FP8 + FP8 lm_head |
| fp8 | nvfp4 | FP8 + NVFP4 lm_head |
| int4_awq | None | INT4 AWQ 量化 |
| int4_awq | fp8 | INT4 + FP8 lm_head |
| nvfp4 | None | NVFP4 量化 |
| nvfp4 | fp8 | NVFP4 + FP8 lm_head |
| nvfp4 | nvfp4 | NVFP4 + NVFP4 lm_head |

---

*文档生成时间：2026-03-04*
