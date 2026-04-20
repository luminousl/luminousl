# DriveLlm 学习计划

> 基于 zdrive_qnx/DriveLlm (TensorRT Edge-LLM) 源码学习

---

## 项目概述

**来源**：NVIDIA TensorRT Edge-LLM (改编版)
**用途**：车载 VLM (Vision-Language Model) 部署
**架构**：HuggingFace → Python Export → ONNX → Engine Builder → TensorRT Engine → C++ Runtime

---

## 学习目标

1. 理解 LLM/VLM 推理原理
2. 掌握模型量化与 ONNX 导出流程
3. 理解 C++ Runtime 核心机制
4. 能够进行二次开发

---

## 整体学习路径

```
第1周：环境熟悉 + 跑通流程
第2周：理解量化模块
第3周：理解 ONNX 导出
第4周：C++ Runtime 入门
第5周：深入核心模块 (选学)
```

---

## 第一周：环境熟悉 + 跑通流程

### 任务 1.1：环境准备

**具体内容：**
- 克隆仓库，配置环境变量
- 安装 Python 依赖
- 验证安装成功

**参考命令：**
```bash
cd /home/qianli/code/zdrive_qnx/DriveLlm
pip install -e .
python -c "import tensorrt_edgellm; print(tensorrt_edgellm.__version__)"
```

**交付物：** 能运行 `python -c "import tensorrt_edgellm"`

---

### 任务 1.2：模型量化

**具体内容：**
- 使用官方工具量化一个小模型
- 理解量化流程

**参考命令：**
```bash
tensorrt-edgellm-quantize-llm \
    --model_dir Qwen/Qwen2-0.5B-Instruct \
    --output_dir ./quantized/qwen_0.5b_fp8 \
    --quantization fp8
```

**交付物：** 获得 FP8 量化后的模型目录

---

### 任务 1.3：ONNX 导出

**具体内容：**
- 将量化模型导出为 ONNX

**参考命令：**
```bash
tensorrt-edgellm-export-llm \
    --model_dir ./quantized/qwen_0.5b_fp8 \
    --output_dir ./onnx/qwen_0.5b_onnx
```

**交付物：** 获得 ONNX 模型文件

---

## 第二周：理解量化模块

### 任务 2.1：阅读量化代码结构

**关键文件：**
- `tensorrt_edgellm/quantization/llm_quantization.py`

**学习重点：**
- 量化流程入口
- 配置管理
- ModelOpt 集成

**交付物：** 画出量化流程图

---

### 任务 2.2：理解量化配置差异

**学习内容：**
- FP8 量化配置
- INT4 AWQ 量化配置
- NVFP4 量化配置
- lm_head 特殊处理

**交付物：** 整理配置对比表

---

### 任务 2.3：修改配置实践

**具体内容：**
- 修改 lm_head 量化配置
- 重新量化模型
- 验证量化后模型可用

**思考题：**
- FP8 和 INT4 的区别是什么？
- 为什么 lm_head 需要单独配置？

---

## 第三周：理解 ONNX 导出

### 任务 3.1：阅读导出代码

**关键文件：**
- `tensorrt_edgellm/onnx_export/llm_export.py`
- `tensorrt_edgellm/onnx_export/visual_export.py`

**学习重点：**
- HuggingFace 模型结构
- ONNX 节点转换
- 权重导出

**交付物：** 整理导出流程

---

### 任务 3.2：理解模型结构转换

**学习内容：**
- Transformer 层转换
- Attention 算子处理
- 融合策略

**交付物：** 标注关键转换函数

---

### 任务 3.3：可视化 ONNX 模型

**具体内容：**
- 用 Netron 打开 ONNX 文件
- 分析算子结构
- 理解数据流

**交付物：** 截图 + 结构说明

---

## 第四周：C++ Runtime 入门

### 任务 4.1：编译 Runtime

**具体内容：**
- 配置编译环境
- 编译 C++ Runtime
- 产出可执行文件

**参考命令：**
```bash
mkdir build && cd build
cmake .. -DTRT_PACKAGE_DIR=/path/to/TensorRT
make -j$(nproc)
```

**交付物：** 获得 `llm_inference` 可执行文件

---

### 任务 4.2：运行 Demo

**具体内容：**
- 运行推理 demo
- 分析日志输出
- 理解推理流程

**交付物：** 整理推理流程日志

---

### 任务 4.3：阅读 Runtime 接口

**关键文件：**
- `cpp/runtime/llmInferenceRuntime.h`

**学习重点：**
- 类结构
- 核心方法
- 数据流

**交付物：** 整理类图和方法说明

---

## 第五周：深入核心模块 (选学)

### 任务 5.1：KV Cache 实现分析

**关键文件：**
- `cpp/runtime/linearKVCache.h`

**学习重点：**
- 内存布局
- 分配与回收
- 复用机制

**交付物：** 整理 linearKVCache 原理

---

### 任务 5.2：Flash Attention Kernel

**关键文件：**
- `cpp/kernels/contextAttentionKernels/`

**学习重点：**
- 核心计算流程
- Memory Access 优化

---

### 任务 5.3：VLM 多模态流程

**关键文件：**
- `cpp/multimodal/multimodalRunner.h`
- `zeekr_adapt/docs/vlm_itf.md`

**学习重点：**
- 图像预处理
- Token 拼接
- 端到端流程

**交付物：** 整理图像→Token→LLM 流程

---

## 核心技术点汇总

| 模块 | 技术点 | 难度 |
|------|--------|------|
| 量化 | FP8/INT4/NVFP4, AWQ/GPTQ | ⭐⭐⭐ |
| ONNX 导出 | 模型结构转换, 算子融合 | ⭐⭐⭐ |
| C++ Runtime | KV Cache, CUDA Graph | ⭐⭐⭐⭐ |
| CUDA Kernel | Flash Attention, RoPE | ⭐⭐⭐⭐⭐ |
| VLM | 多模态融合, 动态分辨率 | ⭐⭐⭐⭐ |

---

## 参考资料

- 官方文档：`developer_guide/`
- 源码：各模块 `*.py`, `*.h`, `*.cpp`
- NVIDIA TensorRT 官方文档

---

## 学习笔记

> 此处记录学习过程中的笔记和思考

### 

---

*创建时间：2026-03-04*
*最后更新：2026-03-04*
