# Luminousl

Model deployment toolkit for Zdrive platform, providing complete tools for ONNX model processing, TensorRT engine building, model quantization, and performance analysis.

## Overview

```
luminousl/
├── onnx_utils/              # ONNX model processing tools
├── quantization_utils/      # Model quantization tools
├── tensorrt_utils/          # TensorRT build tools
├── matool/                  # TensorRT model analysis
├── perfview/                # Performance analysis
└── polygraph_tools/         # Accuracy comparison
```

## Installation

```bash
# Development installation
cd luminousl
pip install -e .

# With all dependencies
pip install -e ".[dev,tensorrt,polygraphy,ort]"
```

## Quick Start

### 1. ONNX Utils

```bash
# Health check
luminousl onnx health-check model.onnx
luminousl onnx health-check model.onnx --summary
luminousl onnx health-check model.onnx --filter "Accuracy.*" "Performance.*"

# Convert to FP16 strongly
luminousl onnx to-strongly input.onnx output.onnx

# Topological sort
python -m luminousl.onnx_utils.topological_sort model.onnx
```

### 2. Quantization Utils

```bash
# Optimize QDQ scales
luminousl quant optimize-qdq model.onnx -o optimized.onnx
```

### 3. TensorRT Utils

```bash
# Build TensorRT engine
luminousl trt build model.onnx -o model.trt --fp16
luminousl trt build model.onnx -o model.int8.trt --int8 --fp16 \
    --calibration-data ./calibration_images/

# Run inference
luminousl trt exec model.trt --input input.bin --output output.bin
```

### 4. MATool

```bash
# Process ONNX model
python -m luminousl.matool.utils.process_onnx model.onnx best

# Process TRT engine
python -m luminousl.matool.utils.process_trt model.trt best

# Compare models
python -m luminousl.matool.utils.compare_trt -b before_profile/ -a after_profile/
```

### 5. Perfview

```bash
# Create ONNX view
python -m luminousl.perfview.create_view onnx model.onnx --layers=layers.json

# Create TRT view
python -m luminousl.perfview.create_view trex layers_file --profile=profile.json
```

### 6. Polygraph Tools

```bash
# Compare accuracy
python -m luminousl.polygraph_tools.compare.py config.yaml
```

## Module Details

### onnx_utils

| Tool | Description |
|------|-------------|
| `health_check.py` | ONNX model health checking |
| `onnx2strongly.py` | Convert to FP16 precision |
| `topological_sort.py` | Node topological sorting |

### quantization_utils

| Tool | Description |
|------|-------------|
| `optimize_qdq_scales.py` | Optimize QDQ scale values |
| `qdq_translator.py` | Translate QDQ nodes |

### tensorrt_utils

| Tool | Description |
|------|-------------|
| `onnx_to_trt.py` | ONNX to TensorRT conversion |
| `trt_execution.py` | TRT inference execution |
| `show_trtexec_cmd.py` | Generate trtexec commands |

## Common Workflows

### Full Model Deployment

```bash
# 1. Model health check
luminousl onnx health-check model.onnx

# 2. Convert to FP16 strongly
luminousl onnx to-strongly model.onnx model_fp16.onnx

# 3. Optimize quantization scales (if needed)
luminousl quant optimize-qdq model_fp16.onnx -o model_opt.onnx

# 4. Build TensorRT engine
luminousl trt build model_opt.onnx -o model.trt --fp16

# 5. Accuracy verification
python -m luminousl.polygraph_tools.compare.py config.yaml
```

### INT8 Quantization

```bash
# 1. Generate fake calibration data
python -m luminousl.tensorrt_utils.onnx2trt_fakedata \
    --onnx model.onnx --int8 --calibration-data model.calib \
    -g fake_data

# 2. Build INT8 engine
luminousl trt build model.onnx -o model.int8.trt --int8 --fp16 \
    --calibration-data ./fake_data/
```

## CLI Reference

```
luminousl onnx health-check <model> [options]
luminousl onnx to-strongly <input> [output]
luminousl onnx topological-sort <model>
luminousl quant optimize-qdq <model> [-o output]
luminousl trt build <model> -o <engine> [options]
luminousl trt exec <engine> --input <input> --output <output>
luminousl matool process <target> [best|onnx|trt]
luminousl perfview create <type> <model>
luminousl polygraph compare <config>
```

## License

MIT
