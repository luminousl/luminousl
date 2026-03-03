# 工具脚本
  * [profile_analyzer.py]
  * [process_onnx.py]
  * [process_trt.py]
  * [process_engine.py]
  * [draw_engine.py]
  * [draw_onnx.py]
  * [parse_trtexec_log.py]

<br>

## profile_analyzer.py

功能： 对模型profile文件进行分析，给出分析报告。

## process_onnx.py

功能： 将onnx模型转为trt模型，并生成profile文件

## process_trt.py

功能： 处理trt模型，生成profile文件

## process_engine.py

功能:
1. Build a TensorRT engine from an ONNX file.
2. Profile an engine plan file.
3. Generate JSON files for exploration with trex.
4. Draw an SVG graph from an engine.

```
usage: process_engine.py [-h] [--print_only] [--build_engine] [--profile_engine] [--draw_engine] input outdir [trtexec [trtexec ...]]

Utility to build and profile TensorRT engines

positional arguments:
  input                 input file (ONNX or engine)
  outdir                directory to store output artifacts
  trtexec               trtexec commands not including the preceding -- (e.g. int8 shapes=input_ids:32x512,attention_mask:32x512

optional arguments:
  -h, --help            show this help message and exit
  --print_only          print the command-line and exit
  --build_engine, -b    build the engine
  --profile_engine, -p  engine the engine
  --draw_engine, -d     draw the engine
```

The script can run the entire ONNX to JSON files pipeline, or it can execute a single sub-command. For example, the following command line builds and profiles an engine from the ONNX model stored in a file named `my_onnx.onnx`:
```
$ process_engine.py my_onnx.onnx outputs_dir int8
```

This will generate the following files in directory `outputs_dir`:
* `my_onnx.onnx.engine` - the built engine file.
* `my_onnx.onnx.engine.build.log` - trtexec engine building log.
* `my_onnx.onnx.engine.build.metadata.json` - JSON of metadata parsed from the build log.
* `my_onnx.onnx.engine.graph.json` - JSON of engine graph.
* `my_onnx.onnx.engine.graph.json.svg` - SVG diagram of engine graph.
* `my_onnx.onnx.engine.profile.json` - JSON of engine layers profiling.
* `my_onnx.onnx.engine.profile.log` - trtexec engine profiling log.
* `my_onnx.onnx.engine.profile.metadata.json` - JSON of metadata parsed from the profiling log.
* `my_onnx.onnx.engine.timing.json` - JSON of engine profiling iteration timing.


Requirements:
* Path to trtexec binary is in $PATH.
* trex is installed (for graph drawing).
* Graphviz is installed (for graph drawing).
```
$ sudo apt-get --yes install graphviz
```

## draw_engine.py

功能：根据trt模型graph.json，输出trt模型结构（SVG）。

## draw_onnx.py

功能：输入onnx模型，输出onnx模型结构（SVG）。


## parse_trtexec_log.py

功能：根据`trtexec` log files，生成`metadata JSON files`。