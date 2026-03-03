# 模型分析工具 MATool

## 工具说明

该模型分析工具基于Tensorrt工具包trt-engine-explorer，用于对onnx模型以及trt engine进行模型结构、参数及性能分析。

## 工具环境配置

  0. 第一次使用该工具，若未进行过安装，则在工具包目录`/your_path/ModelAnalysisTool/`下进行环境安装及配置:
     ```
     source install.sh
     ```
     注： 该环境为虚拟python环境，每次使用使用命令： `source /env_MATool/bin/activate` 激活环境。
     推荐使用NV的镜像 docker pull nvcr.io/nvidia/pytorch:23.10-py3
## 工具使用说明

### 步骤1、 模型profile文件准备
注：该步骤使用`./utils/process_onnx.py`工具对模型文件进行profile文件生成，进行该步骤不需要进行环境配置。
在orin设备上进行profile生成时，只需把utils文件夹拷贝到orin设备上即可。

  1. 如果输入为onnx模型文件:
     使用如下命令对该模型文件进行build和profile操作：
     ```
     python3 ./utils/process_onnx.py A.onnx best
     ```

  2. 如果输入trt模型文件：
     ```
     python3 ./utils/process_trt.py A.trt best
     ```

### 步骤2、 对profile文件进行处理及解析
注：该步骤使用`./notebook/mago.py`工具对profile文件进行处理及解析

  3. 对`./your_model_directory/profiles`文件夹下的profile文件进行解析：
     ```
     python3 ./utils/mago.py -m /your_model_directory/profiles
     ```

### 步骤3、 查看模型分析报告

  4. 使用浏览器打开模型分析报告
  `./your_model_directory/profiles/report/MATool.html`

### 其他功能说明 

#### 模型差异对比

使用./utils/compare_trt.py文件可以生成两个模型的差异对比报告（需提前生成模型分析报告）

命令使用：
```bash
usage: compare_trt.py [-h] -b BEFORE -a AFTER -s SAVE

对比优化前后的trt差异

optional arguments:
  -h, --help            show this help message and exit
  -b BEFORE, --before BEFORE
                        优化前模型profile路径
  -a AFTER, --after AFTER
                        优化后模型profile路径
  -s SAVE, --save SAVE  报告保存路径
```
例子：
```bash
python3 ./utils/compare_trt.py -b /workspace/model2profile/profile/env_cognition/ -a /workspace/model2profile/profile/env -s ./utils/compare_test/
```