import sys
import os
import argparse
import yaml
from typing import Dict, List, Any, Generator, Optional, Tuple
import numpy as np
import onnx
import torch
import onnxruntime as ort
import pandas as pd
import onnx_graphsurgeon as gs
from tabulate import tabulate

try:
    import tensorrt as trt
    from polygraphy import constants
    from polygraphy.backend.onnx import (
        BytesFromOnnx,
        ModifyOutputs as ModifyOnnxOutputs,
        OnnxFromPath
    )
    from polygraphy.backend.onnxrt import OnnxrtRunner
    from polygraphy.backend.trt import (
        CreateConfig as CreateTRTConfig,
        EngineBytesFromNetwork,
        EngineFromBytes,
        ModifyNetworkOutputs,
        LoadPlugins,
        NetworkFromOnnxPath,
        TrtRunner,
        ShapeTuple,
        Profile
    )
    from polygraphy.comparator import Comparator
    POLYGRAPHY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorRT/Polygraphy not available: {e}")
    POLYGRAPHY_AVAILABLE = False
    trt = None
    constants = None

try:
    from onnxruntime_extensions import PyOp, onnx_op, PyOrtFunction, get_library_path as _get_library_path
except ImportError:
    PyOp = None
    onnx_op = None
    PyOrtFunction = None

try:
    from custom_op_library import *
except:
    pass

def process_fastmsda_for_onnx(onnx_graph):
    node_list = [node for node in onnx_graph.nodes if node.op == "FastMSDA"]
    for node in node_list:
        if node.attrs['static_msda'] == 1:
            spatial_shape = node.attrs['spatial_shape']
            if len(spatial_shape) == 2:
                spatial_shape_str = f"{spatial_shape[0]}_{spatial_shape[1]}"
                node.attrs['spatial_shape'] = spatial_shape_str
                node.op = "FastMSDA_STATIC"
                node.domain  = "ai.onnx.contrib"
            else:
                print("该FastMSDA类型不支持")

        elif node.attrs['static_msda'] == 0:
            spatial_shape = node.attrs['spatial_shape']
            if len(spatial_shape) == 18:
                node.op = "FastMSDA_BEVOD"
                node.domain  = "ai.onnx.contrib"
                assert len(node.inputs) == 5
            elif len(spatial_shape) == 26:
                node.op = "FastMSDA_BEVOD_v9"
                node.domain  = "ai.onnx.contrib"
                assert len(node.inputs) == 6
            elif len(spatial_shape) == 56:
                node.op = "FastMSDA_transformer2"
                node.domain  = "ai.onnx.contrib"
                assert len(node.inputs) == 4
            else:
                print("该FastMSDA类型不支持")
        else:
            print("该FastMSDA类型不支持")

    node_list = [node for node in onnx_graph.nodes if node.op == "MultiscaleDeformableAttnPlugin_TRT"]
    for node in node_list:
        node.domain  = "ai.onnx.contrib"

    node_list = [node for node in onnx_graph.nodes if node.op == "bev_pool_v2"]
    for node in node_list:
        node.domain  = "ai.onnx.contrib"    
    
    onnx_model = gs.export_onnx(onnx_graph)
    return onnx_model

def process_fastmsda_for_trt(trt_graph):
    trt_model = gs.export_onnx(trt_graph)
    inits = {init.name: init for init in trt_model.graph.initializer}
    for i in range(len(trt_model.graph.node)-1, -1, -1):
        node = trt_model.graph.node[i]
        if node.op_type == "MultiscaleDeformableAttnPlugin_TRT":
            a = node.input[1]  #.copy()
            b = node.input[2]   #.copy()
            node.op_type = "MultiscaleDeformableAttnPlugin_TRT_new"
            inits[a].CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(inits[a]).astype(np.int32), a))
            inits[b].CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(inits[b]).astype(np.int32), b))
    return trt_model

def unified_process(onnx_path, config):
    model = onnx.load(onnx_path)
    # 将模型输入输出的fp16-->fp32
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    for out in model.graph.output:
        if out.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    
    graph = gs.import_onnx(model)
    tensors = graph.tensors()
    # 去除confident_filter算子
    for out in graph.outputs[::-1]:
        if out.name.startswith("Confidence_Filter"):
            graph.outputs.remove(out)
            node = out.inputs[0]
            for inp in node.inputs:
                if isinstance(inp, gs.Variable):
                    graph.outputs.append(inp)
    # 添加配置的额外输入
    add_input_name_list = config['additional_operation'].get('add_input_name_list', [])
    if add_input_name_list:
        print(f"\nAdding {len(add_input_name_list)} additional inputs to TRT model:")
        for name in add_input_name_list:
            if name in tensors:
                inp_tensor = tensors[name]   #.to_variable(dtype=np.float32, shape=[3,256,64,120])
                inp_tensor.name = inp_tensor.name.replace('/', '_').replace(':', '_')
                shape = get_input_shape(config, name)
                inp_tensor.dtype = np.float32
                inp_tensor.shape = shape
                graph.inputs.append(inp_tensor)
                print(f"  ✓ Added input: {name}")
            else:
                print(f"  ✗ Tensor not found: {name}")
    graph.cleanup()
    free_inputs = [inp for inp in graph.inputs if len(inp.outputs)<1]
    for inp in free_inputs:
        graph.inputs.remove(inp)
    return graph

def get_input_shape(config, input_name):
    additional_inputs_data_root = config["additional_operation"]["add_input_data_root"]
    name_list = os.listdir(additional_inputs_data_root)
    input_name = input_name.replace('/', '_').replace(':', '_')
    for bin_name in name_list:
        if bin_name.startswith(input_name+'.tag_shape.'):
            bin_path = os.path.join(additional_inputs_data_root, bin_name)
            break
    dtype = bin_name.split('.')[-2]
    shape_list = bin_name.split('.tag_shape.')[1].split(f'.{dtype}')[0].split('.')
    shape = [int(sha) for sha in shape_list]
    return shape

def onnx_preprocess(config):
    onnx_path = config['onnx']['onnx_path']
    # trt/onnxruntime 统一预处理
    graph = unified_process(onnx_path, config)
    trt_graph = graph.copy()
    onnx_graph = graph.copy()

    # 预处理给trt推理
    output_path = onnx_path.replace(".onnx", ".trtinfer.onnx")
    trt_model = process_fastmsda_for_trt(trt_graph)
    trt_model.ir_version = 9
    onnx.save(trt_model, output_path)
    print(f"Model preprcoessing is done. The new model has been saved to {output_path}")
    config['onnx']['trtinfer_onnx_path'] = output_path

    # 预处理给 onnxruntime 
    onnx_model = process_fastmsda_for_onnx(onnx_graph)
    output_path = onnx_path.replace(".onnx", ".onnxinfer.onnx")
    print(f"Model preprcoessing is done. The new model has been saved to {output_path}")
    onnx_model.ir_version = 9
    onnx.save(onnx_model, output_path)
    config['onnx']['onnxinfer_onnx_path'] = output_path

    # 打印预处理总结
    print(f"\n{'='*60}")
    print(f"Preprocessing Summary:")
    print(f"{'='*60}")
    print(f"✓ Original model: {onnx_path}")
    print(f"✓ TRT inference model: {config['onnx']['trtinfer_onnx_path']}")
    print(f"✓ ONNX Runtime inference model: {config['onnx']['onnxinfer_onnx_path']}")
    print(f"✓ Additional inputs added: {len(config['additional_operation'].get('add_input_name_list', []))}")
    print(f"{'='*60}\n")
    
    return config

class ConfigManager:
    """配置文件管理类"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """加载和验证配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 验证必要配置项
        required_sections = ['onnx', 'trt', 'input_data_path']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config")
        
        # 验证ONNX配置
        if 'onnx_path' not in config['onnx']:
            raise ValueError("Missing 'onnx_path' in onnx section")
        
        if not os.path.exists(config['onnx']['onnx_path']):
            raise FileNotFoundError(f"ONNX model not found: {config['onnx']['onnx_path']}")
        
        # 设置默认值
        config['onnx'].setdefault('action', False)
        
        config['trt'].setdefault('action', False)
        config['trt'].setdefault('fp16', True)
        config['trt'].setdefault('fp32_layer_list', [])

        config['additional_operation'].setdefault('add_input_name_list', [])
        config['additional_operation'].setdefault('add_input_data_root', './save_bin_root/onnxruntime_outputs')
        config['additional_operation'].setdefault('save_name_list', [])
        config['additional_operation'].setdefault('save_bin_root', './save_bin_root')
        
        config.setdefault('outputs', [])
        config.setdefault('save_path', 'output_report.csv')
        config.setdefault('verbose', False)
        
        # 创建输出目录
        save_bin_root = config['additional_operation']['save_bin_root']
        os.makedirs(os.path.join(save_bin_root, 'onnxruntime_outputs'), exist_ok=True)
        os.makedirs(os.path.join(save_bin_root, 'trtinfer_outputs'), exist_ok=True)
        
        return config


class ModelRunnerFactory:
    """模型运行器工厂类"""
    
    @staticmethod
    def create_onnx_runner(config: Dict[str, Any], outputs: List[str]) -> ort.InferenceSession:
        """创建ONNX运行器"""
        onnx_path = config['onnx']['onnxinfer_onnx_path']
        
        # 配置ONNX Runtime会话
        sess_options = ort.SessionOptions()
        
        if _get_library_path is not None:
            try:
                sess_options.register_custom_ops_library(_get_library_path())
            except:
                pass
        
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        
        try:
            session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
            )
            return session
        except Exception as e:
            raise RuntimeError(f"Failed to create ONNX Runtime session: {e}")
    
    @staticmethod
    def create_trt_runner(config: Dict[str, Any], outputs: List[str]) -> Any:
        """创建TensorRT运行器"""
        onnx_path = config['onnx']['trtinfer_onnx_path']
        
        # 配置精度设置
        fp32_layer_dict = None
        if config['trt'].get('fp32_layer_list'):
            # breakpoint()
            fp32_layer_dict = {
                layer: trt.DataType.FLOAT 
                for layer in config['trt']['fp32_layer_list']
            }
        
        # 构建TensorRT引擎
        parse_network = NetworkFromOnnxPath(onnx_path)
        modified_network = ModifyNetworkOutputs(parse_network, outputs=outputs)
        
        # 如果存在精度设置，需要导入SetLayerPrecisions
        from polygraphy.backend.trt import SetLayerPrecisions
        if fp32_layer_dict:
            modified_network = SetLayerPrecisions(modified_network, layer_precisions=fp32_layer_dict)
        
        # 动态shape
        dynamic_shape_dict = config['trt'].get('dynamic_shape', {})
        for key in dynamic_shape_dict:
            shapes = dynamic_shape_dict[key]
            dynamic_shape_dict[key] = ShapeTuple((shapes[0],), (shapes[1],), (shapes[2],))
        profiles = Profile(dynamic_shape_dict)
        # breakpoint()
        create_trt_config = CreateTRTConfig(
            fp16=config['trt'].get('fp16', True),
            profiles=[profiles],
            tactic_sources=config['trt'].get('tactic_sources', [])
        )
        
        build_engine = EngineBytesFromNetwork(modified_network, config=create_trt_config)
        load_plugins = LoadPlugins(plugins=config['trt']['plugins'], obj=build_engine)
        deserialize_engine = EngineFromBytes(load_plugins)
        
        return deserialize_engine

class OutputSaver:
    """输出保存器"""
    
    def __init__(self, save_bin_root: str):
        self.save_bin_root = save_bin_root
        self.onnx_output_dir = os.path.join(save_bin_root, 'onnxruntime_outputs')
        self.trt_output_dir = os.path.join(save_bin_root, 'trtinfer_outputs')
        
        # 创建输出目录
        os.makedirs(self.onnx_output_dir, exist_ok=True)
        os.makedirs(self.trt_output_dir, exist_ok=True)
    
    def save_outputs(self, results: List[Tuple[str, List[Dict]]], save_name_list: List[str]) -> None:
        """
        保存指定张量的输出
        
        Args:
            results: 推理结果
            save_name_list: 要保存的张量名称列表
        """
        if not save_name_list:
            return
        
        print(f"\n{'='*60}")
        print(f"Saving outputs for {len(save_name_list)} tensors...")
        print(f"{'='*60}")
        
        # 解析结果
        flat_dict = {}
        for runner_name, runner_results in results:
            if not runner_results:
                continue
                
            single_result = runner_results[0]
            for tensor_name, tensor_value in single_result.items():
                if tensor_name not in flat_dict:
                    flat_dict[tensor_name] = {}
                flat_dict[tensor_name][runner_name] = tensor_value
        
        # 保存每个张量
        saved_count = 0
        for tensor_name in save_name_list:
            if tensor_name not in flat_dict:
                print(f"  ✗ Tensor not found in results: {tensor_name}")
                continue
            
            tensor_data = flat_dict[tensor_name]
            
            # 保存ONNX输出
            for runner_name, tensor_value in tensor_data.items():
                if 'onnx' in runner_name.lower() or 'onnx' in runner_name:
                    self._save_single_tensor(tensor_value, tensor_name, self.onnx_output_dir, 'ONNX')
                    saved_count += 1
                elif 'trt' in runner_name.lower() or 'tensorrt' in runner_name:
                    self._save_single_tensor(tensor_value, tensor_name, self.trt_output_dir, 'TRT')
                    saved_count += 1
        
        print(f"\n✓ Saved {saved_count} tensor outputs:")
        print(f"  - ONNX Runtime outputs: {self.onnx_output_dir}")
        print(f"  - TensorRT outputs: {self.trt_output_dir}")
        print(f"{'='*60}")
    
    def _save_single_tensor(self, tensor: np.ndarray, tensor_name: str, output_dir: str, runner_type: str) -> None:
        """保存单个张量到二进制文件"""
        # 清理张量名称中的非法字符
        safe_name = tensor_name.replace('/', '_').replace(':', '_')
        shape = tensor.shape
        dtype = tensor.dtype
        safe_name += f'.tag_shape'
        for sha in shape:
            safe_name += f'.{sha}'
        safe_name += f'.{dtype}'
        
        # 保存为二进制文件
        bin_path = os.path.join(output_dir, f"{safe_name}.bin")
        tensor.tofile(bin_path)
        
        # 保存形状信息
        # shape_path = os.path.join(output_dir, f"{safe_name}_shape.txt")
        # with open(shape_path, 'w') as f:
        #     f.write(f"original_name: {tensor_name}\n")
        #     f.write(f"saved_name: {safe_name}\n")
        #     f.write(f"runner_type: {runner_type}\n")
        #     f.write(f"shape: {tensor.shape}\n")
        #     f.write(f"dtype: {tensor.dtype}\n")
        #     f.write(f"min: {tensor.min():.6f}\n")
        #     f.write(f"max: {tensor.max():.6f}\n")
        #     f.write(f"mean: {tensor.mean():.6f}\n")
        #     f.write(f"std: {tensor.std():.6f}\n")
        
        print(f"  ✓ Saved {runner_type}: {tensor_name} -> {safe_name}.bin (shape: {tensor.shape})")


class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_input_data(config) -> Generator[Dict[str, np.ndarray], None, None]:
        """加载输入数据"""
        input_data_path = config["input_data_path"]
        additional_inputs = config["additional_operation"]["add_input_name_list"]
        additional_inputs_data_root = config["additional_operation"]["add_input_data_root"]

        if not os.path.exists(input_data_path):
            raise FileNotFoundError(f"Input data not found: {input_data_path}")
        
        try:
            inputs = torch.load(input_data_path, map_location="cpu")
            inputs = {key: inputs[key] for key in inputs}   #.numpy()
            name_list = os.listdir(additional_inputs_data_root)
            if additional_inputs:
                for input_name in additional_inputs:
                    input_name = input_name.replace('/', '_').replace(':', '_')
                    for bin_name in name_list:
                        if bin_name.startswith(input_name+'.tag_shape.'):
                            bin_path = os.path.join(additional_inputs_data_root, bin_name)
                            break
                    dtype = bin_name.split('.')[-2]
                    shape_list = bin_name.split('.tag_shape.')[1].split(f'.{dtype}')[0].split('.')
                    shape = np.array([int(sha) for sha in shape_list])
                    data = np.fromfile(bin_path, dtype=np.float32).reshape(shape)
                    inputs[input_name] = data
            yield {key: inputs[key].numpy() for key in inputs}
        except Exception as e:
            raise RuntimeError(f"Failed to load input data: {e}")
    
    @staticmethod
    def load_input_bin(config) -> Generator[Dict[str, np.ndarray], None, None]:
        """加载输入数据"""
        input_data_path = config["input_data_path"]
        onnx_path = config["onnx"]["onnx_path"]
        additional_inputs = config["additional_operation"]["add_input_name_list"]
        additional_inputs_data_root = config["additional_operation"]["add_input_data_root"]

        if not os.path.exists(input_data_path):
            raise FileNotFoundError(f"Input data not found: {input_data_path}")
        inputs = {}
        try:
            model = onnx.load(onnx_path)
            graph = gs.import_onnx(model)
            for out in graph.inputs:
                name = out.name
                shape = out.shape
                types = out.dtype
                all_int = all(isinstance(item, int) for item in shape)
                has_string = any(isinstance(item, str) for item in shape)
                if all_int:
                    if name == "SS_bev_static_6":
                        data = np.fromfile(os.path.join(input_data_path, name+"_chw.bin"), types).reshape(shape)
                    else:
                        data = np.fromfile(os.path.join(input_data_path, name+".bin"), types).reshape(shape)
                    # data = np.fromfile(os.path.join(input_data_path, name+".bin"), types).reshape(shape)
                elif has_string and len(shape) == 1:
                    data = np.fromfile(os.path.join(input_data_path, name+".bin"), types).reshape([-1])
                else:
                    raise RuntimeError(f"Failed to load input data: {name}")
                if types == np.float16:
                    data = data.astype(np.float32)
                inputs[name] = data
                print(f"✓ Loaded input: {name} (shape: {shape}, dtype: {types})")

            name_list = os.listdir(additional_inputs_data_root)
            if additional_inputs:
                for input_name in additional_inputs:
                    input_name = input_name.replace('/', '_').replace(':', '_')
                    for bin_name in name_list:
                        if bin_name.startswith(input_name+'.tag_shape.'):
                            bin_path = os.path.join(additional_inputs_data_root, bin_name)
                            break
                    dtype = bin_name.split('.')[-2]
                    shape_list = bin_name.split('.tag_shape.')[1].split(f'.{dtype}')[0].split('.')
                    shape = np.array([int(sha) for sha in shape_list])
                    data = np.fromfile(bin_path, dtype=np.float32).reshape(shape)  ## dtype的设置还没完成
                    inputs[input_name] = data

            yield {key: inputs[key] for key in inputs}
        except Exception as e:
            raise RuntimeError(f"Failed to load input data: {e}")



class ResultAnalyzer:
    """结果分析器"""
    
    @staticmethod
    def flatten_results(results: List[Tuple[str, List[Dict]]]) -> Dict[str, Dict[str, np.ndarray]]:
        """扁平化结果数据"""
        flat_dict = {}
        
        for runner_name, runner_results in results:
            if not runner_results:
                continue
                
            single_result = runner_results[0]
            for tensor_name, tensor_value in single_result.items():
                if tensor_name not in flat_dict:
                    flat_dict[tensor_name] = {}
                flat_dict[tensor_name][runner_name] = tensor_value
        
        return flat_dict
    
    @staticmethod
    def validate_tensor(tensor: np.ndarray, tensor_name: str) -> Optional[np.ndarray]:
        """验证并预处理张量"""
        if len(tensor.shape) == 1 and tensor.shape[0] == 0:
            print(f"Warning: Skipping empty tensor: {tensor_name}")
            return None
        
        # 处理标量张量
        if len(tensor.shape) == 0:
            tensor = np.array([tensor]).reshape(1)
        
        return tensor.astype(np.float32)
    
    @staticmethod
    def calculate_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
        """计算比较指标"""
        diff = np.abs(a - b)
        
        # 计算余弦相似度
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            cossim = 0.0
        else:
            cossim = (a * b).sum() / (norm_a * norm_b) * 100
        
        return {
            'cossim': cossim,
            'absdiff_max': diff.max(),
            'absdiff_sum': diff.sum(),
            'a_min': a.min(),
            'b_min': b.min(),
            'a_max': a.max(),
            'b_max': b.max(),
            'a_std': a.std(),
            'b_std': b.std()
        }


class ReportGenerator:
    """报告生成器"""
    
    @staticmethod
    def print_detailed_output(
        flat_dict: Dict[str, Dict[str, np.ndarray]],
        printed_tensors: set,
        runner_tag: List[str]
    ) -> None:
        """打印详细的张量输出"""
        for tensor_name in printed_tensors:
            if tensor_name not in flat_dict:
                print(f"Warning: Tensor '{tensor_name}' not found in results")
                continue
            
            values = flat_dict[tensor_name]
            print(f"\n{'='*20} {tensor_name} {'='*20}")
            
            for i, tag in enumerate(runner_tag):
                runner_name = list(values.keys())[i]
                tensor_value = values[runner_name]
                
                print(f"{tag} Prediction ({tensor_value.shape}):")
                print(tensor_value)
            print("=" * (len(tensor_name) + 40))
    
    @staticmethod
    def generate_comparison_report(
        flat_dict: Dict[str, Dict[str, np.ndarray]],
        model_outputs: List[str],
        runner_names: List[str],
        save_path: str = "report.csv"
    ) -> pd.DataFrame:
        """生成比较报告"""
        report_data = []
        
        for tensor_name in sorted(flat_dict.keys()):
            values = flat_dict[tensor_name]
            
            # 检查是否所有运行器都有该张量
            if not all(name in values for name in runner_names):
                print(f"Warning: Skipping tensor '{tensor_name}' - missing from some runners")
                continue
            
            a = ResultAnalyzer.validate_tensor(values[runner_names[0]], tensor_name)
            b = ResultAnalyzer.validate_tensor(values[runner_names[1]], tensor_name)
            
            if a is None or b is None:
                continue
            if not (a.size > 0 and b.size > 0):
                continue

            if a.shape == b.shape:
                metrics = ResultAnalyzer.calculate_metrics(a, b)
            
                report_data.append([
                    f"{metrics['cossim']:.2f} %",
                    f"{metrics['absdiff_max']:.6f}",
                    f"{metrics['absdiff_sum']:.6f}",
                    f"{metrics['a_min']:.6f}",
                    f"{metrics['b_min']:.6f}",
                    f"{metrics['a_max']:.6f}",
                    f"{metrics['b_max']:.6f}",
                    f"{metrics['a_std']:.6f}",
                    f"{metrics['b_std']:.6f}",
                    "x".join(map(str, a.shape)),
                    tensor_name,
                    tensor_name in model_outputs
                ])
            else:
                print(f"{tensor_name} shapes of trt and onnx are different ")
        
        headers = [
            "cossim", "absdiff max", "absdiff sum",
            "ort.min", "trt.min", "ort.max", "trt.max",
            "ort.std", "trt.std", "tensor shape",
            "tensor name", "is model output"
        ]
        
        # 打印表格
        print(tabulate(report_data, headers, "grid"))
        
        # 保存到CSV
        df = pd.DataFrame(report_data, columns=headers)
        df.to_csv(save_path, index=False)
        print(f"\nReport saved to: {save_path}")
        
        return df
    
    @staticmethod
    def generate_single_runner_report(
        flat_dict: Dict[str, Dict[str, np.ndarray]],
        model_outputs: List[str],
        runner_name: str,
        runner_type: str,
        save_path: str = "single_report.csv"
    ) -> pd.DataFrame:
        """生成单运行器报告"""
        report_data = []
        
        for tensor_name in sorted(flat_dict.keys()):
            if runner_name not in flat_dict[tensor_name]:
                continue
            
            tensor_value = ResultAnalyzer.validate_tensor(
                flat_dict[tensor_name][runner_name],
                tensor_name
            )
            
            if tensor_value is None:
                continue

            if not tensor_value.size > 0:
                continue

            report_data.append([
                f"{tensor_value.min():.6f}",
                f"{tensor_value.max():.6f}",
                f"{tensor_value.std():.6f}",
                "x".join(map(str, tensor_value.shape)),
                tensor_name,
                tensor_name in model_outputs
            ])
        
        headers = [
            f"{runner_type}.min",
            f"{runner_type}.max",
            f"{runner_type}.std",
            "tensor shape",
            "tensor name",
            "is model output"
        ]
        
        print(tabulate(report_data, headers, "grid"))
        
        df = pd.DataFrame(report_data, columns=headers)
        df.to_csv(save_path, index=False)
        print(f"\nReport saved to: {save_path}")
        
        return df


class ModelExecutor:
    """模型执行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.outputs = []
        self.printed_tensors = set()
        self.model_outputs = []
        self.output_saver = OutputSaver(config['additional_operation']['save_bin_root'])
        self.save_outputs = []
    
    def validate_outputs_in_models(self, onnx_path) -> None:
        if not self.outputs:
            return
        # onnx_path = self.config['onnx']['trtinfer_onnx_path']
        model = onnx.load(onnx_path)
        all_tensors = self._get_all_tensor_names(model)
        quant_tensor_outputs = self._get_quantization_tensor_outputs(model)
        non_quant_tensors = [t for t in all_tensors if t not in quant_tensor_outputs]
        input_names = [inp.name for inp in model.graph.input]
        non_quant_tensors = [t for t in non_quant_tensors if t not in input_names]
        initializer_names = self._get_initializer_names(model)
        non_quant_tensors = [t for t in non_quant_tensors if t not in initializer_names]
        # 验证输出是否在推理模型中
        error_outputs = []
        for out in self.outputs:
            if out in non_quant_tensors:
                pass
            else:
                error_outputs.append(out)
        
        if not error_outputs:
            return 
        else:
            raise ValueError(f"输出张量验证失败:\n{error_outputs}")

    def prepare_save_outputs(self) -> None:
        save_name_list = self.config['additional_operation'].get('save_name_list', [])
        save_outputs = save_name_list
        module_outputs = []
        tensor_outputs = []
        all_tag = False
        module_tag = False
        
        for i in range(len(save_outputs) - 1, -1, -1):
            output_name = save_outputs[i]
            
            if output_name.startswith("@") and output_name[1:] == "all":
                all_tag = True
                break
            elif output_name.startswith("@") and output_name[1:] == "model_outputs":
                tensor_outputs.extend(self.model_outputs)
            elif output_name.startswith("@"):
                module_outputs.append(output_name[1:])
                module_tag = True
            else:
                tensor_outputs.append(output_name)
        
        # 特殊处理：获取所有非量化节点的输出
        if module_tag or all_tag:
            if all_tag:
                self.save_outputs = self.outputs
            elif module_tag:
                self.save_outputs = []
                for tensor in self.outputs:
                    for module_name in module_outputs:
                        if tensor.startswith(module_name):
                            self.save_outputs.append(tensor)
                            break

                self.save_outputs.extend(tensor_outputs)
        else:
            self.save_outputs = tensor_outputs

    def prepare_outputs(self) -> None:
        """准备输出张量配置"""
        onnx_path = self.config['onnx']['trtinfer_onnx_path']
        # breakpoint()
        # 加载模型获取原始输出
        model = onnx.load(onnx_path)
        self.model_outputs = [out.name for out in model.graph.output]
        
        # 处理配置的输出
        outputs = self.config.get('outputs', []) # 处理输出
        save_outputs = self.config["additional_operation"].get('save_name_list', []) # 处理需要保存的输出
        for save_out in save_outputs:
            if save_out not in outputs:
                outputs.append(save_out)
        module_outputs = []
        tensor_outputs = []
        all_tag = False
        module_tag = False
        
        for i in range(len(outputs) - 1, -1, -1):
            output_name = outputs[i]
            
            if output_name.startswith("@") and output_name[1:] == "all":
                all_tag = True
                break
            elif output_name.startswith("@") and output_name[1:] == "model_outputs":
                tensor_outputs.extend(self.model_outputs)
            elif output_name.startswith("@"):
                module_outputs.append(output_name[1:])
                module_tag = True
            else:
                tensor_outputs.append(output_name)

        # 特殊处理：获取所有非量化节点的输出
        if module_tag or all_tag:
            # 获取所有张量名称
            all_tensors = self._get_all_tensor_names(model)
            
            # 获取量化节点的输出张量
            quant_tensor_outputs = self._get_quantization_tensor_outputs(model)
            
            # 排除量化节点的输出
            non_quant_tensors = [t for t in all_tensors if t not in quant_tensor_outputs]
            
            # 排除输入节点
            input_names = [inp.name for inp in model.graph.input]
            non_quant_tensors = [t for t in non_quant_tensors if t not in input_names]
            
            # 排除常量节点
            initializer_names = self._get_initializer_names(model)
            non_quant_tensors = [t for t in non_quant_tensors if t not in initializer_names]
            
            if all_tag:
                self.outputs = non_quant_tensors
            elif module_tag:
                self.outputs = []
                for tensor in non_quant_tensors:
                    for module_name in module_outputs:
                        if tensor.startswith(module_name):
                            self.outputs.append(tensor)
                            break

                self.outputs.extend(tensor_outputs)

            # 打印信息
            # print(f"Total tensors found: {len(all_tensors)}")
            # print(f"Quantization tensors excluded ({len(quant_tensor_outputs)}):")
            # for tensor in sorted(quant_tensor_outputs):
            #     print(f"  - {tensor}")
            print(f"Final outputs to monitor: {len(self.outputs)}")
        elif not outputs:
            if len(tensor_outputs) == 0:
                self.outputs = self.model_outputs
            else:
                self.outputs = self.model_outputs + tensor_outputs
        else:
            self.outputs = tensor_outputs

    def _get_all_tensor_names(self, model: onnx.ModelProto) -> List[str]:
        """获取模型中所有张量的名称"""
        all_tensors = set()
        
        # 遍历所有节点，收集输出张量
        for node in model.graph.node:
            for output_name in node.output:
                if output_name:  # 排除空字符串
                    all_tensors.add(output_name)
        
        # 添加图输入和输出
        for inp in model.graph.input:
            if inp.name:
                all_tensors.add(inp.name)
        
        for out in model.graph.output:
            if out.name:
                all_tensors.add(out.name)
        
        return list(all_tensors)

    def _get_quantization_tensor_outputs(self, model: onnx.ModelProto) -> List[str]:
        """获取量化算子（QuantizeLinear/DequantizeLinear）的输出张量"""
        quant_tensors = set()
        
        # 量化相关的算子类型
        QUANT_OPS = ['QuantizeLinear', 'DequantizeLinear']
        
        for node in model.graph.node:
            if node.op_type in QUANT_OPS:
                # 添加量化节点的所有输出
                for output_name in node.output:
                    if output_name:
                        quant_tensors.add(output_name)
                # 注意：这里也可以选择性地排除量化节点的输入
                # 但根据需求，我们只排除输出
        
        return list(quant_tensors)

    def _get_initializer_names(self, model: onnx.ModelProto) -> List[str]:
        """获取所有初始值（常量）的名称"""
        return [init.name for init in model.graph.initializer if init.name]
    
    def create_runners(self) -> Tuple[List, List[str]]:
        """创建运行器"""
        runners = []
        runner_tag = []
        if self.config['onnx'].get('action', False):
            onnx_session = ModelRunnerFactory.create_onnx_runner(self.config, self.outputs)
            runners.append(OnnxrtRunner(onnx_session))
            runner_tag.append('ONNX Runtime')
        
        if self.config['trt'].get('action', False):
            trt_engine = ModelRunnerFactory.create_trt_runner(self.config, self.outputs)
            runners.append(TrtRunner(trt_engine))
            runner_tag.append('TensorRT')
        
        if not runners:
            raise ValueError("No runners configured. Check onnx.action and trt.action in config")
        
        return runners, runner_tag
    
    def execute(self) -> None:
        """执行模型推理并生成报告"""
        # 准备输出配置
        self.prepare_outputs()
        self.validate_outputs_in_models(self.config['onnx']['trtinfer_onnx_path'])
        self.validate_outputs_in_models(self.config['onnx']['onnxinfer_onnx_path'])
        
        # 创建运行器
        runners, runner_tag = self.create_runners()
        
        # 加载输入数据
        if os.path.isfile(self.config['input_data_path']):
            data_loader = DataLoader.load_input_data(self.config)
        else:
            data_loader = DataLoader.load_input_bin(self.config)
        
        # 运行推理
        try:
            results = Comparator.run(runners, data_loader=data_loader)
        except Exception as e:
            raise RuntimeError(f"Model execution failed: {e}")
        
        # 分析结果
        flat_dict = ResultAnalyzer.flatten_results(results)

        # 保存指定张量的输出
        self.prepare_save_outputs()
        save_name_list = self.save_outputs
        if save_name_list:
            self.output_saver.save_outputs(results, save_name_list)
        
        # 打印详细输出（如果配置了要打印的张量）
        if self.printed_tensors:
            ReportGenerator.print_detailed_output(flat_dict, self.printed_tensors, runner_tag)
        
        # 生成报告
        if len(runner_tag) == 2:
            ReportGenerator.generate_comparison_report(
                flat_dict=flat_dict,
                model_outputs=self.model_outputs,
                runner_names=[runner.name for runner in runners],
                save_path=self.config.get('save_path', 'comparison_report.csv')
            )
        else:
            runner_type = runner_tag[0].split()[0].lower()
            ReportGenerator.generate_single_runner_report(
                flat_dict=flat_dict,
                model_outputs=self.model_outputs,
                runner_name=runners[0].name,
                runner_type=runner_type,
                save_path=self.config.get('save_path', 'single_runner_report.csv')
            )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Compare ONNX and TensorRT model inference results"
    )
    parser.add_argument(
        "model_yaml",
        type=str,
        help="Path to the model configuration YAML file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = ConfigManager.load_config(args.model_yaml)
    # 预处理onnx
    config = onnx_preprocess(config)

    # 执行模型推理
    executor = ModelExecutor(config)
    executor.execute()
        


if __name__ == "__main__":
    main()