import onnx
import numpy as np
import os
import argparse
import fnmatch
import enum
import json

class SeverityLevel(enum.Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class HealthChecker:
    ISSUE_FUNCTION_FLOAT_TENSOR_UNKNOWN_VALUE = "Function.FloatTensor.UnknownValue"
    ISSUE_FUNCTION_QDQ_UNKNOWN_VALUE = "Function.QDQ.UnknownValue"
    ISSUE_ACCURACY_FLOATTENSOR_OVERFLOW = "Accuracy.FloatTensor.Overflow"
    ISSUE_ACCURACY_QDQ_SCALE_OVERFLOW = "Accuracy.QDQ.ScaleOverflow"
    ISSUE_ACCURACY_QDQ_SCALE_UNDERFLOW = "Accuracy.QDQ.ScaleUnderflow"
    ISSUE_ACCURACY_QDQ_SCALE_ZERO = "Accuracy.QDQ.ScaleZero"
    ISSUE_ACCURACY_UNFUSED_BN = "Accuracy.UnfusedBN"
    ISSUE_PERFORMANCE_PREBN_CANNOT_BE_FUSED = "Performance.PreBN.CannotBeFused"
    ISSUE_PERFORMANCE_DANGLING_BN = "Performance.DanglingBN"
    ISSUE_PERFORMANCE_USE_GROUP_CONV = "Performance.UseGroupConv"
    ISSUE_ACCURACY_BN_RUNNING_VARS_OVERFLOW = "Accuracy.BN.RunningVarsOverflow"
    ISSUE_PERFORMANCE_QDQ_SCALE_MISMATCH = "Performance.QDQ.ScaleMismatch"
    ISSUE_PERFORMANCE_QDQ_GRIDSAMPLE_INPUT_SHOULD_BE_HWC8 = "Performance.QDQ.GridSampleInputShouldBeHWC8"
    ISSUE_PERFORMANCE_QDQ_GRIDSAMPLE_INPUT_SHOULD_BE_CONV = "Performance.QDQ.GridSampleInputShouldBeConv"
    ISSUE_PERFORMANCE_QDQ_DANGLING_QDQ = "Performance.QDQ.DanglingQDQ"
    ISSUE_PERFORMANCE_REFORMAT_COPY = "Performance.ReformatCopy"
    ISSUE_PERFORMANCE_REFORMAT = "Performance.Reformat"
    all_issue_types = [
        ISSUE_FUNCTION_FLOAT_TENSOR_UNKNOWN_VALUE,
        ISSUE_FUNCTION_QDQ_UNKNOWN_VALUE,
        ISSUE_ACCURACY_FLOATTENSOR_OVERFLOW,
        ISSUE_ACCURACY_QDQ_SCALE_OVERFLOW,
        ISSUE_ACCURACY_QDQ_SCALE_UNDERFLOW,
        ISSUE_ACCURACY_QDQ_SCALE_ZERO,
        ISSUE_ACCURACY_UNFUSED_BN,
        ISSUE_PERFORMANCE_PREBN_CANNOT_BE_FUSED,
        ISSUE_PERFORMANCE_DANGLING_BN,
        ISSUE_PERFORMANCE_USE_GROUP_CONV,
        ISSUE_ACCURACY_BN_RUNNING_VARS_OVERFLOW,
        ISSUE_PERFORMANCE_QDQ_SCALE_MISMATCH,
        ISSUE_PERFORMANCE_QDQ_GRIDSAMPLE_INPUT_SHOULD_BE_HWC8,
        ISSUE_PERFORMANCE_QDQ_GRIDSAMPLE_INPUT_SHOULD_BE_CONV,
        ISSUE_PERFORMANCE_QDQ_DANGLING_QDQ,
        ISSUE_PERFORMANCE_REFORMAT_COPY,
        ISSUE_PERFORMANCE_REFORMAT,
    ]

    allow_penetrate_nodes = set(["QuantizeLinear", "DequantizeLinear", "Concat", "Split", "Slice", "Squeeze", "Unsqueeze", "Reshape", "Transpose", "Resize", "Gather", "Scatter", "Max", "ReduceMax", "TopK", "Flatten", "Pad", "Cast", "Identity"])

    def __init__(self, model, layers=None, profile=None, float_limit=10000, qscale_limit=1000, qscale_underflow_limit=1e-5, title=""):
        self.float_limit = float_limit
        self.qscale_limit = qscale_limit
        self.qscale_underflow_limit = qscale_underflow_limit
        self.model = model
        self.layers = layers
        self.profile = profile
        self.title = title
        self.issues = {}
        self.consumers = {}
        self.producers = {}
        self.initializers = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
        for i, node in enumerate(model.graph.node):
            if node.op_type == "Constant":
                self.initializers[node.output[0]] = onnx.numpy_helper.to_array(node.attribute[0].t)
                continue

            for input in node.input:
                if input not in self.consumers:
                    self.consumers[input] = []
                self.consumers[input].append([i + 1, node])

            for output in node.output:
                if output not in self.producers:
                    self.producers[output] = []
                self.producers[output].append([i + 1, node])

    def is_penetration_node(self, node):
        if node.op_type == "Resize":
            for attr in node.attribute:
                if attr.name == "mode":
                    if attr.s != b"nearest":
                        mode = str(attr.s, 'utf-8')
                        # self.log("Function.QDQ.ResizeNotNearest", f"Resize node {node.name} uses {mode} mode, but only 'nearest' mode can be conducted QDQ penetration.")
                        return False
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to":
                    if attr.i not in [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]:
                        return False

        return node.op_type in self.allow_penetrate_nodes

    def same_scales_partition_iterative(self, tensor, subgraph, qdq_set, iter_tensor_map, model_io_set):
        if tensor in iter_tensor_map or tensor is None or tensor == "":
            return

        if self.is_constant(tensor):
            return

        if self.is_model_io(tensor):
            model_io_set.add(tensor)

        iter_tensor_map.add(tensor)
        subgraph.append(tensor)
        for i, node in self.consumer(tensor, None) + self.producer(tensor, None):
            if not self.is_penetration_node(node):
                continue

            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                qdq_set.add(i)
            
            for input in node.input:
                self.same_scales_partition_iterative(input, subgraph, qdq_set, iter_tensor_map, model_io_set)

            for output in node.output:
                self.same_scales_partition_iterative(output, subgraph, qdq_set, iter_tensor_map, model_io_set)

    def find_QDQ_penetration_subgraphs(self):
        iter_tensor_map = set(self.initializers.keys())
        subgraphs = []
        for i, node in enumerate(self.model.graph.node):   
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                subgraph = []
                model_io_set = set()
                qdq_set = set([i + 1])
                for input in node.input:
                    self.same_scales_partition_iterative(input, subgraph, qdq_set, iter_tensor_map, model_io_set)

                for output in node.output:
                    self.same_scales_partition_iterative(output, subgraph, qdq_set, iter_tensor_map, model_io_set)

                if len(model_io_set) > 0:
                    # print(f"A QDQ penetration subgraph is found, but it contains model IO tensors: {model_io_set}")
                    continue

                major_scale = None
                has_mismatch_scale = False
                scales = []
                qdq_node_ids = []
                for node_id in qdq_set:
                    local_node = self.model.graph.node[node_id - 1]
                    if local_node.op_type in ["QuantizeLinear", "DequantizeLinear"] and len(local_node.attribute) == 0:
                        scale = self.initializers.get(local_node.input[1])
                        if scale is None:
                            self.log(type=HealthChecker.ISSUE_FUNCTION_QDQ_UNKNOWN_VALUE, severity=SeverityLevel.CRITICAL, location=local_node.input[1], location_type="tensor", message=f"scale for {local_node.input[1]} is not found")
                            continue

                        qdq_node_ids.append(node_id)
                        scale = scale.item()
                        scales.append(scale)
                        if major_scale is None:
                            major_scale = scale
                        elif major_scale != scale:
                            has_mismatch_scale = True
                if has_mismatch_scale:
                    subgraphs.append([subgraph, scales, qdq_node_ids])
        return subgraphs

    def is_model_io(self, tensor_name):
        return not self.is_constant(tensor_name) and (len(self.consumer(tensor_name, None)) == 0 or len(self.producer(tensor_name, None)) == 0)

    def is_constant(self, tensor_name):
        return tensor_name in self.initializers

    def producer(self, name, i=0):
        ps = self.producers.get(name, [])
        return (ps[i][1] if i < len(ps) else None) if i is not None else ps

    def consumer(self, name, i=0):
        cs = self.consumers.get(name, [])
        return (cs[i][1] if i < len(cs) else None) if i is not None else cs

    def get_tensor_value(self, name):
        return self.initializers.get(name, None)

    def log(self, **options):
        type = options["type"]
        location = options["location"]
        if not isinstance(location, list):
            location = [location]

        options["severity"] = options["severity"].name
        options["location"] = location
        if type not in self.issues:
            self.issues[type] = []
        self.issues[type].append(options)

    @staticmethod
    def get_filtered_types(filter=[])->set:
        if len(filter) == 0:
            return set(HealthChecker.all_issue_types)

        filtered_types = set()
        for type in HealthChecker.all_issue_types:
            for f in filter:
                if fnmatch.fnmatch(type, f):
                    filtered_types.add(type)
                    break
        return filtered_types

    def print_issues(self, filtered_types, summary_only=False):
        filtered_issues = {}
        for type, issues in self.issues.items():
            if type in filtered_types:
                filtered_issues[type] = issues

        total_issues = sum([len(issues) for issues in filtered_issues.values()])
        print(f">=================================== [{self.title}] Analysis Report: {total_issues} Issues Detected =====================================<")
        print(f"  [{self.title}] Total Issues Identified: {total_issues}")
        for type, issues in filtered_issues.items():
            print(f"     {type}: {len(issues)} issues")
        print(f">==============================================================================================================<")

        if summary_only:
            return

        for type, issues in filtered_issues.items():
            print(f"============================== [{self.title}] {type} - {len(issues)} Issues ===================================")
            for i, issue in enumerate(issues):
                print(f"--> {i+1}. {issue}")
            print("==========================================================================================================")

    def check_float_tensor(self, name, node_id, severity):
        tensor = self.get_tensor_value(name)
        if tensor is None:
            self.log(type=HealthChecker.ISSUE_FUNCTION_FLOAT_TENSOR_UNKNOWN_VALUE, severity=severity, location=node_id, location_type="tensor_locate_to_node", message=f"Tensor {name} has no value assigned")
            return

        if tensor.dtype in [np.float32, np.float64, np.float16]:
            mask = tensor > self.float_limit
            if mask.any():
                selected_values = -np.sort(-tensor[mask])[:5]
                shape = 'x'.join(map(str, tensor.shape))
                self.log(type=HealthChecker.ISSUE_ACCURACY_FLOATTENSOR_OVERFLOW, severity=severity, location=node_id, location_type="tensor_locate_to_node", message=f"Float Tensor [shape={shape}] contains values {selected_values} exceeding the float limit of {self.float_limit}: {name}")
                return False
        return True
    
    def check_qscale(self, name, node_id):
        tensor = self.get_tensor_value(name)
        if tensor is None:
            self.log(type=HealthChecker.ISSUE_FUNCTION_QDQ_UNKNOWN_VALUE, severity=SeverityLevel.CRITICAL, location=node_id, location_type="tensor_locate_to_node", message=f"Tensor {name} has no value assigned")
            return

        if tensor.dtype in [np.float32, np.float64, np.float16]:
            mask = tensor > self.qscale_limit
            if mask.any():
                masked_value = tensor[mask]
                num_masked_values = len(masked_value)
                selected_values = -np.sort(-masked_value)[:5]
                self.log(type=HealthChecker.ISSUE_ACCURACY_QDQ_SCALE_OVERFLOW, severity=SeverityLevel.CRITICAL, location=node_id, location_type="tensor_locate_to_node", message=f"Quantization scale contains {num_masked_values} values [{selected_values}] exceeding the scale limit of {self.qscale_limit}: {name}")
                return False
            
            mask = tensor < self.qscale_underflow_limit
            if mask.any():
                masked_value = tensor[mask]
                num_masked_values = len(masked_value)
                selected_values = np.sort(masked_value)[:5]
                self.log(type=HealthChecker.ISSUE_ACCURACY_QDQ_SCALE_UNDERFLOW, severity=SeverityLevel.LOW, location=node_id, location_type="tensor_locate_to_node", message=f"Quantization scale contains {num_masked_values} values [{selected_values}] below the underflow limit of {self.qscale_underflow_limit}: {name}")
                return False
            
            mask = tensor == 0
            if mask.any():
                self.log(type=HealthChecker.ISSUE_ACCURACY_QDQ_SCALE_ZERO, severity=SeverityLevel.CRITICAL, location=node_id, location_type="tensor_locate_to_node", message=f"Quantization scale contains {len(mask)} zero values: {name}")
                return False
        return True
    
    def check_bn(self, node, node_id):
        if node.op_type == "BatchNormalization":
            # self.log(type=HealthChecker.ISSUE_ACCURACY_UNFUSED_BN, location=node_id, location_type="node", message=f"Unfused BatchNormalization may lead to accuracy degradation: {node.name}")

            producer = self.producer(node.input[0])
            if producer is not None and producer.op_type not in ["Conv", "ConvTranspose", "Gemm"]:
                consumer = self.consumer(node.output[0])
                if consumer is not None:
                    if consumer.op_type in ["Conv", "ConvTranspose"]:
                        for attr in consumer.attribute:
                            if attr.name == "pads" and (attr.ints[-1] != 0 or attr.ints[-2] != 0):
                                self.log(type=HealthChecker.ISSUE_PERFORMANCE_PREBN_CANNOT_BE_FUSED, severity=SeverityLevel.LOW, location=node_id, location_type="node", message=f"Pre-BN can only be fused with {consumer.op_type} when using zero padding. Current padding: {attr.ints}: {node.name}")
                                break
                    else:
                        self.log(type=HealthChecker.ISSUE_PERFORMANCE_DANGLING_BN, severity=SeverityLevel.MEDIUM, location=node_id, location_type="node", message=f"Dangling BatchNormalization may impact performance: {node.name}")

            if len(node.input) > 4:
                running_vars = self.get_tensor_value(node.input[4])
                if running_vars is not None:
                    mask = running_vars > self.float_limit
                    if mask.any():
                        selected_values = -np.sort(-running_vars[mask])[:5]
                        self.log(type=HealthChecker.ISSUE_ACCURACY_BN_RUNNING_VARS_OVERFLOW, severity=SeverityLevel.CRITICAL, location=node_id, location_type="node", message=f"BatchNormalization running_vars contains values {selected_values} exceeding the float limit of {self.float_limit}: {node.name}")
    
    def check_gridsample(self, node, node_id):
        if node.op_type == "GridSample":
            producer = self.producer(node.input[0])
            if producer is None:
                self.log(type=HealthChecker.ISSUE_PERFORMANCE_QDQ_GRIDSAMPLE_INPUT_SHOULD_BE_HWC8, severity=SeverityLevel.MEDIUM, location=node_id, location_type="node", message=f"GridSample input should be a fp16:HWC8 tensor: {node.name}")
                return

            if producer.op_type not in ["Conv", "ConvTranspose"]:
                self.log(type=HealthChecker.ISSUE_PERFORMANCE_QDQ_GRIDSAMPLE_INPUT_SHOULD_BE_CONV, severity=SeverityLevel.MEDIUM, location=node_id, location_type="node", message=f"GridSample input should be produced by Conv or ConvTranspose operation: {node.name}")
                return
            
    def check_group_conv(self, node, node_id):
        if node.op_type in ["Conv", "ConvTranspose"]:
            for attr in node.attribute:
                if attr.name == "group":
                    if attr.i > 1:
                        self.log(type=HealthChecker.ISSUE_PERFORMANCE_USE_GROUP_CONV, severity=SeverityLevel.MEDIUM, location=node_id, location_type="node", message=f"Convolution with group size {attr.i} may impact performance: {node.name}")

    def check_all(self):
        for i, node in enumerate(self.model.graph.node):
            node_id = i + 1
            if node.op_type == "Constant":
                self.check_float_tensor(node.output[0], node_id, SeverityLevel.CRITICAL)

        for init in self.model.graph.initializer:
            self.check_float_tensor(init.name, [node_id for node_id, node in self.consumers.get(init.name, [])], SeverityLevel.CRITICAL)

        for i, node in enumerate(self.model.graph.node):
            node_id = i + 1
            if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
                self.check_qscale(node.input[1], node_id)

            self.check_bn(node, node_id)
            self.check_gridsample(node, node_id)
            self.check_group_conv(node, node_id)

        subgraphs = self.find_QDQ_penetration_subgraphs()
        if len(subgraphs) > 0:
            for subgraph, scales, qdq_node_ids in subgraphs:
                # node_names = [self.model.graph.node[i - 1].name for i in qdq_node_ids]
                message = f"Found a QDQ penetration subgraph with different scales: {len(qdq_node_ids)} nodes, Scales[min={min(scales)}, max={max(scales)}]: {scales}"
                self.log(type=HealthChecker.ISSUE_PERFORMANCE_QDQ_SCALE_MISMATCH, severity=SeverityLevel.HIGH, location=qdq_node_ids, location_type="subgraph", message=message)
        
        seq_issues = []
        for type, issues in self.issues.items():
            for issue in issues:
                seq_issues.append({
                    "type": type,
                    "severity": issue["severity"],
                    "location": issue["location"],
                    "location_type": issue["location_type"],
                    "message": issue["message"],
                })

        seq_issues = sorted(seq_issues, key=lambda x: SeverityLevel[x["severity"]].value, reverse=True)
        return seq_issues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("--layers", type=str)
    parser.add_argument("--profile", type=str)
    parser.add_argument("--float-limit", type=float, default=10000)
    parser.add_argument("--qscale-limit", type=float, default=1000)
    parser.add_argument("--qscale-underflow-limit", type=float, default=1e-5)
    parser.add_argument("--summary", action="store_true", help="Display summary only, without detailed information")
    parser.add_argument("--filter", nargs='+', default=[], help=f"Filter issue types, e.g. --filter Accuracy.FloatTensor.Overflow. Available types: {HealthChecker.all_issue_types}")
    args = parser.parse_args()

    filtered_types = HealthChecker.get_filtered_types(args.filter)
    model_name = os.path.splitext(os.path.basename(args.model))[0]
    print(f"Analyzing ONNX model '{model_name}' with {len(filtered_types)} issue types: {filtered_types}")
    model = onnx.load(args.model)
    checker = HealthChecker(model, args.layers, args.profile, args.float_limit, args.qscale_limit, args.qscale_underflow_limit, model_name)
    checker.check_all()
    checker.print_issues(filtered_types, args.summary)
