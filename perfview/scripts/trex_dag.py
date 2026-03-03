from trex.graphing import to_dot, layer_type_formatter, layer_colormap
from trex import EnginePlan
from trex.graphing import OnnxGraph, latency_types, _get_latency
import datetime
import onnx.helper as helper
import onnx

def now():
    return datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

def dprint(msg):
    if not isinstance(msg, str):
        msg = str(msg)

    print(f"{now()}: {msg}", flush=True)

layer_colormap.update({
    "correlation": "#4682B4",
    "kgen": "#34a853",
    "fusion": "#4285f4"
})

def get_dot_dag(graph_file, profile_file=None):
    plan = EnginePlan(graph_file, profile_file)
    formatter = layer_type_formatter
    display_regions = True
    expand_layer_details = False
    graph = to_dot(plan, formatter,
                display_regions=display_regions,
                expand_layer_details=expand_layer_details)
    return graph

def tonptype(desc):
    desc = desc.lower()
    if 'int8' in desc:
        return onnx.TensorProto.INT8
    elif 'fp32' in desc:
        return onnx.TensorProto.FLOAT
    elif 'fp16' in desc:
        return onnx.TensorProto.FLOAT16
    elif 'int32' in desc:
        return onnx.TensorProto.INT32
    elif 'half' in desc:
        return onnx.TensorProto.FLOAT16
    else:
        raise ValueError(f"Uknown precision {desc}")

def get_onnx_dag(graph_file, profile_file=None):
    plan = EnginePlan(graph_file, profile_file)

    nodes = []
    graph_inputs, graph_outputs = plan.get_bindings()
    for layer in plan.layers:
        attributes = [
            helper.make_attribute("metadata", layer.metadata),
            helper.make_attribute("inputs_size_bytes", layer.inputs_size_bytes),
            helper.make_attribute("precision", layer.precision if layer.precision is not None else ""),
            helper.make_attribute("total_footprint_bytes", layer.total_footprint_bytes),
            helper.make_attribute("total_io_size_bytes", layer.total_io_size_bytes),
            helper.make_attribute("outputs_size_bytes", layer.outputs_size_bytes),
            helper.make_attribute("tactic_name", layer.raw_dict.get("TacticName", "")),
            helper.make_attribute("stream_id", layer.raw_dict.get("StreamId", -1)),
        ]

        for latency_type in latency_types:
            attributes.append(helper.make_attribute(f"latency_{latency_type}", _get_latency(plan, layer, latency_type)))

        inputs  = [item.name for item in layer.inputs]
        outputs = [item.name for item in layer.outputs]
        node = helper.make_node(layer.type, inputs, outputs, layer.name)
        node.attribute.extend(attributes)
        nodes.append(node)

    inputs  = [helper.make_tensor_value_info(item.name, tonptype(item.precision), item.shape) for item in graph_inputs]
    outputs = [helper.make_tensor_value_info(item.name, tonptype(item.precision), item.shape) for item in graph_outputs]
    return helper.make_model(
        helper.make_graph(nodes, "model", inputs, outputs)
    )

# graph = get_dot_dag("../datas/zeekrmodelstrack/kpi_models_eq/.trt/.orin8.x_sparsity/15_onemodel_v2_ds_layers.json", "../datas/zeekrmodelstrack/kpi_models_eq/.trt/.orin8.x_sparsity/15_onemodel_v2_ds_profile.json")
graph = get_onnx_dag("../datas/zeekrmodelstrack/kpi_models_eq/.trt/.orin8.x_sparsity/15_onemodel_v2_ds_layers.json", "../datas/zeekrmodelstrack/kpi_models_eq/.trt/.orin8.x_sparsity/15_onemodel_v2_ds_profile.json")
onnx.save(graph, "onnxs/trex.onnx")