import onnx
import json
import os
import argparse
import onnx.helper as helper
import numpy as np
import traceback

def extract_initializer_data_view(init:onnx.TensorProto, num_elm_keeped:int=64):
    result = onnx.numpy_helper.to_array(init)
    if result is not None and result.size > 0:
        return result.reshape(-1)[:num_elm_keeped].astype(np.str_).tolist()
    return []

def attr2val(attr:onnx.AttributeProto):
    if attr.type == onnx.AttributeProto.FLOAT:
        return dict(value=attr.f, dtype="float", name=attr.name)
    elif attr.type == onnx.AttributeProto.STRING:
        return dict(value=str(attr.s, "utf-8"), dtype="string", name=attr.name)
    elif attr.type == onnx.AttributeProto.INT:
        return dict(value=attr.i, dtype="int", name=attr.name)
    elif attr.type == onnx.AttributeProto.FLOATS:
        return dict(value=list(attr.floats), dtype="float_array", name=attr.name)
    elif attr.type == onnx.AttributeProto.INTS:
        return dict(value=list(attr.ints), dtype="int_array", name=attr.name)
    elif attr.type == onnx.AttributeProto.TENSOR:
        return dict(value=extract_initializer_data_view(attr.t), dtype="tensor", name=attr.name)
    return dict(value="unknow", dtype="unknow " + int(attr.type), name=attr.name)

def shape2list(shape):
    return [dim.dim_param if dim.dim_param != "" else dim.dim_value for dim in shape]

def dtype2str(dtype):
    if dtype == onnx.TensorProto.FLOAT:
        return "float32"
    elif dtype == onnx.TensorProto.FLOAT16:
        return "float16"
    elif dtype == onnx.TensorProto.BOOL:
        return "bool"
    elif dtype == onnx.TensorProto.INT8:
        return "int8"
    elif dtype == onnx.TensorProto.INT16:
        return "int16"
    elif dtype == onnx.TensorProto.INT32:
        return "int32"
    elif dtype == onnx.TensorProto.DOUBLE:
        return "float64"
    elif dtype == onnx.TensorProto.FLOAT8E4M3FN:
        return "flaot8e4m3fn"
    elif dtype == onnx.TensorProto.FLOAT8E4M3FNUZ:
        return "flaot8e4m3fnuz"
    elif dtype == onnx.TensorProto.FLOAT8E5M2:
        return "flaot8e5m2"
    elif dtype == onnx.TensorProto.FLOAT8E5M2FNUZ:
        return "flaot8e5m2fnuz"
    elif dtype == onnx.TensorProto.STRING:
        return "string"
    elif dtype == onnx.TensorProto.BFLOAT16:
        return "bfloat16"
    elif dtype == onnx.TensorProto.UINT32:
        return "uint32"
    elif dtype == onnx.TensorProto.UINT16:
        return "uint16"
    elif dtype == onnx.TensorProto.UINT8:
        return "uint8"
    elif dtype == onnx.TensorProto.UINT64:
        return "uint64"
    elif dtype == onnx.TensorProto.INT64:
        return "int64"
    return "unknow_type_" + str(dtype)

def dump_onnx(proto:onnx.ModelProto):
    graph_nodes = []
    addition_inits = []
    for inode, node in enumerate(proto.graph.node):
        if node.op_type == "Constant":
            if len(node.attribute) > 0 and len(node.output) > 0:
                data = node.attribute[0]
                data.t.name = node.output[0]
                addition_inits.append(data.t)
                continue

        graph_nodes.append(dict(
            name=node.name,
            idd=inode + 1,
            input=list(node.input),
            output=list(node.output),
            optype=node.op_type,
            domain=node.domain,
            attrs=[attr2val(attr) for attr in node.attribute]
        ))

    return dict(
        node = graph_nodes,
        initializer = [
            dict(
                name=init.name,
                shape=list(init.dims),
                dtype=dtype2str(init.data_type),
                data_view=extract_initializer_data_view(init)
            ) for init in list(proto.graph.initializer) + addition_inits
        ],
        tensor_info = [
            dict(
                name=info.name,
                dtype=dtype2str(info.type.tensor_type.elem_type),
                shape=shape2list(info.type.tensor_type.shape.dim)
            ) for info in proto.graph.value_info
        ],
        input = [
            dict(name=inp.name, shape=shape2list(inp.type.tensor_type.shape.dim), dtype=dtype2str(inp.type.tensor_type.elem_type), idd=len(proto.graph.node) + i + 1) for i, inp in enumerate(proto.graph.input)
        ],
        output = [
            dict(name=oup.name, shape=shape2list(oup.type.tensor_type.shape.dim), dtype=dtype2str(oup.type.tensor_type.elem_type), idd=len(proto.graph.node) + len(proto.graph.input) + i + 1) for i, oup in enumerate(proto.graph.output)
        ]
    )

def onnx_layout_generate(onnx_proto, graph_file=None, additions={}, caching_file=None):
    try:
        graph = dump_onnx(onnx_proto)
        graph.update(additions)

        layout_done = False
        if caching_file is not None:
            try:
                if os.path.exists(caching_file):
                    with open(caching_file, "r") as f:
                        graph["layout"] = json.load(f)
                        layout_done = True
                        print(f"Loaded layout from caching file: {caching_file}")
            except Exception as e:
                print(f"Failed to load caching file: {caching_file}, {e}")

        with open(graph_file, "w") as f:
            f.write(json.dumps(graph, indent=4))

        if layout_done:
            return True

        args = [graph_file]
        if caching_file is not None:
            args.append(caching_file)
        
        args = " ".join([f'"{item}"' for item in args])
        code = os.system(f"node --max-old-space-size=40960 layout.js {args}")
        if code == 0:
            print(f"Graph.json has saved in {graph_file}")
            return True

        print(f"Failed to run layout.js: code = {code}")
    except Exception as e:
        # traceback.print_exception(e)
        print(f"Error: {e}, graph_file = {graph_file}")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate ONNX layout json.")
    parser.add_argument("file", type=str, help="Which onnx file are need to generate.")
    args = parser.parse_args()
    onnx_layout_generate(onnx.load(args.file), f"{args.file}.json")
