import onnx
import numpy as np
import onnx.helper as helper
import onnx.numpy_helper as nphelper
import traceback
from io import StringIO
import contextlib
import onnx.shape_inference

# onnx_model:onnx.ModelProto = onnx.load("/datav/onnx_views_data/product_data/files/000000006936")
to_npdtype = {
    "float16": np.float16,
    "int8": np.int8,
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
    "float8e4m3fn": np.int8
}

to_onnxdtype = {
    "float16": onnx.TensorProto.FLOAT16,
    "float32": onnx.TensorProto.FLOAT,
    "int8": onnx.TensorProto.INT8,
    "int32": onnx.TensorProto.INT32,
    "int64": onnx.TensorProto.INT64,
    "float8e4m3fn": onnx.TensorProto.FLOAT8E4M3FN
}

keywords = set(["Q", "DQ", "QDQ", "WQDQ", "Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Abs", "Neg", "Floor", "Ceil", "Round", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Atan2", "Sinh", "Cosh", "Conv", "Gemm", "Matmul", "MatMul", "MaxPool", "AvgPool", "GlobalAvgPool", "GlobalAveragePool", "GlobalAveragePooling", "Linear", "Dense", "LayerNorm", "BN", "BatchNorm", "BatchNormalization", "Dropout", "Softmax", "LogSoftmax", "Gelu", "SiLU", "Concat", "Split", "Slice", "Clip", "HardSigmoid", "HardSwish", "Transpose", "Reshape", "Flatten", "Unsqueeze", "Squeeze", "Deconv", "ConvTranspose", "LSTM", "RNN", "LSTMCell", "GRUCell", "Gather", "Scatter", "ScatterND", "Upsample", "Resize", "ArgMax", "ArgMin", "TopK", "OneHot", "Softplus", "Softsign", "Cast", "QuantizeLinear", "DequantizeLinear"])

def generate_onnx_from_code(code:str, reference_model:onnx.ModelProto=None):
    initializers = {}
    if reference_model is not None:
        initializers = {init.name : init for init in reference_model.graph.initializer}
        for op in reference_model.graph.node:
            if op.op_type == "Constant":
                initializers[op.output[0]] = op.attribute[0].t

    new_inputs = []
    new_outputs = []
    new_initializers = []
    new_nodes = []
    initializer_idd = 0
    current_codeline = 0
    vars_codeline_mapping = {}
    nodes_codeline_mapping = {}
    codeline_to_objs = {}
    qdq_default_type = "float32", "int8"
    def config_qdq_type(scale, zero_point):
        nonlocal qdq_default_type
        qdq_default_type = scale, zero_point

    def set_codeline(i):
        nonlocal current_codeline
        current_codeline = i

    def set_codeline_obj(obj):
        nonlocal codeline_to_objs
        if current_codeline not in codeline_to_objs:
            codeline_to_objs[current_codeline] = []
        codeline_to_objs[current_codeline].append(obj)

    def get_initializer_name():
        nonlocal initializer_idd
        initializer_idd += 1
        name = f"t{initializer_idd}"
        vars_codeline_mapping[name] = current_codeline
        set_codeline_obj(dict(name=name, type="initializer"))
        return name

    layer_output_idd = 0
    def get_layer_variable_name():
        nonlocal layer_output_idd
        layer_output_idd += 1
        name = f"x{layer_output_idd}"
        vars_codeline_mapping[name] = current_codeline
        set_codeline_obj(dict(name=name, type="variable"))
        return name

    layer_idd_per_op = {}
    def get_layer_name(optype):
        nonlocal layer_idd_per_op
        if optype not in layer_idd_per_op:
            layer_idd_per_op[optype] = 0
        layer_idd_per_op[optype] += 1
        return f"{optype}{layer_idd_per_op[optype]}"

    def input_to_name(input):
        if isinstance(input, str):
            return input
        
        name = get_initializer_name()
        if isinstance(input, float):
            new_initializers.append(nphelper.from_array(np.array(input, dtype=np.float32), name))
        elif isinstance(input, int):
            new_initializers.append(nphelper.from_array(np.array(input, dtype=np.int64), name))
        elif isinstance(input, (tuple, list)):
            array = np.array(input)
            if array.dtype == np.float64:
                array = array.astype(np.float32)
            new_initializers.append(nphelper.from_array(array, name))
        else:
            assert isinstance(input, np.ndarray), f"Must be a ndarray object, but give a {type(input)}: {input}"
            new_initializers.append(nphelper.from_array(input, name))
        return name

    def Tensor(dtype, shape=[1], name=None):
        if name in initializers:
            vars_codeline_mapping[name] = current_codeline
            set_codeline_obj(dict(name=name, type="variable"))
            return nphelper.to_array(initializers[name]).reshape(shape).astype(to_npdtype[dtype])
        return np.zeros(shape, dtype=to_npdtype[dtype])
    
    def full(val, shape, dtype):
        if shape is None: return np.array(val, dtype=dtype)
        return np.full(shape, val, dtype=dtype)

    def i8(val=0, shape=None):
        return full(val, shape, np.int8)
    
    def i32(val=0, shape=None):
        return full(val, shape, np.int32)
    
    def i64(val=0, shape=None):
        return full(val, shape, np.int64)
    
    def f16(val=0, shape=None):
        return full(val, shape, np.float16)
    
    def f32(val=0, shape=None):
        return full(val, shape, np.float32)
    
    def expend_value(val, n):
        if isinstance(val, (int, float)):
            return [val] * n
        return val

    def make_layer(optype):

        if optype in ["BN", "BatchNorm"]: optype = "BatchNormalization"
        if optype in ["ReLU"]: optype = "Relu"

        def impl(*inputs, **attrs):
            num_outputs = attrs.get("outputs", 1)
            if "outputs" in attrs:
                del attrs["outputs"]

            if "name" in attrs:
                layer_name = attrs["name"]
                del attrs["name"]
            else:
                layer_name = get_layer_name(optype)

            if optype in ["QDQ", "Q", "DQ"]:
                inputs = list(inputs)
                if len(inputs) == 1:
                    inputs.append(full(1, None, qdq_default_type[0]))
                    inputs.append(full(0, None, qdq_default_type[1]))
                elif len(inputs) == 2:
                    inputs.append(full(0, None, qdq_default_type[1]))

                assert len(inputs) == 3, "Missing inputs for QDQ node."
                if optype == "QDQ":
                    Q = make_layer("QuantizeLinear")
                    DQ = make_layer("DequantizeLinear")
                    others = inputs[1:]
                    return DQ(Q(*inputs, **attrs), *others, **attrs)
                elif optype == "Q":
                    return make_layer("QuantizeLinear")(*inputs, **attrs)
                elif optype == "DQ":
                    return make_layer("DequantizeLinear")(*inputs, **attrs)
            elif optype == "WQDQ":
                # WQDQ(w, 128)
                assert len(inputs) >= 2, f"Missing dims argument for WQDQ node. WQDQ(w, dims, scale, zero_point, axis=0)"
                inputs = list(inputs)
                dims   = inputs[1]
                assert isinstance(dims, int), f"Invalid argument type for argument #2."
                del inputs[1]
                if len(inputs) == 1:
                    inputs.append(full(1, dims, qdq_default_type[0]))
                    inputs.append(full(0, dims, qdq_default_type[1]))
                elif len(inputs) == 2:
                    inputs.append(full(0, dims, qdq_default_type[1]))

                if "axis" not in attrs:
                    attrs["axis"] = 0

                assert len(inputs) == 3, "Missing inputs for WQDQ node."
                Q = make_layer("QuantizeLinear")
                DQ = make_layer("DequantizeLinear")
                others = inputs[1:]
                return DQ(Q(*inputs, **attrs), *others, **attrs)

            if optype in ["Conv", "ConvTranspose"]:
                for name in attrs:
                    if name == "kernel_shape": attrs[name] = expend_value(attrs[name], 2)
                    elif name == "pads": attrs[name] = expend_value(attrs[name], 4)
                    elif name == "strides": attrs[name] = expend_value(attrs[name], 2)
                    elif name == "dilations": attrs[name] = expend_value(attrs[name], 2)
                
                if "dilations" not in attrs: attrs["dilations"] = [1, 1]
                if "pads" not in attrs: attrs["pads"] = [0, 0, 0, 0]
                if "strides" not in attrs: attrs["strides"] = [1, 1]
                if "group" not in attrs: attrs["group"] = 1

            inputs_names = [input_to_name(input) for input in inputs]
            output_names = [get_layer_variable_name() for i in range(num_outputs)]
            new_nodes.append(helper.make_node(optype, inputs_names, output_names, layer_name, **attrs))
            node_idd = len(new_nodes)
            nodes_codeline_mapping[node_idd] = current_codeline
            set_codeline_obj(dict(node_idd=node_idd, type="node"))

            if len(output_names) == 1:
                return output_names[0]
            return output_names
        return impl

    def Input(dtype, shape=[], name=None):
        if name is None:
            name = get_layer_variable_name()
        new_inputs.append(helper.make_tensor_value_info(name, to_onnxdtype[dtype], shape))
        vars_codeline_mapping[name] = current_codeline
        set_codeline_obj(dict(name=name, type="variable"))
        return name

    def Output(tensor, dtype="float32", shape=[], name=None):
        if name is not None:
            for node in new_nodes:
                for i in range(len(node.input)):
                    if node.input[i] == tensor:
                        node.input[i] = name

                for i in range(len(node.output)):
                    if node.output[i] == tensor:
                        node.output[i] = name

            if tensor in vars_codeline_mapping:
                line = vars_codeline_mapping[tensor]
                del vars_codeline_mapping[tensor]
                vars_codeline_mapping[name] = line

            for codeline in codeline_to_objs:
                for obj in codeline_to_objs[codeline]:
                    if obj["type"] == "variable" and obj["name"] == tensor:
                        obj["name"] = name
            
            tensor = name

        new_outputs.append(helper.make_tensor_value_info(tensor, to_onnxdtype[dtype], shape))
        vars_codeline_mapping[tensor] = current_codeline
        set_codeline_obj(dict(name=tensor, type="variable"))
        return None

    def layer(optype, *inputs, **attrs):
        return make_layer(optype)(*inputs, **attrs)

    def warning(message):
        def impl(*args, **kwrags):
            print(message)
        return impl

    global_vars = dict(
        layer = layer,
        Input = Input,
        Output = Output,
        Tensor = Tensor,
        int8 = i8,
        int32 = i32,
        int64 = i64,
        float16 = f16,
        float32 = f32,
        fp32 = "float32",
        fp16 = "float16",
        i8 = "int8",
        i32 = "int32",
        i64 = "int64",
        fp8 = "float8e4m3fn",
        full = full,
        config_qdq_type = config_qdq_type,
        np = np,
        set_codeline = set_codeline,
        exit = warning("exit function is not allowed.")
    )
    local_vars = dict()

    for keyword in keywords:
        global_vars[keyword] = make_layer(keyword)

    codelines = code.split("\n")
    newlines  = []
    for i in range(len(codelines)):
        if len(codelines[i]) > 0 and codelines[i][0] not in [" ", "\t"] and not codelines[i].startswith("def ") and not codelines[i].startswith("@") \
             and not codelines[i].startswith("class ") and not codelines[i].startswith("import ") and not codelines[i].startswith("from "):
            newlines.append(f"set_codeline({i})")
        newlines.append(codelines[i])
    
    new_code = "\n".join(newlines)
    console_output = StringIO()
    try:
        with contextlib.redirect_stderr(console_output):
            with contextlib.redirect_stdout(console_output):
                exec(new_code, global_vars, local_vars)
    except Exception as e:
        # traceback.print_exc()
        return dict(status="error", message=str(e), traceback=traceback.format_exc(), console_output=console_output.getvalue())

    model = helper.make_model(
        helper.make_graph(
            new_nodes, "", new_inputs, new_outputs, new_initializers
        )
    )

    try:
        model = onnx.shape_inference.infer_shapes(model, False, False)
    except Exception as e:
        pass

    return dict(
        status="success", model=model, 
        console_output=console_output.getvalue(), 
        code_metas=dict(
            vars_codeline_mapping = vars_codeline_mapping,
            nodes_codeline_mapping = nodes_codeline_mapping,
            codeline_to_objs = codeline_to_objs
        )
    )