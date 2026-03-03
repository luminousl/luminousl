import onnx
import traceback

model_path = "/datav/jingweid/onnx_playboard/product_datas/files/000000000015"
model = onnx.load(model_path)

try:
    model = onnx.shape_inference.infer_shapes(model)
except Exception as e:
    traceback.print_exception(e)

layerids = [2285,2288,2302,2317,2318,2338,5190]
initializer_mapping = set([item.name for item in model.graph.initializer])
constant_output_to_nodeid_mapping = {node.output[0]:node_id + 1 for node_id, node in enumerate(model.graph.node) if node.op_type == "Constant"}
reference_initializers = set()
reference_constant_node_id = set()
reference_constant_inp       = set()
for i, node in enumerate(model.graph.node):
    node_id = i + 1
    if node_id in layerids:
        for inp in node.input:
            if inp in initializer_mapping:
                reference_initializers.add(inp)
            elif inp in constant_output_to_nodeid_mapping:
                reference_constant_node_id.add(constant_output_to_nodeid_mapping[inp])
                reference_constant_inp.add(inp)
        
for i in range(len(model.graph.node)-1, -1, -1):
    node_id = i + 1
    if node_id not in layerids and node_id not in reference_constant_node_id:
        del model.graph.node[i]

for i in range(len(model.graph.initializer)-1, -1, -1):
    if model.graph.initializer[i].name not in reference_initializers:
        del model.graph.initializer[i]

input_to_node  = {}
output_to_node = {}
for node in model.graph.node:
    if node.op_type == "Constant":
        continue

    for inp in node.input:
        if inp != "":
            input_to_node[inp] = node

    for out in node.output:
        output_to_node[out] = node

network_outputs   = {item.name: item for item in model.graph.output}
undefined_inputs  = []
undefined_outputs = []
for inp in input_to_node:
    if inp not in output_to_node and inp not in reference_initializers and inp not in reference_constant_inp:
        undefined_inputs.append(inp)

for out in output_to_node:
    if out in input_to_node and out in network_outputs or out not in input_to_node:
        undefined_outputs.append(out)

for i in range(len(model.graph.input)-1, -1, -1):
    if model.graph.input[i].name not in undefined_inputs:
        del model.graph.input[i]
    else:
        del undefined_inputs[undefined_inputs.index(model.graph.input[i].name)]

for i in range(len(model.graph.output)-1, -1, -1):
    if model.graph.output[i].name not in undefined_outputs:
        del model.graph.output[i]
    else:
        del undefined_outputs[undefined_outputs.index(model.graph.output[i].name)]

tensor_info_mapping = {item.name : item for item in model.graph.value_info}
for inp in undefined_inputs:
    info = tensor_info_mapping[inp]
    model.graph.input.append(onnx.ValueInfoProto(name=inp, type=info.type))

for out in undefined_outputs:
    if out in tensor_info_mapping:
        info = tensor_info_mapping[out]
        model.graph.output.append(onnx.ValueInfoProto(name=out, type=info.type))
    else:
        model.graph.output.append(onnx.ValueInfoProto(name=out))

print(model.graph.output)
# onnx.save_model(model, "subgraph.onnx")