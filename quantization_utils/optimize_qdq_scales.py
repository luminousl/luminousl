import onnx
import argparse
import os

penetration_nodes = set(["QuantizeLinear", "DequantizeLinear", "Concat", "Split", "Slice", "Squeeze", "Unsqueeze", "Reshape", "Transpose", "Resize", "Gather", "Scatter", "Max", "ReduceMax", "TopK", "Flatten", "Pad", "Cast", "Identity"])

def is_penetration_node(node):
    if node.op_type == "Resize":
        for attr in node.attribute:
            if attr.name == "mode":
                if attr.s != b"nearest":
                    # print(f"Warning: Resize node {node.name} uses {str(attr.s, 'utf-8')} mode, but only 'nearest' mode can be conducted QDQ penetration.")
                    return False
                
    return node.op_type in penetration_nodes

def same_scales_partition_iterative(producers, consumers, tensor, subgraph, qdq_set, iter_tensor_map):
    if tensor in iter_tensor_map or tensor is None or tensor == "":
        return

    iter_tensor_map.add(tensor)
    subgraph.append(tensor)
    for i, node in consumers.get(tensor, []) + producers.get(tensor, []):
        if not is_penetration_node(node):
            continue

        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            qdq_set.add(i)
        
        for input in node.input:
            same_scales_partition_iterative(producers, consumers, input, subgraph, qdq_set, iter_tensor_map)

        for output in node.output:
            same_scales_partition_iterative(producers, consumers, output, subgraph, qdq_set, iter_tensor_map)

def find_QDQ_penetration_subgraphs(model):
    producers = {}
    consumers = {}
    initializer_names = set([init.name for init in model.graph.initializer])
    initializers      = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Constant":
            initializer_names.add(node.output[0])
            initializers[node.output[0]] = onnx.numpy_helper.to_array(node.attribute[0].t)
            continue

        for input in node.input:
            if input not in consumers:
                consumers[input] = []
            consumers[input].append([i, node])

        for output in node.output:
            if output not in producers:
                producers[output] = []
            producers[output].append([i, node])
    
    iter_tensor_map = initializer_names
    subgraphs = []
    for i, node in enumerate(model.graph.node):   
        if node.op_type in ["QuantizeLinear", "DequantizeLinear"]:
            subgraph = []
            qdq_set = set([i])
            for input in node.input:
                same_scales_partition_iterative(producers, consumers, input, subgraph, qdq_set, iter_tensor_map)

            for output in node.output:
                same_scales_partition_iterative(producers, consumers, output, subgraph, qdq_set, iter_tensor_map)

            major_scale = None
            has_mismatch_scale = False
            scales = []
            qdq_node_ids = []
            for node_id in qdq_set:
                local_node = model.graph.node[node_id]
                if local_node.op_type in ["QuantizeLinear", "DequantizeLinear"] and len(local_node.attribute) == 0:
                    scale = initializers.get(local_node.input[1])
                    if scale is None:
                        print(f"{'':>6}Warning: scale for {local_node.input[1]} is not found")
                        continue

                    qdq_node_ids.append(node_id)
                    try:
                        scale = scale.item()
                    except:
                        breakpoint()
                    scales.append(scale)
                    if major_scale is None:
                        major_scale = scale
                    elif major_scale != scale:
                        has_mismatch_scale = True
            if has_mismatch_scale:
                subgraphs.append([subgraph, scales, qdq_node_ids])
    return subgraphs

def optimize_qdq_scales(model_path, out_path=None, scale_threshold=0.95):
    print("=== optimize qdq scales ===")
    model = onnx.load(model_path)
    if out_path is None:
        out_path = model_path.replace(".onnx", "_qdqopt.onnx")
    subgraphs = find_QDQ_penetration_subgraphs(model)
    num_changed = 0
    for igraph, (subgraph, scales, qdq_node_ids) in enumerate(subgraphs):
        qdq_node_names = [model.graph.node[i].name for i in qdq_node_ids]
        min_scale = min(scales)
        max_scale = max(scales)
        if min_scale < max_scale * scale_threshold:
            print(f"{'':>6}***Warning: the min scale {min_scale} is less than {scale_threshold} of the max scale {max_scale} for {qdq_node_names}, accuracy may be affected")
            for a,b in zip(scales, qdq_node_names):
                print(f"--- {a}, {b}")
            continue
        
        num_changed += 1
        print(f"{'':>6}The min scale is {min_scale} and the max scale is {max_scale} for {len(qdq_node_names)} nodes, scales: {scales}")
        
        max_scale_init = onnx.helper.make_tensor(
            name=f"{'':>6}max_scale_for_subgraph{igraph}",
            data_type=onnx.TensorProto.FLOAT,
            dims=[1],
            vals=[max_scale]
        )
        model.graph.initializer.append(max_scale_init)
        for node_id in qdq_node_ids:
            node = model.graph.node[node_id]
            print(f"{'':>6}subgraphs{igraph}, {node.name}")
            node.input[1] = max_scale_init.name

    if num_changed > 0:
        print(f"{'':>6}Save the processed model to {out_path}, {num_changed} subgraphs are changed")
        onnx.save(model, out_path)
    else:
        print(f"{'':>6}No subgraphs are changed, no new model is saved.")
        return model_path
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the scale subgraphs and replace the scales with the max scale")
    parser.add_argument("model", type=str, help="The model to process")
    parser.add_argument("--output", "-o", type=str, help="The output model")
    parser.add_argument("--force", "-f", action="store_true", help="Force to run even if the output file exists")
    parser.add_argument("--scale-threshold", "-t", type=float, default=0.95, help="The threshold for the scale")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.splitext(args.model)[0] + ".processed.onnx"

    if not args.force and os.path.exists(args.output):
        option = input(f"The output file {args.output} already exists, do you want to overwrite it? (y/n): ")
        if option != "y" and option != "Y":
            print("Cancel to run.")
            exit(0)

    model = onnx.load(args.model)
    subgraphs = find_QDQ_penetration_subgraphs(model)
    num_changed = 0
    for igraph, (subgraph, scales, qdq_node_ids) in enumerate(subgraphs):
        qdq_node_names = [model.graph.node[i].name for i in qdq_node_ids]
        min_scale = min(scales)
        max_scale = max(scales)
        if min_scale < max_scale * args.scale_threshold:
            print(f"***Warning: the min scale {min_scale} is less than {args.scale_threshold} of the max scale {max_scale} for {qdq_node_names}, accuracy may be affected")
            continue
        
        num_changed += 1
        print(f"The min scale is {min_scale} and the max scale is {max_scale} for {len(qdq_node_names)} nodes")
        
        max_scale_init = onnx.helper.make_tensor(
            name=f"max_scale_for_subgraph{igraph}",
            data_type=onnx.TensorProto.FLOAT16,
            dims=[1],
            vals=[max_scale]
        )
        model.graph.initializer.append(max_scale_init)
        for node_id in qdq_node_ids:
            node = model.graph.node[node_id]
            print(f"subgraphs{igraph}, {node.name}")
            node.input[1] = max_scale_init.name

    if num_changed > 0:
        print(f"Save the processed model to {args.output}, {num_changed} subgraphs are changed")
        onnx.save(model, args.output)
    else:
        print(f"No subgraphs are changed, no new model is saved.")