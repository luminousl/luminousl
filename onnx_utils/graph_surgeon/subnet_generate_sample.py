import onnx
import onnx_graphsurgeon as gs
import numpy as np

def subnet_generate_example(onnx_path):
    # 加载ONNX模型
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    
    # 获取图中所有的张量（tensors）
    tensors = graph.tensors()
    sub_model_name = onnx_path.replace(".onnx", ".subnet.onnx")
    
    # 标记所有的中间层的特征图输出，将作为子网络的输入
    target_input_names = [
        "/group1/stage2/block5/activation/Relu_output_0", # -> neck1
        "/group1/stage3/block15/activation/Relu_output_0", # -> neck1
        "/group1/stage4/block0/activation/Relu_output_0", # ->neck1
        "/group0/stage2/block5/activation/Relu_output_0", # ->neck4
        "/group0/stage3/block15/activation/Relu_output_0", # ->neck4
        "/group0/stage4/block0/activation/Relu_output_0", # ->neck4
    ]
    
    # 将目标输入张量添加到图的输入列表中
    for input_name in target_input_names:
        graph.inputs.append(tensors[input_name])
    
    # 标记需要作为子模型输出的张量名称
    target_output_names = [
        "MONO_OD_1",
        "MONO_OD_2",
        "MONO_OD_3",
        "SIDE_OD_1",
        "SIDE_OD_2",
        "SIDE_OD_3",
    ]
    
    # 清空原有的输出列表，重新设置输出
    graph.outputs = []
    
    # 将目标输出张量添加到图的输出列表中
    for output_name in target_output_names:
        graph.outputs.append(tensors[output_name])
    
    # 清理图中无用的节点并进行拓扑排序，确保图结构正确
    graph.cleanup().toposort()
    
    # 查找并移除没有下游节点的输入（游离输入）
    free_inputs = [inp for inp in graph.inputs if len(inp.outputs) < 1]
    for inp in free_inputs:
        graph.inputs.remove(inp)
    
    # 将graphsurgeon图转换回ONNX模型格式, 保存生成的子模型到文件
    sub_model = gs.export_onnx(graph)
    onnx.save(sub_model, sub_model_name)
    print("=== onnx exported to: ", sub_model_name)
    print(f"节点数量: {len(sub_model.graph.node)} 输入数量:{len(graph.inputs)} 输出数量:{len(graph.outputs)}")
    return sub_model_name