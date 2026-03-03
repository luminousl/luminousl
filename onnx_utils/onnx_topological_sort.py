import onnx
import onnx_graphsurgeon as gs
from collections import defaultdict
from collections import deque

# 基于广度的拓扑排序
def topological_sort_by_wfs(onnx_graph):
    # 创建节点名称到节点的映射
    name_to_node = {node.name: node for node in onnx_graph.nodes}
    
    # 构建邻接表和入度表
    in_degree = {node.name: 0 for node in onnx_graph.nodes}
    adj = {node.name: [] for node in onnx_graph.nodes}
    
    # 填充邻接表和入度表
    for node in onnx_graph.nodes:
        for output in node.outputs:
            for other_node in onnx_graph.nodes:
                if output in other_node.inputs:
                    adj[node.name].append(other_node.name)
                    in_degree[other_node.name] += 1
    
    # 初始化队列 (入度为0的节点)
    queue = deque([node.name for node in onnx_graph.nodes if in_degree[node.name] == 0])
    sorted_nodes = []
    
    # 执行拓扑排序
    while queue:
        current = queue.popleft()
        sorted_nodes.append(name_to_node[current])
        
        for neighbor in adj[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return sorted_nodes

# 基于深度的拓扑排序
def topological_sort_by_dfs(onnx_graph):
    # 创建节点名称到节点的映射
    name_to_node = {node.name: node for node in onnx_graph.nodes}
    
    # 构建邻接表和入度表
    in_degree = {node.name: 0 for node in onnx_graph.nodes}
    adj = {node.name: [] for node in onnx_graph.nodes}
    
    # 填充邻接表和入度表
    for node in onnx_graph.nodes:
        for output in node.outputs:
            for other_node in onnx_graph.nodes:
                if output in other_node.inputs:
                    adj[node.name].append(other_node.name)
                    in_degree[other_node.name] += 1
    
    # 初始化队列 (入度为0的节点)
    queue = deque([node.name for node in onnx_graph.nodes if in_degree[node.name] == 0][::-1])
    sorted_nodes = []

    while queue:
        current = queue.pop()
        sorted_nodes.append(name_to_node[current])

        for neighbor in adj[current][::-1]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return sorted_nodes

# 没有考虑相同输入的并列节点的顺序问题 导致的dla加载异常
def alternate_topological_sort_dfs(onnx_graph, gpu_node_name_list):
    # 创建节点名称到节点的映射
    name_to_node = {node.name: node for node in onnx_graph.nodes}
    
    # 构建邻接表和入度表
    in_degree = {node.name: 0 for node in onnx_graph.nodes}
    adj = {node.name: [] for node in onnx_graph.nodes}
    
    # 填充邻接表和入度表
    for node in onnx_graph.nodes:
        for output in node.outputs:
            for other_node in onnx_graph.nodes:
                if output in other_node.inputs:
                    adj[node.name].append(other_node.name)
                    in_degree[other_node.name] += 1
    
    # 初始化队列 (入度为0的节点)
    dla_deque = deque([node.name for node in onnx_graph.nodes if in_degree[node.name] == 0 and node.name not in gpu_node_name_list])
    gpu_deque = deque([node.name for node in onnx_graph.nodes if in_degree[node.name] == 0 and node.name in gpu_node_name_list])
    sorted_nodes = []

    while dla_deque or gpu_deque:
        while dla_deque:
            current = dla_deque.pop()
            sorted_nodes.append(name_to_node[current])

            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in gpu_node_name_list:
                    dla_deque.append(neighbor)
                elif in_degree[neighbor] == 0 and neighbor in gpu_node_name_list:
                    gpu_deque.append(neighbor)

        while gpu_deque:
            current = gpu_deque.pop()
            sorted_nodes.append(name_to_node[current])

            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in gpu_node_name_list:
                    dla_deque.append(neighbor)
                elif in_degree[neighbor] == 0 and neighbor in gpu_node_name_list:
                    gpu_deque.append(neighbor)
    return sorted_nodes

def alternate_topological_sort_by_wfs(onnx_graph, gpu_node_name_list):
    # 创建节点名称到节点的映射
    name_to_node = {node.name: node for node in onnx_graph.nodes}
    
    # 构建邻接表和入度表
    in_degree = {node.name: 0 for node in onnx_graph.nodes}
    adj = {node.name: [] for node in onnx_graph.nodes}
    
    # 填充邻接表和入度表
    for node in onnx_graph.nodes:
        for output in node.outputs:
            for other_node in onnx_graph.nodes:
                if output in other_node.inputs:
                    adj[node.name].append(other_node.name)
                    in_degree[other_node.name] += 1
    
    # 初始化队列 (入度为0的节点)
    dla_deque = deque([node.name for node in onnx_graph.nodes if in_degree[node.name] == 0 and node.name not in gpu_node_name_list])
    gpu_deque = deque([node.name for node in onnx_graph.nodes if in_degree[node.name] == 0 and node.name in gpu_node_name_list])
    sorted_nodes = []

    while dla_deque or gpu_deque:
        while gpu_deque:
            current = gpu_deque.popleft()
            sorted_nodes.append(name_to_node[current])

            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in gpu_node_name_list:
                    dla_deque.append(neighbor)
                elif in_degree[neighbor] == 0 and neighbor in gpu_node_name_list:
                    gpu_deque.append(neighbor)

        while dla_deque:
            current = dla_deque.popleft()
            sorted_nodes.append(name_to_node[current])

            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in gpu_node_name_list:
                    dla_deque.append(neighbor)
                elif in_degree[neighbor] == 0 and neighbor in gpu_node_name_list:
                    gpu_deque.append(neighbor)

    return sorted_nodes

def alternate_topological_sort_by_dfs(onnx_graph, gpu_node_name_list):
    # 创建节点名称到节点的映射
    name_to_node = {node.name: node for node in onnx_graph.nodes}
    
    # 构建邻接表和入度表
    in_degree = {node.name: 0 for node in onnx_graph.nodes}
    adj = {node.name: [] for node in onnx_graph.nodes}
    
    # 填充邻接表和入度表
    for node in onnx_graph.nodes:
        for output in node.outputs:
            for other_node in onnx_graph.nodes:
                if output in other_node.inputs:
                    adj[node.name].append(other_node.name)
                    in_degree[other_node.name] += 1
    
    # 初始化队列 (入度为0的节点)
    dla_list = [node.name for node in onnx_graph.nodes if in_degree[node.name] == 0 and node.name not in gpu_node_name_list]
    dla_deque = deque(dla_list[::-1])
    gpu_deque = deque([node.name for node in onnx_graph.nodes if in_degree[node.name] == 0 and node.name in gpu_node_name_list])
    sorted_nodes = []

    while dla_deque or gpu_deque:

        while gpu_deque:
            current = gpu_deque.pop()
            sorted_nodes.append(name_to_node[current])

            for neighbor in adj[current][::-1]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in gpu_node_name_list:
                    dla_deque.append(neighbor)
                elif in_degree[neighbor] == 0 and neighbor in gpu_node_name_list:
                    gpu_deque.append(neighbor)
        
        dla_deque = split_dla_deque_by_batch_num(dla_deque, name_to_node)
        while dla_deque:
            current = dla_deque.pop()
            sorted_nodes.append(name_to_node[current])

            for neighbor in adj[current][::-1]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in gpu_node_name_list:
                    dla_deque.append(neighbor)
                elif in_degree[neighbor] == 0 and neighbor in gpu_node_name_list:
                    gpu_deque.append(neighbor)
    return sorted_nodes

def alternate_topological_sort_by_dfs_v2(onnx_graph, gpu_node_name_list):
    # 创建节点名称到节点的映射
    name_to_node = {node.name: node for node in onnx_graph.nodes}
    
    # 构建邻接表和入度表
    in_degree = {node.name: 0 for node in onnx_graph.nodes}
    adj = {node.name: [] for node in onnx_graph.nodes}
    
    # 填充邻接表和入度表
    for node in onnx_graph.nodes:
        for output in node.outputs:
            for other_node in onnx_graph.nodes:
                if output in other_node.inputs:
                    adj[node.name].append(other_node.name)
                    in_degree[other_node.name] += 1
    
    # 初始化队列 (入度为0的节点)
    dla_list = [node.name for node in onnx_graph.nodes if in_degree[node.name] == 0 and node.name not in gpu_node_name_list]
    dla_deque = deque(dla_list[::-1])
    gpu_deque = deque([node.name for node in onnx_graph.nodes if in_degree[node.name] == 0 and node.name in gpu_node_name_list])
    sorted_nodes = []

    
    while dla_deque or gpu_deque:

        while gpu_deque:
            current = gpu_deque.pop()
            sorted_nodes.append(name_to_node[current])

            for neighbor in adj[current][::-1]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in gpu_node_name_list:
                    dla_deque.append(neighbor)
                elif in_degree[neighbor] == 0 and neighbor in gpu_node_name_list:
                    gpu_deque.append(neighbor)
        
        dla_deque = split_dla_deque_by_batch_num(dla_deque, name_to_node)
        while dla_deque:
            current = dla_deque.pop()
            sorted_nodes.append(name_to_node[current])

            for neighbor in adj[current][::-1]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in gpu_node_name_list:
                    dla_deque.append(neighbor)
                elif in_degree[neighbor] == 0 and neighbor in gpu_node_name_list:
                    gpu_deque.append(neighbor)
    return sorted_nodes



def split_dla_deque_by_batch_num(dla_deque, name_to_node):
    new_dla_deque = sorted(dla_deque, key=lambda x: name_to_node[x].inputs[0].shape[0])
    return new_dla_deque


def all_integers(lst):
    for x in lst:
        try:
            if float(x) != int(x):
                return False
        except (ValueError, TypeError):
            return False
    return True

def get_gpu_name_list(onnx_path, qdq_scale_name_list):
    gpu_name_list = []
    # original_gpu_name_list =[]
    original_gpu_name_list = ['/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Concat',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/conv_gate/conv_gate.0/Conv',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/sigmoid/Sigmoid',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Split_slice1',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Split_slice2',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Mul',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Concat_1',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Sub',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Mul_1',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/conv_can/conv_can.0/Conv',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/tanh/Tanh',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Mul_2',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Add',

                            '/TRAFFICLANE_2D/head1/keypoint/kp_hm/kp_hm.3/Sigmoid',
                            '/TRAFFICLANE_2D/head1/keypoint/kp_hm/kp_hm.3/MaxPool',
                            '/TRAFFICLANE_2D/head1/keypoint/kp_hm/kp_hm.3/Equal',
                            '/TRAFFICLANE_2D/head1/keypoint/kp_hm/kp_hm.3/Cast',
                            '/TRAFFICLANE_2D/head1/keypoint/kp_hm/kp_hm.3/Mul',

                            '/neck11/temporal_layer/cell_list_0.0/conv_down_channel/Conv',
                            '/neck11/temporal_layer/cell_list_0.0/Concat',
                            '/neck11/temporal_layer/cell_list_0.0/conv_gate/Conv',
                            '/neck11/temporal_layer/cell_list_0.0/Split_slice1',
                            '/neck11/temporal_layer/cell_list_0.0/Split_slice2',
                            '/neck11/temporal_layer/cell_list_0.0/sigmoid_1/Sigmoid',
                            '/neck11/temporal_layer/cell_list_0.0/Mul',
                            '/neck11/temporal_layer/cell_list_0.0/Concat_1',
                            '/neck11/temporal_layer/cell_list_0.0/conv_can/Conv',
                            '/neck11/temporal_layer/cell_list_0.0/tanh/Tanh',
                            '/neck11/temporal_layer/cell_list_0.0/sigmoid/Sigmoid',
                            '/neck11/temporal_layer/cell_list_0.0/Sub',
                            '/neck11/temporal_layer/cell_list_0.0/Mul_1',
                            '/neck11/temporal_layer/cell_list_0.0/Mul_2',
                            '/neck11/temporal_layer/cell_list_0.0/Add',
                            '/neck11/temporal_layer/cell_list_0.0/conv_adjust_channel/Conv',
                            '/neck11/Concat',
                            '/MONO_OD/head1/head0/light_convs/conv/Conv',
                            '/MONO_OD/head1/head0/light_convs/relu/Relu',
                            '/MONO_OD/head1/head0/light_predconvs_stop/conv/Conv',
                            '/MONO_OD/head1/head0/light_predconvs_stop/relu/Relu',
                            '/MONO_OD/head1/head0/sequence_module/cell_list_0.0/Add',

                            '/neck0/Concat_15',
                            '/neck0/Concat_11',
                            '/neck0/Concat_7',
                            '/neck0/Concat_3',
                            ]
    cant_on_dla_op_list = ['Transpose', 'Reshape'] # 'Resize',]
    conditional_dla_op_list = ['Slice', 'Resize', 'Scale', 'Shuffle', 'Softmax']
    dla_op_list = ['Relu', 'Sigmoid', 'Tanh', 'Leany_Relu', 'Clipped_Relu', 
                   'Equal', 'Greater', 'Less', 'Sum', 'Sub', 'Product', 'Max', 'Min', 'Div', 'Pow', 'Reduce', 
                   'Concat', 'Conv', 'ConvTranspose', 'Add', ] # 
    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    for node in graph.nodes:
        if node.name in original_gpu_name_list:
            gpu_name_list.append(node.name)
        elif node.inputs[0].name not in qdq_scale_name_list or node.outputs[0].name not in qdq_scale_name_list:
            gpu_name_list.append(node.name)
        elif node.op == 'Resize':
            if len(node.inputs) == 3:
                scales = node.inputs[2].values
                if not all_integers(scales):
                    gpu_name_list.append(node.name)
            elif len(node.inputs) == 4:
                scales =  node.inputs[3].values / node.inputs[0].shape
                if not all_integers(scales):
                    gpu_name_list.append(node.name)
        elif node.op in cant_on_dla_op_list:
            gpu_name_list.append(node.name)
        elif node.op in dla_op_list:
            pass
        else:
            gpu_name_list.append(node.name)
            print(node.name, node.op)
    return gpu_name_list

def get_calib_name_list(calib_path):
    with open(calib_path) as T:
        lines = T.readlines()
    qdq_scales_name_list = []
    for line in lines[1:]:
        qdq_scales_name_list.append(line.split(': ')[0])
    return qdq_scales_name_list

def write_necessary_gpu_layers(onnx_path, graph, gpu_node_name_list):
    cant_on_dla_op_list = ['Transpose', 'Reshape', 'Resize', 'Pad']
    dla_op_list = ['Relu', 'Sigmoid', 'Tanh', 'Leany_Relu', 'Clipped_Relu', 
                   'Equal', 'Greater', 'Less', 'Sum', 'Sub', 'Product', 'Max', 'Min', 'Div', 'Pow', 'Reduce', 
                   'Concat', 'Conv', 'ConvTranspose', 'Add', ] 
    name_to_node = {node.name: node for node in graph.nodes}
    T = open(onnx_path[:-4]+'necessary_gpu_layers', 'w')
    for name in gpu_node_name_list:
        if name_to_node[name].op in dla_op_list:
            T.write(name+':GPU,')
        elif name_to_node[name].op in cant_on_dla_op_list:
            pass
        elif name_to_node[name].op == 'Slice':
            # import pdb
            # pdb.set_trace()
            if name_to_node[name].inputs[3].shape == [1] and name_to_node[name].inputs[3].values[0] != 0:
                T.write(name+':GPU,')
            else:
                pass
        else:
            T.write(name+':GPU,')
    T.close()

def write_all_gpu_layers(onnx_path, gpu_node_name_list):
    T = open(onnx_path[:-4]+'all_gpu_layers', 'w')
    for name in gpu_node_name_list:
        T.write(name+':GPU,')
    T.close()



onnx_path = '/home/zee001-w/公共的/V8-model/V7.26/2d_V7.26部署_dla融合方案/_all_V8.0_2D_one_model_int8_seq_addm1depth_sim_opt_plugin.processed_rmQDQ.backbone+dynamic.onnx'
calib_path = '/home/zee001-w/公共的/V8-model/V7.26/2d_V7.26部署_dla融合方案/_all_V8.0_2D_one_model_int8_seq_addm1depth_sim_opt_plugin.processed_propagation_calib_new'

# adj, all_nodes = build_graph_from_onnx(onnx_path)
model = onnx.load(onnx_path)
graph = gs.import_onnx(model)
# for node in graph.nodes:print(node.name)
# input()

# # 广度优先的拓扑排序
# subgraph = graph.copy()
# sorted_nodes = topological_sort_by_wfs(subgraph)
# assert len(sorted_nodes) == len(graph.nodes)
# subgraph.nodes = sorted_nodes
# sub_model = gs.export_onnx(subgraph)
# sub_model_name = onnx_path[:-4] + 'toposort_by_wfs.onnx'
# onnx.save(sub_model, sub_model_name)
# print("save model: ", sub_model_name)
# print("节点数量: ", len(sub_model.graph.node))

# 深度优先的拓扑排序
subgraph = graph.copy()
sorted_nodes = topological_sort_by_dfs(subgraph)
# for node in sorted_nodes:print(node.name)
# input()
assert len(sorted_nodes) == len(graph.nodes)
subgraph.nodes = sorted_nodes
sub_model = gs.export_onnx(subgraph)
sub_model_name = onnx_path[:-4] + 'toposort_by_dfs.onnx'
onnx.save(sub_model, sub_model_name)
print("save model: ", sub_model_name)
print("节点数量: ", len(sub_model.graph.node))



# 自动生成上GPU的节点列表
qdq_scale_name_list = get_calib_name_list(calib_path)
gpu_node_name_list = get_gpu_name_list(onnx_path, qdq_scale_name_list)
write_necessary_gpu_layers(onnx_path, graph, gpu_node_name_list)
write_all_gpu_layers(onnx_path, gpu_node_name_list)
# print(gpu_node_name_list, len(gpu_node_name_list))

# # 基于广度的分组拓扑排序
# subgraph = graph.copy()
# sorted_nodes = alternate_topological_sort_by_wfs(subgraph, gpu_node_name_list)
# assert len(sorted_nodes) == len(graph.nodes)
# subgraph.nodes = sorted_nodes
# sub_model = gs.export_onnx(subgraph)
# sub_model_name = onnx_path[:-4] + 'alternate_toposort_by_wfs.onnx'
# onnx.save(sub_model, sub_model_name)
# print("save model: ", sub_model_name)
# print("节点数量: ", len(sub_model.graph.node))



# 基于深度的分组拓扑排序
subgraph = graph.copy()
sorted_nodes = alternate_topological_sort_by_dfs(subgraph, gpu_node_name_list)
# for node in sorted_nodes:print(node.name)
# input()
assert len(sorted_nodes) == len(graph.nodes)
subgraph.nodes = sorted_nodes
sub_model = gs.export_onnx(subgraph)
sub_model_name = onnx_path[:-4] + 'alternate_toposort_by_dfs.onnx'
onnx.save(sub_model, sub_model_name)
print("save model: ", sub_model_name)
print("节点数量: ", len(sub_model.graph.node))



# assert len(sorted_nodes) == len(graph.nodes)
# print('node number asserted!')
# graph.nodes = sorted_nodes
# # for node in sorted_nodes:print(node.name)
# sub_model = gs.export_onnx(graph)
# sub_model_name = onnx_path[:-4] + 'sorted_conditional_dfs+topo_nixu.onnx'
# onnx.save(sub_model, sub_model_name)
# print("save model: ", sub_model_name)
# print("节点数量: ", len(sub_model.graph.node))