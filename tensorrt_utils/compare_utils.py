# +
import onnxruntime
import onnx
import numpy as np
import copy
import sys
import re

sys.path.append('/media/Projects/ModelToolbox/onnxToolbox')
sys.path.append('/media/Projects/MATool/utils')
sys.path.append('/media/Projects/MATool')

from onnx_add_QDQ import *
from process_onnx import onnx2trt
from trex import *

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_svg import FigureCanvasSVG

import networkx as nx

def print_summary(trt_path, end = 'trt_profile'):
    base = trt_path.split('.')[0]
    engine_name = base + '.' + end + '/' + base.split('/')[-1]
    plan = EnginePlan(f'{engine_name}.trt.graph.json', 
                  f'{engine_name}.trt.profile.json', 
                  f'{engine_name}.trt.profile.metadata.json')
    print("ModelName:", trt_path)
    print("Throughput: \t", plan.performance_summary['Throughput'])
    print("Latency: \t",plan.performance_summary['Latency'][2])
    
def get_engine_plan(trt_path, end = 'trt_profile'):
    base = trt_path.split('.')[0]
    engine_name = base + '.' + end + '/' + base.split('/')[-1]
    plan = EnginePlan(f'{engine_name}.trt.graph.json', 
                  f'{engine_name}.trt.profile.json', 
                  f'{engine_name}.trt.profile.metadata.json')
#     print(engine_name)
    return plan

# Set a color for each layer type.
colormap = defaultdict(lambda: 'gray', {
    # https://htmlcolorcodes.com/
    "Convolution":    "#4682B4", # SteelBlue
    "Conv":           "#4682B4",
    "Deconvolution":  "#7B68EE", # MediumSlateBlue
    "CaskDeconvolutionV2":  "#7B68EE",
    "ConvTranspose":  "#7B68EE",
    "ConvActPool":    "#6495ED", # CornflowerBlue
    "MatrixMultiply": "#1E90FF", # DodgerBlue
    "Reformat":       "#00FFFF", # Cyan
    "Reshape":        "#00FFFF",
    "Concat":         "#00FFFF",
    "Shuffle":        "#BC8F8F", # RosyBrown
    "Slice":          "#FFA500", # Orange
    "Scale":          "#8FBC8B", # DarkSeaGreen
    "Quantize":       "#6B8E23", # OliveDrab
    "Pooling":        "#3CB371", # MediumSeaGreen
    "PluginV2":       "#C71585", # MediumVioletRed
    "PointWise":      "#9ACD32", # YellowGreen
    "Add":            "#9ACD32",
    "ElementWise":    "#9ACD32", # YellowGreen
    "Relu":           "#9ACD32",
    "Reduce":         "#90EE90", # LightGreen
    "SoftMax":        "#DA70D6", # Orchid
    "Myelin":         "#800080", # Purple
})

# {'Reshape', 'Concat', 'Conv', 'Add', 'ConvTranspose', 'Relu'}

def get_map(model, plan):
    onnx_map = []
    trt_map_list ={}
    trt_map = {}
    for index1, node in enumerate(model.graph.node):
        index2 = []
        for i, j in plan.df['Name'].items():
            name_cadi = re.split(r'\+|\ |\(|\)|\,', j)
            if node.name in name_cadi:
                if plan.df['type'][i] != 'Reformat':
                    index2.append(i)
        if len(index2)==1:
            onnx_map.append({'Name':node.name, 
                              'id0':index1 ,
                              'id1': index2[0]})
            if index2[0] not in trt_map_list:
                trt_map_list[index2[0]] = [index1]
            else:
                trt_map_list[index2[0]].append(index1)
        elif len(index2)>1:
            raise ValueError(f'算子定位失败:{node.name}')
    for key in trt_map_list:
        tmp = (len(trt_map_list[key]) + 0.01 - 1)/2
        trt_map[key] = trt_map_list[key][round(tmp)]
    return onnx_map, trt_map

def print_text(name, n=30):
    return '\n'.join(name[i:i+n] for i in range(0, len(name), n))

def trt_vs_onnx(onnx, 
                 trt_plan1=None, trt_plan2=None, 
                 max_height=250, save_name='test.svg'):
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(30,max_height))
    plt.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.0)

    # 去除边框
#     ax.set_frame_on(False)

    # 去除坐标轴
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.set_xlim(-15, 15)
    ax.set_ylim(-max_height, 0)
#     
    h_list_1 = []
    h_list_2 = []
    h_list_3 = []
    
    gap = 0.1
    
    # 添加onnx方块
    x = -2.0
    y = 0.0
    width = 4.0  # 方块的宽度
    height_c = max_height / len(onnx.graph.node)  # 方块的高度
    for node in onnx.graph.node:
        rectangle = Rectangle((x, y - height_c * gap / 2), 
                              width, - (height_c * (1 - gap)), 
                              edgecolor='none', facecolor=colormap[node.op_type])
        ax.add_patch(rectangle)
        ax.text(x+width/2, y-height_c/2, print_text(node.name), 
                ha='center', va='center', color='white', fontsize=10)
        h_list_1.append(y)
        y -= height_c
    
    if trt_plan1:
        onnx_MAP, trt_MAP = get_map(onnx, trt_plan1)
        min_lat = min(trt_plan1.df.loc[list(trt_MAP.keys())]['latency.avg_time'])
        max_lat = max(trt_plan1.df.loc[list(trt_MAP.keys())]['latency.avg_time'])
    if trt_plan2:
        onnx_MAP2, trt_MAP2 = get_map(onnx, trt_plan2)
        min_lat2 = min(trt_plan2.df.loc[list(trt_MAP2.keys())]['latency.avg_time'])
        max_lat2 = max(trt_plan2.df.loc[list(trt_MAP2.keys())]['latency.avg_time'])
        min_lat = min(min_lat,min_lat2)
        max_lat = max(max_lat,max_lat2)
    
    if trt_plan1:
        # 添加trt 匹配 方块
        x = 4.0
        y = 0.0
        width = 4.0  
        height = height_c
        for index, node in trt_plan1.df.iterrows():
            if index in trt_MAP:
                y = h_list_1[trt_MAP[index]]
                op_type = trt_plan1.df.loc[index]['type']
    #             height = max_height * node['latency.pct_time'] / 100
                rectangle = Rectangle((x, y - height_c * gap / 2), 
                                      width, -(height - height_c * gap), 
                                      edgecolor='none', facecolor=colormap[op_type])
                ax.add_patch(rectangle)
                ax.text(x+width/2, y-height/2, print_text(node['Name']), 
                        ha='center', va='center', color='white', fontsize=12)
            h_list_3.append(y-height/2)


        def linear_map(value, a, b, c, d):
            # 确保 a 不等于 b，以避免除以零错误
            if a == b:
                b = a + 1
                # raise ValueError("Invalid input: a must be different from b.")

            # 计算线性映射
            mapped_value = c + (value - a) * (d - c) / (b - a)

            # 确保映射后的值在 [c, d] 范围内
            return max(min(mapped_value, d), c)

        # 添加onnx -> trt1 连接线

        x1 = 2.05
        x2 = 3.95
        for index, node in enumerate(onnx_MAP):
            op_type = onnx.graph.node[node['id0']].op_type
            y1 = h_list_1[node['id0']] - height_c / 2 
            y2 = h_list_1[trt_MAP[node['id1']]] - height_c / 2 
            ax.plot([x1, x2], [y1, y2], 
                    color=colormap[op_type], linestyle='-')

        # 添加trt 匹配 延迟方块
        x = 8.3
        y = 0.0
        width = 4.0  
        height = height_c
        min_wid = 0.01
        max_wid = 6
        min_lat = min(trt_plan1.df.loc[list(trt_MAP.keys())]['latency.avg_time'])
        max_lat = max(trt_plan1.df.loc[list(trt_MAP.keys())]['latency.avg_time'])
        for index, node in trt_plan1.df.iterrows():
            if index in trt_MAP:
                y = h_list_1[trt_MAP[index]]
                op_type = trt_plan1.df.loc[index]['type']
                lat = trt_plan1.df.loc[index]['latency.avg_time']
                width = linear_map(lat, 
                                 min_lat, max_lat,
                                 min_wid, max_wid)
    #             height = max_height * node['latency.pct_time'] / 100
                rectangle = Rectangle((x, y - height_c * gap / 2), 
                                      width, -(height - height_c * gap), 
                                      edgecolor='none', facecolor=colormap[op_type])
                ax.add_patch(rectangle)
                ax.text(x + width + 0.05, y-height/2, "{:.4f}".format(lat), 
                        ha='left', va='center', color='black', fontsize=12)
        
    if trt_plan2:

        # 添加trt2 匹配 方块
        x = -4.0
        y = 0.0
        width = -4.0  
        height = height_c
        for index, node in trt_plan2.df.iterrows():
            if index in trt_MAP2:
                y = h_list_1[trt_MAP2[index]]
                op_type = trt_plan2.df.loc[index]['type']
                rectangle = Rectangle((x, y - height_c * gap / 2), 
                                      width, -(height - height_c * gap), 
                                      edgecolor='none', facecolor=colormap[op_type])
                ax.add_patch(rectangle)
                ax.text(x+width/2, y-height/2, print_text(node['Name']), 
                        ha='center', va='center', color='white', fontsize=12)

        # 添加onnx -> trt2 连接线
        x1 = -2.05
        x2 = -3.95
        for index, node in enumerate(onnx_MAP2):
            random_color = np.random.rand(3,)
            op_type = onnx.graph.node[node['id0']].op_type
            y1 = h_list_1[node['id0']] - height_c / 2 
            y2 = h_list_1[trt_MAP2[node['id1']]] - height_c / 2 
            ax.plot([x1, x2], [y1, y2], 
                    color=colormap[op_type], linestyle='-')
            
        # 添加trt2 匹配 延迟方块
        x = -8.3
        y = 0.0 
        height = height_c
        min_wid = 0.01
        max_wid = 6
        
        for index, node in trt_plan2.df.iterrows():
            if index in trt_MAP2:
                y = h_list_1[trt_MAP2[index]]
                random_color = np.random.rand(3,)
                op_type = trt_plan2.df.loc[index]['type']
                lat = trt_plan2.df.loc[index]['latency.avg_time']
                width = -linear_map(lat, 
                                 min_lat, max_lat,
                                 min_wid, max_wid)
    #             height = max_height * node['latency.pct_time'] / 100
                rectangle = Rectangle((x, y - height_c * gap / 2), 
                                      width, -(height - height_c * gap), 
                                      edgecolor='none', facecolor=colormap[op_type])
                ax.add_patch(rectangle)
                ax.text(x + width - 0.05, y-height/2, "{:.4f}".format(lat), 
                        ha='right', va='center', color='black', fontsize=12)


#         plt.show()
        canvas = FigureCanvasSVG(fig)
        canvas.print_svg(save_name)
    
    

### onnx trt node name match
# onnx_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub1/sub1.onnx"
# trt_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub1/sub1.trt"
# trt_path2 = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub1/sub1_t2.trt"

# model = onnx.load(onnx_path)
# plan1 = get_engine_plan(trt_path)
# plan2 = get_engine_plan(trt_path2)
# trt_vs_onnx(model, plan1, plan2)

def build_subgraphs(onnx_MAP1, onnx_MAP2):
#     onnx_MAP1, _ = get_map(onnx, trt_plan1)
#     onnx_MAP2, _ = get_map(onnx, trt_plan2)
    
    G = nx.Graph()
    
    for edge in onnx_MAP1:
        G.add_node(f"onnx_{edge['id0']}", model=0, ind=edge['id0'])
        G.add_node(f"trt1_{edge['id1']}", model=1, ind=edge['id1'])
        G.add_edge(f"onnx_{edge['id0']}", f"trt1_{edge['id1']}")
        
    for edge in onnx_MAP2:
        G.add_node(f"onnx_{edge['id0']}", model=0, ind=edge['id0'])
        G.add_node(f"trt2_{edge['id1']}", model=2, ind=edge['id1'])
        G.add_edge(f"onnx_{edge['id0']}", f"trt2_{edge['id1']}")
    
    connected_components = list(nx.connected_components(G))
    
    cc3s = []
    for cc in connected_components:
        cc3 = [[],[],[],[0.0]]
        for node in cc:
            cc3[G.nodes[node]['model']].append(G.nodes[node]['ind'])
        cc3[0].sort()
        cc3[1].sort()
        cc3[2].sort()
        cc3s.append(cc3)
    return cc3s

def linear_map(value, a, b, c, d):
    # 确保 a 不等于 b，以避免除以零错误
    if a == b:
        b = a + 1
        # raise ValueError("Invalid input: a must be different from b.")

    # 计算线性映射
    mapped_value = c + (value - a) * (d - c) / (b - a)

    # 确保映射后的值在 [c, d] 范围内
    return max(min(mapped_value, d), c)

def trt_vs_onnx_draw_subgraph(ax, subgraph, onnx, 
                 trt_plan1=None, trt_plan2=None,
                 x_0=0.0, y_0=0.0, min_lat=0.0, max_lat=1.0, diff=0.0):
    # 添加onnx方块
    gap = 0.1
    width = 4.0  # 方块的宽度
    max_height = 250
    height_c = max_height / len(onnx.graph.node)  # 方块的高度
#     height_c = 0.6
    min_wid = 0.01
    max_wid = 6
    
    ## add onnx node
    
    y = y_0
    x = x_0
    for ind in subgraph[0]:
        node = onnx.graph.node[ind]
        rectangle = Rectangle((x, y - height_c * gap / 2), 
                              width, - (height_c * (1 - gap)), 
                              edgecolor='none', facecolor=colormap[node.op_type])
        ax.add_patch(rectangle)
        ax.text(x+width/2, y-height_c/2, print_text(node.name), 
                ha='center', va='center', color='white', fontsize=10)
        y -= height_c
    y_new_1 = y
    
    ## add trt 1
    
    h_tmp = (len(subgraph[1]) + len(subgraph[2]) - len(subgraph[0]))/2.0 + 1.0/3
    h_tmp = min(1.0/3, h_tmp)
    y = y_0 + h_tmp * height_c
    y_0_new = max(y_0, y)
    
    x = x_0 + width + 2.0
    
    lat_1 = 0.0
    for index in subgraph[1]:
        lat_1 += trt_plan1.df.loc[index]['latency.avg_time']
    lat_1_w = linear_map(lat_1, 
         min_lat, max_lat,
         min_wid, max_wid)
    lat_2 = 0.0
    for index in subgraph[2]:
        lat_2 += trt_plan2.df.loc[index]['latency.avg_time']
    lat_2_w = linear_map(lat_2, 
         min_lat, max_lat,
         min_wid, max_wid)
    
    for index in subgraph[1]:
        op_type = trt_plan1.df.loc[index]['type']
        name = trt_plan1.df.loc[index]['Name']
        rectangle = Rectangle((x, y - height_c * gap / 2), 
                              width, - (height_c * (1 - gap)), 
                              edgecolor='none', facecolor=colormap[op_type])
        ax.add_patch(rectangle)
        ax.text(x+width/2, y-height_c/2, print_text(name), 
                ha='center', va='center', color='white', fontsize=10)
        y -= height_c

    x_txt_1 = x + width + 0.5
    y_txt_1 = y + height_c
    
    rectangle = Rectangle((x_txt_1, y_txt_1 - height_c * gap / 2), 
                  lat_1_w, - (height_c * (1 - gap)), 
                  edgecolor='none', facecolor='black')
    ax.add_patch(rectangle)
    ax.text(x_txt_1 + lat_1_w + 0.05, y_txt_1 - height_c/2, 
            "{:.4f}".format(lat_1), 
            ha='left', va='center', color='black', fontsize=12)
    y_new_2 = y
    
    ## add trt 2
    y -= height_c *2 / 3
    x = x_0 + width + 2.0
    
    rectangle = Rectangle((x + width + 0.5, y - height_c * gap / 2), 
                  lat_2_w, - (height_c * (1 - gap)), 
                  edgecolor='none', facecolor='red')
    ax.add_patch(rectangle)
    ax.text(x + width + 0.5 + lat_2_w + 0.05, y-height_c/2, 
            "{:.4f}".format(lat_2), 
            ha='left', va='center', color='black', fontsize=12)
    
    for index in subgraph[2]:
        op_type = trt_plan2.df.loc[index]['type']
        name = trt_plan2.df.loc[index]['Name']
        rectangle = Rectangle((x, y - height_c * gap / 2), 
                              width, - (height_c * (1 - gap)), 
                              edgecolor='none', facecolor=colormap[op_type])
        ax.add_patch(rectangle)
        ax.text(x+width/2, y-height_c/2, print_text(name), 
                ha='center', va='center', color='white', fontsize=10)
        y -= height_c 
    y_new_3 = y
    y_new = min(y_new_1, y_new_2, y_new_3) 
    
    
    rectangle = Rectangle((x_0 - 1/3, y_0_new), 
                        19, 
                        y_new - y_0_new, 
                        edgecolor='grey', facecolor='none')
    ax.add_patch(rectangle)
    
    ## add diff
    diff += lat_2 - lat_1
    ax.text(x_txt_1 + 8, y_txt_1 - 1/4 * height_c, "Diff: {:.4f}".format(lat_2 - lat_1), 
            ha='right', va='top', 
            color='black', fontsize=12,
            fontdict={'family': 'sans-serif', 'weight': 'bold'})
    ax.text(x_txt_1 + 8, y_txt_1 - 1/2 * height_c, "Cumulative Diff: {:.4f}".format(diff), 
            ha='right', va='top', 
            color='black', fontsize=12,
            fontdict={'family': 'sans-serif', 'weight': 'bold'})
    return y_new - height_c, diff

def trt_vs_onnx_draw_subgraphs(onnx, 
                 trt_plan1=None, trt_plan2=None, 
                 max_height=250, save_name='test2.svg'):
    # 创建一个新的图形
    fig, ax = plt.subplots(figsize=(20,max_height))
    plt.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.0)

    # 去除边框
#     ax.set_frame_on(False)

    # 去除坐标轴
    ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

    
    onnx_MAP1, trt_MAP1 = get_map(onnx, trt_plan1)
    onnx_MAP2, trt_MAP2 = get_map(onnx, trt_plan2)
    min_lat = min(min(trt_plan1.df.loc[list(trt_MAP1.keys())]['latency.avg_time']),
                  min(trt_plan2.df.loc[list(trt_MAP2.keys())]['latency.avg_time']))
    max_lat = max(max(trt_plan1.df.loc[list(trt_MAP1.keys())]['latency.avg_time']),
                  max(trt_plan2.df.loc[list(trt_MAP2.keys())]['latency.avg_time']))
    lat_count1 = sum(trt_plan1.df.loc[list(trt_MAP1.keys())]['latency.avg_time'])
    lat_count2 = sum(trt_plan2.df.loc[list(trt_MAP2.keys())]['latency.avg_time'])
    print("计算在内的延迟1：", lat_count1)
    print("计算在内的延迟2：", lat_count2)
    
    subgraphs = build_subgraphs(onnx_MAP1, onnx_MAP2)
    for subgraph in subgraphs:
        lat = 0.0
        for index in subgraph[1]:
            lat += trt_plan1.df.loc[index]['latency.avg_time']
        for index in subgraph[2]:
            lat -= trt_plan2.df.loc[index]['latency.avg_time']
        subgraph[3] = lat
    subgraphs = sorted(subgraphs, key=lambda x: x[3])
    x = 0.0
    y = 0.0
    diff = 0.0
    for subgraph in subgraphs:
        y, diff = trt_vs_onnx_draw_subgraph(ax, subgraph, onnx, 
                                      trt_plan1, trt_plan2,
                                      x, y, min_lat, max_lat, diff)
        
    ax.set_xlim(-1, 20)
    ax.set_ylim(y, 2)
    
#     plt.show()
    canvas = FigureCanvasSVG(fig)
    canvas.print_svg(save_name)


# # +
# onnx_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub1/sub1.onnx"
# trt_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub1/sub1.trt"
# trt_path2 = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub1/sub1_t2.trt"

# trt_path = "/media/models/fisheye/opt_QDQ/fisheye_orin_test/sub1.trt"
# trt_path2 = "/media/models/fisheye/opt_QDQ/fisheye_orin_test/sub1_QDQ_best.trt"

# print_summary(trt_path)
# print_summary(trt_path2)
# model = onnx.load(onnx_path)
# plan1 = get_engine_plan(trt_path)
# plan2 = get_engine_plan(trt_path2)
# trt_vs_onnx(model, plan1, plan2, save_name='test1.svg')
# trt_vs_onnx_draw_subgraphs(model, plan1, plan2, save_name='test2.svg')
# # -



# # +
# onnx_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub1/sub1.onnx"
# trt_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2/sub2.trt"
# trt_path2 = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2/sub2_QDQ.trt"
# trt_path3 = "/media/models/fish  eye/opt_QDQ/QDQ_opt_sub2/sub2_t9.trt"

# print_summary(trt_path)
# print_summary(trt_path2)
# print_summary(trt_path3)
# # model = onnx.load(onnx_path)
# # plan1 = get_engine_plan(trt_path)
# # plan2 = get_engine_plan(trt_path2)
# # trt_vs_onnx(model, plan1, plan2, save_name='test1.svg')
# # trt_vs_onnx_draw_subgraphs(model, plan1, plan2, save_name='test2.svg')

# # +
# onnx_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2/sub2.onnx"
# trt_dic = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2"
# trt_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2/sub2.trt_profile"
# trt_path2 = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2/sub2_t9_QDQ_best.trt_profile"
# # trt_path3 = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2/sub2_t9.trt"

# print_summary(trt_path)
# print_summary(trt_path2)
# model = onnx.load(onnx_path)
# num = len(model.graph.node)
# plan1 = get_engine_plan(trt_path)
# plan2 = get_engine_plan(trt_path2)
# trt_vs_onnx(model, plan1, plan2, save_name=trt_dic + '/test1.svg', max_height=num)
# trt_vs_onnx_draw_subgraphs(model, plan1, plan2, save_name=trt_dic + '/test2.svg', max_height=num)
# # +
# ## sub2_comparison
# onnx_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2/sub2.onnx"
# trt_dic = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2"
# trt_path = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2/sub2.trt_profile_orin"
# trt_path2 = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2/sub2_t9_QDQ_best.trt_profile"
# trt_path2 = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2_v2/sub2_t11.trt"

# print_summary(trt_path, end = 'trt_profile_orin')
# print_summary(trt_path2)
# model = onnx.load(onnx_path)
# num = len(model.graph.node)
# plan1 = get_engine_plan(trt_path, end = 'trt_profile_orin')
# plan2 = get_engine_plan(trt_path2)
# trt_vs_onnx(model, plan1, plan2, save_name=trt_dic + '/test1.svg', max_height=num)
# trt_vs_onnx_draw_subgraphs(model, plan1, plan2, save_name=trt_dic + '/test2.svg', max_height=num)


# # +
# onnx_path = "/media/models/fisheye/fisheye_20231117_sta_sim.onnx"
# fig_dic = "/media/models/fisheye/opt_QDQ"
# trt_path = "/media/models/fisheye/fisheye_20231117_sta_sim.trt"
# trt_path2 = "/media/models/fisheye/opt_QDQ/QDQ_opt.trt"
# # trt_path3 = "/media/models/fisheye/opt_QDQ/QDQ_opt_sub2/sub2_t9.trt"

# print_summary(trt_path)
# print_summary(trt_path2)
# model = onnx.load(onnx_path)
# num = len(model.graph.node)
# num = 600
# plan1 = get_engine_plan(trt_path)
# plan2 = get_engine_plan(trt_path2)
# trt_vs_onnx(model, plan1, plan2, save_name=fig_dic + '/test1.svg', max_height=num)
# trt_vs_onnx_draw_subgraphs(model, plan1, plan2, save_name=fig_dic + '/test2.svg', max_height=num)
# # -

# print(get_print_name(A, 10))
