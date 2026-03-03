# +
import json

def analys(folder_name):
    with open(f"{folder_name}.trt.profile.json", 'r') as f:
        prof = json.load(f)

    with open(f"{folder_name}.trt.graph.json", 'r') as f:
        graph_json = json.load(f)

    name_lat_dict={}
    for info in prof[1:]:
        name_lat_dict[info['name']] = info['averageMs']
    for layer in graph_json['Layers']:
        layer["lat"] = name_lat_dict[layer['Name']]

    keywords = ['backbone0','backbone3', 'backbone4', 
                    'group0', 'group1','group2','group3', 'group8',
                    "neck10",'neck0','neck1','neck2','neck3','neck4','neck5', 
                    'transformer0',
                    'fuser0',
                    'sequence_static',
                    "bev_encoder",
                    "bev_sdmap_fuser",
#                     "DriveBoundHead",
#                     "GuideLineActionHead",
                    "TurnGuideLineHead",
                    "GuideLineHead",
                    "BevVecHead",
                    "HybridVecHead",
                    'BEV_OD',
                    "OccHead",
                    'OCC_OD', 
                    "LanemarkHead",
#                     "bev_fpn",
#                     "bev_backbone",
#                     "BevSegHead", "BevVecHead", "ArrowHead", "CenterlineHead", "PendingHead",
                    'TRAFFICLANE_BEV',
                    'TRAFFICLANE_2D', 
                    'MONO_OD',
                    'SIDE_OD', 
                    'GATE_LEVER',
                    ]

    lat_dict = {k:0.0 for k in keywords}
    lat_dict['others'] = 0.0
    sum_lat = 0.0
    for layer in graph_json['Layers']:
        matched = False
        for k in keywords:
            if k in layer['Name'] or k in layer['Metadata']:
                lat_dict[k] += layer["lat"]
                sum_lat += layer["lat"]
                matched = True
#                 if k == "sequence":
#                     print(layer["lat"], layer['Name'])
                break
        if not matched:
            lat_dict['others']+= layer["lat"]
            sum_lat += layer["lat"]
#             print("Not matched", layer["lat"], layer['Name'], layer['Metadata'])

    sum_lat = 0.0
    for k in lat_dict.keys():
#         print(f"{k:<40s}  {lat_dict[k]:>5.2f} " )
        print(f"{lat_dict[k]:>5.2f} " )
        sum_lat += lat_dict[k]
    total = "total"
    print(f"{total:<40s}  {sum_lat:<5.2f} ")
    lat_dict['total'] = sum_lat
    return lat_dict

# lat_dict_1 =analys("/media/models/sorin/0422_v6.16/TASK11111111111_V6.9pre_model_int8_seq_addm1depth_sim_opt_plugin.partB.thor.weakly/TASK11111111111_V6.9pre_model_int8_seq_addm1depth_sim_opt_plugin.partB")
# lat_dict_2 = analys("/media/models/sorin/0422_v6.16/TASK11111111111_V6.12pre_model_int8_seq_addm1depth_sim_opt_plugin_seq2single.partB_aarch64_7030.thor.weakly/TASK11111111111_V6.12pre_model_int8_seq_addm1depth_sim_opt_plugin_seq2single.partB_aarch64_7030")

# lat_dict_1 =analys("/media/models/sorin/0422_v6.16/TASK11111111111_V6.9pre_model_int8_seq_addm1depth_sim_opt_plugin.partA.thor.weakly/TASK11111111111_V6.9pre_model_int8_seq_addm1depth_sim_opt_plugin.partA")
# lat_dict_2 = analys("/media/models/sorin/0422_v6.16/TASK11111111111_V6.12pre_model_int8_seq_addm1depth_sim_opt_plugin_seq2single.partA_aarch64_7030.thor.weakly/TASK11111111111_V6.12pre_model_int8_seq_addm1depth_sim_opt_plugin_seq2single.partA_aarch64_7030")

# lat_dict_1 =analys("/media/models/sorin/v6.17_split/TASK11111111111_orinY_sim_int8opt_pluginv2.orin/TASK11111111111_orinY_sim_int8opt_pluginv2")
# lat_dict_2 =analys("/media/models/sorin/v6.26_0512/v6.26_aligned.orin/v6.26_aligned")

# lat_dict_2 = analys("/media/models/sorin/v6.17/TASK11111111111_V6.17_model_int8_seq_addm1depth_sim_int8opt_plugin_seq2single_qdqopt.orin/TASK11111111111_V6.17_model_int8_seq_addm1depth_sim_int8opt_plugin_seq2single_qdqopt")
# lat_dict_2 =analys("/media/models/sorin/0516_v6.36/v6.36.orin/v6.36")
lat_dict_2 =analys("/media/models/sorin/0520_v6.40/UNIFIED_v6.40_Lanemark24m47kLane24m50kCurb6.26Pend50kArrow18kCenter30kLR19kUturn18kBase6.34_sim_int8opt_plugin_qdqopt_1747643872.orin/UNIFIED_v6.40_Lanemark24m47kLane24m50kCurb6.26Pend50kArrow18kCenter30kLR19kUturn18kBase6.34_sim_int8opt_plugin_qdqopt_1747643872")
lat_dict_1 = analys("/media/models/sorin/0528_gaoliang/UNIFIED_v6_42_lane_sim_int8opt_plugin.orin.orin/UNIFIED_v6_42_lane_sim_int8opt_plugin.orin")
print("==============")

for k in lat_dict_1.keys():
    if abs(lat_dict_1[k]-lat_dict_2[k]) > 0.05:
        print(f"{k:<40s}  {lat_dict_2[k]:>5.1f} {lat_dict_1[k]:>5.1f}  {lat_dict_1[k] - lat_dict_2[k]:>5.1f} " )
    else:
        print(f"{k:<40s}  {lat_dict_2[k]:>5.1f} {lat_dict_1[k]:>5.1f}  " )

# +
import json

def analys(folder_name):
    with open(f"{folder_name}.trt.profile.json", 'r') as f:
        prof = json.load(f)

    with open(f"{folder_name}.trt.graph.json", 'r') as f:
        graph_json = json.load(f)

    name_lat_dict={}
    for info in prof[1:]:
        name_lat_dict[info['name']] = info['averageMs']
    for layer in graph_json['Layers']:
        layer["lat"] = name_lat_dict[layer['Name']]

    keywords = ['backbone0',
                    'group0', 'group2','group8',
                    "neck10","neck11",'neck0','neck4', 
                    'TRAFFICLANE_2D',
                    'TRAFFICLIGHT_2D',
                    'MONO_OD',
                    'GATE_LEVER',
                    ]

    lat_dict = {k:0.0 for k in keywords}
    lat_dict['others'] = 0.0
    sum_lat = 0.0
    for layer in graph_json['Layers']:
        matched = False
        for k in keywords:
            if k in layer['Name'] or k in layer['Metadata']:
                lat_dict[k] += layer["lat"]
                sum_lat += layer["lat"]
                matched = True
#                 if k == "sequence":
#                     print(layer["lat"], layer['Name'])
                break
        if not matched:
            lat_dict['others']+= layer["lat"]
            sum_lat += layer["lat"]
#             print("Not matched", layer["lat"], layer['Name'], layer['Metadata'])

    sum_lat = 0.0
    for k in lat_dict.keys():
#         print(f"{k:<40s}  {lat_dict[k]:>5.2f} " )
        print(f"{lat_dict[k]:>5.2f} " )
        sum_lat += lat_dict[k]
    total = "total"
    print(f"{total:<40s}  {sum_lat:<5.2f} ")
    lat_dict['total'] = sum_lat
    return lat_dict

# lat_dict_1 = analys("/media/models/unified_2d/0520/TASK110111110000_V8.0_2D_one_model_b0_int8_seq_addm1depth_sim_opt_plugin.orin/TASK110111110000_V8.0_2D_one_model_b0_int8_seq_addm1depth_sim_opt_plugin")
# lat_dict_2 = analys("/media/models/unified_2d/0520/TASK110111110000_V8.0_2D_one_model_b1_int8_seq_addm1depth_sim_opt_plugin.orin/TASK110111110000_V8.0_2D_one_model_b1_int8_seq_addm1depth_sim_opt_plugin")
# lat_dict_3 = analys("/media/models/unified_2d/0520/TASK110111110000_V8.0_2D_one_model_b1_s64_int8_seq_addm1depth_sim_opt_plugin.orin/TASK110111110000_V8.0_2D_one_model_b1_s64_int8_seq_addm1depth_sim_opt_plugin")

lat_dict_2 =analys("/media/models/unified_2d/0520/UNIFIED_2d_b0.strongly.thor/UNIFIED_2d_b0.strongly")
lat_dict_1 = analys("/media/models/unified_2d/0520/UNIFIED_2d_b0.thor/UNIFIED_2d_b0")

print("==============")

for k in lat_dict_1.keys():
    if abs(lat_dict_1[k]-lat_dict_2[k]) > 0.05:
        print(f"{k:<40s}  {lat_dict_2[k]:>5.1f} {lat_dict_1[k]:>5.1f}  {lat_dict_1[k]-lat_dict_2[k]:>5.1f} " )
    else:
        print(f"{k:<40s}  {lat_dict_2[k]:>5.1f} {lat_dict_1[k]:>5.1f}  " )


# +
# 3d 模型
import json

def analys(folder_name):
    with open(f"{folder_name}.trt.profile.json", 'r') as f:
        prof = json.load(f)

    with open(f"{folder_name}.trt.graph.json", 'r') as f:
        graph_json = json.load(f)

    name_lat_dict={}
    for info in prof[1:]:
        name_lat_dict[info['name']] = info['averageMs']
    for layer in graph_json['Layers']:
        layer["lat"] = name_lat_dict[layer['Name']]

    keywords = ['backbone0','backbone3', 'backbone4', 
                    'group0', 'group1','group2','group3', 'group8',
                    "neck10",'neck0','neck1','neck2','neck3','neck4','neck5', 
                    'transformer0',
                    'fuser0',
                    'sequence_static',
                    "bev_encoder",
                    "bev_sdmap_fuser",
#                     "DriveBoundHead",
#                     "GuideLineActionHead",
                    "TurnGuideLineHead",
                    "GuideLineHead",
                    "BevVecHead",
                    "OccHead",
                    'OCC_OD', 
                    "LanemarkHead",
#                     "bev_fpn",
#                     "bev_backbone",
#                     "BevSegHead", "BevVecHead", "ArrowHead", "CenterlineHead", "PendingHead",
                    'TRAFFICLANE_BEV',
                    'TRAFFICLANE_2D', 
                    'MONO_OD',
                    'SIDE_OD', 
                    'BEV_OD',
                    'GATE_LEVER',
                    ]

    lat_dict = {k:0.0 for k in keywords}
    lat_dict['others'] = 0.0
    sum_lat = 0.0
    for layer in graph_json['Layers']:
        matched = False
        for k in keywords:
            if k in layer['Name'] or k in layer['Metadata']:
                lat_dict[k] += layer["lat"]
                sum_lat += layer["lat"]
                matched = True
#                 if k == "sequence":
#                     print(layer["lat"], layer['Name'])
                break
        if not matched:
            lat_dict['others']+= layer["lat"]
            sum_lat += layer["lat"]
#             print("Not matched", layer["lat"], layer['Name'], layer['Metadata'])

    sum_lat = 0.0
    for k in lat_dict.keys():
#         print(f"{k:<40s}  {lat_dict[k]:>5.2f} " )
        print(f"{lat_dict[k]:>5.2f} " )
        sum_lat += lat_dict[k]
    total = "total"
    print(f"{total:<40s}  {sum_lat:<5.2f} ")
    lat_dict['total'] = sum_lat
    return lat_dict

# lat_dict_1 = analys("/media/models/unified_2d/0520/TASK110111110000_V8.0_2D_one_model_b0_int8_seq_addm1depth_sim_opt_plugin.orin/TASK110111110000_V8.0_2D_one_model_b0_int8_seq_addm1depth_sim_opt_plugin")
# lat_dict_2 = analys("/media/models/unified_2d/0520/TASK110111110000_V8.0_2D_one_model_b1_int8_seq_addm1depth_sim_opt_plugin.orin/TASK110111110000_V8.0_2D_one_model_b1_int8_seq_addm1depth_sim_opt_plugin")
# lat_dict_3 = analys("/media/models/unified_2d/0520/TASK110111110000_V8.0_2D_one_model_b1_s64_int8_seq_addm1depth_sim_opt_plugin.orin/TASK110111110000_V8.0_2D_one_model_b1_s64_int8_seq_addm1depth_sim_opt_plugin")

lat_dict_2 =analys("/media/models/unified_3d/0513_v7_7cam/UNIFIED_3d_7cam.strongly.thor/UNIFIED_3d_7cam.strongly")
lat_dict_1 = analys("/media/models/unified_3d/0513_v7_7cam/UNIFIED_3d_7cam.thor/UNIFIED_3d_7cam")

print("==============")

for k in lat_dict_1.keys():
    if abs(lat_dict_1[k]-lat_dict_2[k]) > 0.05:
        print(f"{k:<40s}  {lat_dict_2[k]:>5.1f} {lat_dict_1[k]:>5.1f}  {lat_dict_1[k]-lat_dict_2[k]:>5.1f} " )
    else:
        print(f"{k:<40s}  {lat_dict_2[k]:>5.1f} {lat_dict_1[k]:>5.1f}  " )

# +
import json

def analys(folder_name):
    with open(f"{folder_name}.trt.profile.json", 'r') as f:
        prof = json.load(f)

    with open(f"{folder_name}.trt.graph.json", 'r') as f:
        graph_json = json.load(f)

    name_lat_dict={}
    for info in prof[1:]:
        name_lat_dict[info['name']] = info['averageMs']
    for layer in graph_json['Layers']:
        layer["lat"] = name_lat_dict[layer['Name']]

    keywords = ['hd_map_encoder',
                'bev_map_fusion', 
                'bev_seg_downsample_aux'
                    ]

    lat_dict = {k:0.0 for k in keywords}
    lat_dict['others'] = 0.0
    sum_lat = 0.0
    for layer in graph_json['Layers']:
        matched = False
        for k in keywords:
            if k in layer['Name'] or k in layer['Metadata']:
                lat_dict[k] += layer["lat"]
                sum_lat += layer["lat"]
                matched = True
#                 if k == "sequence":
#                     print(layer["lat"], layer['Name'])
                break
        if not matched:
            lat_dict['others']+= layer["lat"]
            sum_lat += layer["lat"]
#             print("Not matched", layer["lat"], layer['Name'], layer['Metadata'])

    sum_lat = 0.0
    for k in lat_dict.keys():
        print(f"{k:<40s}  {lat_dict[k]:>5.2f} " )
#         print(f"{lat_dict[k]:>5.2f} " )
        sum_lat += lat_dict[k]
    total = "total"
    print(f"{total:<40s}  {sum_lat:<5.2f} ")
    lat_dict['total'] = sum_lat
    return lat_dict

lat_dict_1 = analys("/media/models/0520_static/20250520-1617__model.trt_profile/model")
# lat_dict_2 = analys("/media/models/unified_2d/0520/TASK110111110000_V8.0_2D_one_model_b1_int8_seq_addm1depth_sim_opt_plugin.orin/TASK110111110000_V8.0_2D_one_model_b1_int8_seq_addm1depth_sim_opt_plugin")
# lat_dict_3 = analys("/media/models/unified_2d/0520/TASK110111110000_V8.0_2D_one_model_b1_s64_int8_seq_addm1depth_sim_opt_plugin.orin/TASK110111110000_V8.0_2D_one_model_b1_s64_int8_seq_addm1depth_sim_opt_plugin")

print("==============")

# for k in lat_dict_1.keys():
#     if abs(lat_dict_1[k]-lat_dict_2[k]) > 0.05:
#         print(f"{k:<40s}  {lat_dict_2[k]:>5.1f} {lat_dict_1[k]:>5.1f}  {lat_dict_1[k]-lat_dict_2[k]:>5.1f} " )
#     else:
#         print(f"{k:<40s}  {lat_dict_2[k]:>5.1f} {lat_dict_1[k]:>5.1f}  " )
for k in lat_dict_1.keys():
    print(f"{k:<40s}  {lat_dict_1[k]:>5.1f}")

# +

import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan = get_plan("/media/Projects/ModelBoard/database/offline_models/fisheye/bevfs_new_model_20231205_orinN_6080.trt_profile")


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

# for i in range(len(plan.df)):
#     for out_name in plan.df.iloc[i]['out_names']:
#         for j in range(len(plan.df)):
#             if out_name in plan.df.iloc[j]['inp_names']:
# #                 print(i, j)
#                 G.add_edge(i, j)
with open('/media/Projects/ModelBoard/database/offline_models/fisheye/bevfs_new_model_20231205_orinN_6080.trt_profile/edge_info.json'
          , 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

edge_to_remove = ['170-171', '191-193', '191-192']
# edge_to_remove = ['170-171']
for edge in edge_to_remove:
    nodes = edge.split('-')
    node1 = int(nodes[0])
    node2 = int(nodes[1])
    print(node1, node2)
    G.remove_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

# +

import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan_dir = "/media/models/sorin/0219_v6.0/v4.0_all_sim_opt_plugin.trt_profile_orin"
plan = get_plan(plan_dir)


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

# for i in range(len(plan.df)):
#     for out_name in plan.df.iloc[i]['out_names']:
#         for j in range(len(plan.df)):
#             if out_name in plan.df.iloc[j]['inp_names']:
# #                 print(i, j)
#                 G.add_edge(i, j)
with open(plan_dir + '/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

# backbone
edge_to_remove = []
for edge in edge_to_remove:
    nodes = edge.split('-')
    node1 = int(nodes[0])
    node2 = int(nodes[1])
    print(node1, node2)
    G.remove_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

keywords = ['backbone0', 'backbone3','backbone4','backbone5', 
            'group0', 'group1','group3', 'group7',
            'neck0','neck1','neck2','neck3','neck4','neck5', 
            'transformer0','transformer1','fuser0','fuser1',
            'sequence',
            'MultiscaleDeformableAttnPlugin_TRT',
            'MONO_OD','MONO3D','STATIC_OD','SIDE_OD','OCC_OD', 'BEV_OD','MONODEPTH',
            "bev_encoder",
            "maptrv2_head",
            'GuideLineHead',
            'TRAFFICLANE_2D', 'TRAFFICLANE_BEV',
            ]


flag = [0] * len(G.nodes)
lats=0.0
for keyword in keywords:
    lat = 0.0
    for i in range(len(G.nodes)):
        if keyword in G.nodes[i]['name'] and flag[i] == 0:
#             print("@@@ ",G.nodes[i]['name'])
            lat += G.nodes[i]['weight']
            flag[i] += 1
    print(f"{keyword:<30s}  {lat:>5.2f}")
    lats += lat
    
other_value = 0
for i in range(len(G.nodes)):
    if flag[i] == 0:
#         if (G.nodes[i]['weight'] >0.2):
        other_value+=G.nodes[i]['weight']
print(f"others: {other_value:.2f}")
lats +=other_value
print("overal:", lats)

# +
import sys
sys.path.append('/media/Projects/MATool')
from trex import *
import networkx as nx
def plan_test(plan_dir, keywords):
    plan = get_plan(plan_dir)

    # 创建有向图对象
    G = nx.DiGraph()
    for i in range(len(plan.df)):
        node_ind = i
    # 添加带有权重的节点
        G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

    with open(plan_dir + '/edge_info.json', 'r') as file:
        edge_to_add = json.load(file)
    for edge in edge_to_add:
        if "X" not in edge:
            nodes = edge.split('-')
            node1 = int(nodes[0])
            node2 = int(nodes[1])
    #         print(node1, node2)
            G.add_edge(node1, node2)

    connected_components = list(nx.weakly_connected_components(G))
    # 计算节点权重的总和
    total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
    print("Total Node Weight:", total_weight)
    print("subgraph num: ", len(connected_components))
#     for cc in connected_components:
#         print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

    flag = [0] * len(G.nodes)
    lats=0.0
    lats_dict = {}
    for keyword in keywords:
        lat = 0.0
        for i in range(len(G.nodes)):
            if keyword in G.nodes[i]['name'] and flag[i] == 0:
    #             print("@@@ ",G.nodes[i]['name'])
                lat += G.nodes[i]['weight']
                flag[i] += 1
#         print(f"{keyword:<30s}  {lat:>5.2f}")
        lats_dict[keyword] = lat
        lats += lat

    other_value = 0
    for i in range(len(G.nodes)):
        if flag[i] == 0:
    #         if (G.nodes[i]['weight'] >0.2):
            other_value+=G.nodes[i]['weight']
            print(f"others {G.nodes[i]['name']}: {G.nodes[i]['weight']:.2f}")
    lats_dict["others"] = other_value
    lats +=other_value
#     print("overal:", lats)
    lats_dict["overal"] = lats
    return lats_dict
keywords = ['backbone0','backbone3', 'backbone4', 
                'group0', 'group1','group2','group3', 'group8',
                "neck10",'neck0','neck1','neck2','neck3','neck4','neck5', 
                'transformer0',
#                 'transformer1',
                'fuser0',
                'sequence',
                'MONO_OD','STATIC_OD','SIDE_OD','OCC_OD', 'BEV_OD', 'GATE_LEVER',
                "bev_encoder",
                "bev_sdmap_fuser",
                 "BevVecHead",
                "GuideLineHead",
                "maptrv2_head",
                'TRAFFICLANE_2D', 
                "OccHead",
                'TRAFFICLANE_BEV',
    #             'seg_static_freespace_head',
    #             'seg_speedbump_head',
    #             'BevSegHead',
    #             'PendingHead',
    #             'ArrowHead',
    #             'CenterlineHead',
                ]
# plan_test(plan_dir="/media/models/sorin/0219_v6.0/unified_model_0215_ptq_orin.trt_profile",
#           keywords=keywords)

# lats_dict1 = plan_test(plan_dir="/media/models/sorin/0226_ptq_dummy_v6.0_test3/TASK11111111_sim_opt_plugin_new.trt_profile.orin",
#           keywords=keywords)
# lats_dict2 = plan_test(plan_dir="/media/models/sorin/0224_v6_ptq_test/v4.0_all_sim_opt_plugin.trt_profile_orin",
#           keywords=keywords)

# lats_dict3 = plan_test(plan_dir="/media/models/sorin/0219_v6.0/TASK11111111_sim.trt_profile",
#           keywords=keywords)
# lats_dict4 = plan_test(plan_dir="/media/models/sorin/0224_v6_ptq_test/v4.2-20250220-ptq_orin.trt_profile",
#           keywords=keywords)

# lats_dict5 = plan_test(plan_dir="/media/models/sorin/0301_ptq_test2/TASK11111111_sim_opt_plugin.trt_profile.orin",
#           keywords=keywords)
# lats_dict6 = plan_test(plan_dir="//media/models/sorin/0303_ptq_test1/TASK11111111_sim_opt_plugin.trt_profile",
#           keywords=keywords)
lats_dict1 = plan_test(plan_dir="/media/models/sorin/0312_ptq_dummy/20250311-1953__dummy_test_orin.trt_profile",
          keywords=keywords)
lats_dict2 = plan_test(plan_dir="/media/models/sorin/0313_ptq_dummy/TASK11111111_sim_opt_plugin.trt_profile.orin",
          keywords=keywords)


# -

for k in lats_dict1.keys():
#     print(f"{k}" )
#     print(f"{lats_dict1[k]:>.2f}" )
    print(f"{k:<40s}  {lats_dict1[k]:>5.1f} {lats_dict2[k]:>5.1f}  {lats_dict2[k]-lats_dict1[k]:>5.1f} " )

# +
keywords = ['backbone0', 'group0', 'group1',
                'neck0','neck5', 
                'transformer0','fuser0',
                'sequence',
                'TRAFFICLANE_BEV',
                ]
# plan_test(plan_dir="/media/models/sorin/0219_v6.0/unified_model_0215_ptq_orin.trt_profile",
#           keywords=keywords)

lats_dict1 = plan_test(plan_dir="/media/models/sorin/0225_latency_test/v6.0_trafficlane_bev.trt_profile",
          keywords=keywords)
# lats_dict2 = plan_test(plan_dir="/media/models/sorin/0224_v6_ptq_test/v4.0_all_sim_opt_plugin.trt_profile_orin",
#           keywords=keywords)

# lats_dict3 = plan_test(plan_dir="/media/models/sorin/0219_v6.0/TASK11111111_sim.trt_profile",
#           keywords=keywords)
lats_dict4 = plan_test(plan_dir="/media/models/sorin/0225_latency_test/v4.0_trafficlane_bev.trt_profile",
          keywords=keywords)
# -

for k in lats_dict1.keys():
#     print(f"{k}" )
#     print(f"{lats_dict1[k]:>.2f}" )
    print(f"{k:<40s}  {lats_dict4[k]:>5.2f} {lats_dict1[k]:>5.2f}  {lats_dict1[k]-lats_dict4[k]:>5.2f} " )

# +
import json
folder_name = "/media/models/sorin/0313_ptq_dummy/0313_onemodel_strongly.trt_profile.thor"

with open(f"{folder_name}/0313_onemodel_strongly.trt.profile.json", 'r') as f:
    prof = json.load(f)

with open(f"{folder_name}/0313_onemodel_strongly.trt.graph.json", 'r') as f:
    graph_json = json.load(f)

name_lat_dict={}
for info in prof[1:]:
    name_lat_dict[info['name']] = info['averageMs']
for layer in graph_json['Layers']:
    layer["lat"] = name_lat_dict[layer['Name']]

keywords = ['backbone0','backbone3', 'backbone4', 
                'group0', 'group1','group2','group3', 'group8',
                "neck10",'neck0','neck1','neck2','neck3','neck4','neck5', 
                'transformer0',
#                 'transformer1',
                'fuser0',
                'sequence',
                'MONO_OD','STATIC_OD','SIDE_OD','OCC_OD', 'BEV_OD', 'GATE_LEVER',
                'TRAFFICLANE_2D', 
                "bev_encoder",
                "bev_sdmap_fuser",
                 "BevVecHead",
                "GuideLineHead",
                "maptrv2_head",
                "OccHead",
                'TRAFFICLANE_BEV',
                ]

lat_dict = {k:0.0 for k in keywords}
lat_dict['others'] = 0.0
sum_lat = 0.0
for layer in graph_json['Layers']:
    matched = False
    for k in keywords:
        if k in layer['Name'] or k in layer['Metadata']:
            lat_dict[k] += layer["lat"]
            sum_lat += layer["lat"]
            matched = True
            if k == "backbone0":
                print(layer["lat"], layer['Name'])
            break
    if not matched:
        lat_dict['others']+= layer["lat"]
        sum_lat += layer["lat"]
#         print("Not matched", layer["lat"], layer['Name'], layer['Metadata'])

sum_lat = 0.0
for k in lat_dict.keys():
    print(f"{k:<40s}  {lat_dict[k]:>5.2f} " )
    sum_lat += lat_dict[k]
total = "total"
print(f"{total:<40s}  {sum_lat:>5.2f} ")

# +
import json
folder_name = "/media/models/sorin/0313_ptq_dummy/patched_degroup_strongly_15_onemodel_v2_ds.onnx.trt_profile/patched_degroup_strongly_15_onemodel_v2_ds.onnx.trt"

with open(f"{folder_name}.profile.json", 'r') as f:
    prof = json.load(f)

with open(f"{folder_name}.graph.json", 'r') as f:
    graph_json = json.load(f)

name_lat_dict={}
for info in prof[1:]:
    name_lat_dict[info['name']] = info['averageMs']
for layer in graph_json['Layers']:
    layer["lat"] = name_lat_dict[layer['Name']]

keywords = ['backbone0','backbone3', 'backbone4', 
                'group0', 'group1','group2','group3', 'group8',
                "neck10",'neck0','neck1','neck2','neck3','neck4','neck5', 
                'transformer0',
#                 'transformer1',
                'fuser0',
                'sequence',
                'MONO_OD','STATIC_OD','SIDE_OD','OCC_OD', 'BEV_OD', 'GATE_LEVER',
                'TRAFFICLANE_2D', 
                "bev_encoder",
                "bev_sdmap_fuser",
                 "BevVecHead",
                "GuideLineHead",
                "maptrv2_head",
                "OccHead",
                'TRAFFICLANE_BEV',
                ]

lat_dict = {k:0.0 for k in keywords}
lat_dict['others'] = 0.0
sum_lat = 0.0
for layer in graph_json['Layers']:
    matched = False
    for k in keywords:
        if k in layer['Name'] or k in layer['Metadata']:
            lat_dict[k] += layer["lat"]
            sum_lat += layer["lat"]
            matched = True
            if k == "backbone0":
                print(layer["lat"], layer['Name'])
            break
    if not matched:
        lat_dict['others']+= layer["lat"]
        sum_lat += layer["lat"]
#         print("Not matched", layer["lat"], layer['Name'], layer['Metadata'])

sum_lat = 0.0
for k in lat_dict.keys():
    print(f"{k:<40s}  {lat_dict[k]:>5.2f} " )
    sum_lat += lat_dict[k]
total = "total"
print(f"{total:<40s}  {sum_lat:>5.2f} ")

# +
import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan_dir = "/media/models/sorin/0219_v6.0/unified_model_0215_ptq_orin.trt_profile"
plan = get_plan(plan_dir)


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

with open(plan_dir + '/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

# backbone
edge_to_remove = []
# neck 0
# edge_to_remove = ['5-6', '12-13', '32-33','34-35','34-99','32-98' ,
#                  '12-80', '52-53', '111-123', '125-129','44-149', 
#                  '148-150', '148-153','110-124'
#                  ]
# neck 1
# edge_to_remove = ['98-99', '116-117', '118-122','122-125','132-139','152-182', '141-195']
# lss
# edge_to_remove = ['195-198', '196-198']
# IDAup
# edge_to_remove = ['5-17', '11-12', '39-44','43-47','56-58', '56-57','70-71']
# head
# edge_to_remove = ["37-123", "120-124"]
for edge in edge_to_remove:
    nodes = edge.split('-')
    node1 = int(nodes[0])
    node2 = int(nodes[1])
    print(node1, node2)
    G.remove_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

keywords = ['backbone0', 'backbone3','backbone4','backbone5', 
            'group0', 'group1','group3', 'group7',
            'neck0','neck1','neck2','neck3','neck4','neck5', 
            'transformer0','transformer1','fuser0','fuser1',
            'sequence',
            'MultiscaleDeformableAttnPlugin_TRT',
            'MONO_OD','MONO3D','STATIC_OD','SIDE_OD','OCC_OD', 'BEV_OD','MONODEPTH',
            "bev_encoder",
            "maptrv2_head",
            'GuideLineHead',
            'TRAFFICLANE_2D', 'TRAFFICLANE_BEV',
#             'seg_static_freespace_head',
#             'seg_speedbump_head',
#             'BevSegHead',
#             'PendingHead',
#             'ArrowHead',
#             'CenterlineHead',
            ]


flag = [0] * len(G.nodes)
lats=0.0
for keyword in keywords:
    lat = 0.0
    for i in range(len(G.nodes)):
        if keyword in G.nodes[i]['name'] and flag[i] == 0:
#             print("@@@ ",G.nodes[i]['name'])
            lat += G.nodes[i]['weight']
            flag[i] += 1
    print(f"{keyword:<30s}  {lat:>5.2f}")
    lats += lat
    
other_value = 0
for i in range(len(G.nodes)):
    if flag[i] == 0:
#         if (G.nodes[i]['weight'] >0.2):
        other_value+=G.nodes[i]['weight']
print(f"others: {other_value:.2f}")
lats +=other_value
print("overal:", lats)

# +
import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan_dir = "/media/models/sorin/0219_v6.0/TASK11111111_sim.trt_profile"
plan = get_plan(plan_dir)


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

with open(plan_dir + '/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

# backbone
edge_to_remove = []
# neck 0
# edge_to_remove = ['5-6', '12-13', '32-33','34-35','34-99','32-98' ,
#                  '12-80', '52-53', '111-123', '125-129','44-149', 
#                  '148-150', '148-153','110-124'
#                  ]
# neck 1
# edge_to_remove = ['98-99', '116-117', '118-122','122-125','132-139','152-182', '141-195']
# lss
# edge_to_remove = ['195-198', '196-198']
# IDAup
# edge_to_remove = ['5-17', '11-12', '39-44','43-47','56-58', '56-57','70-71']
# head
# edge_to_remove = ["37-123", "120-124"]
for edge in edge_to_remove:
    nodes = edge.split('-')
    node1 = int(nodes[0])
    node2 = int(nodes[1])
    print(node1, node2)
    G.remove_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

keywords = ['backbone0', 'backbone3','backbone4','backbone5', 
            'group0', 'group1','group3', 'group7',
            'neck0','neck1','neck2','neck3','neck4','neck5', 
            'transformer0','transformer1','fuser0','fuser1',
            'sequence',
            'MultiscaleDeformableAttnPlugin_TRT',
            'MONO_OD','MONO3D','STATIC_OD','SIDE_OD','OCC_OD', 'BEV_OD','MONODEPTH',
            "bev_encoder",
            "maptrv2_head",
            'GuideLineHead',
            'TRAFFICLANE_2D', 'TRAFFICLANE_BEV',
#             'seg_static_freespace_head',
#             'seg_speedbump_head',
#             'BevSegHead',
#             'PendingHead',
#             'ArrowHead',
#             'CenterlineHead',
            ]


flag = [0] * len(G.nodes)
lats=0.0
for keyword in keywords:
    lat = 0.0
    for i in range(len(G.nodes)):
        if keyword in G.nodes[i]['name'] and flag[i] == 0:
#             print("@@@ ",G.nodes[i]['name'])
            lat += G.nodes[i]['weight']
            flag[i] += 1
    print(f"{keyword:<30s}  {lat:>5.2f}")
    lats += lat
    
other_value = 0
for i in range(len(G.nodes)):
    if flag[i] == 0:
#         if (G.nodes[i]['weight'] >0.2):
        other_value+=G.nodes[i]['weight']
print(f"others: {other_value:.2f}")
lats +=other_value
print("overal:", lats)
# -

for i in range(len(G.nodes)):
    if flag[i] == 0:
        print(G.nodes[i])

# +
import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan_dir = "/media/models/vision+env/unified_QAT/TASK1111111_sim_opt.trt4_profile"
plan = get_plan(plan_dir)


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

# for i in range(len(plan.df)):
#     for out_name in plan.df.iloc[i]['out_names']:
#         for j in range(len(plan.df)):
#             if out_name in plan.df.iloc[j]['inp_names']:
# #                 print(i, j)
#                 G.add_edge(i, j)
with open(plan_dir + '/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

# backbone
edge_to_remove = []
# neck 0
# edge_to_remove = ['5-6', '12-13', '32-33','34-35','34-99','32-98' ,
#                  '12-80', '52-53', '111-123', '125-129','44-149', 
#                  '148-150', '148-153','110-124'
#                  ]
# neck 1
# edge_to_remove = ['98-99', '116-117', '118-122','122-125','132-139','152-182', '141-195']
# lss
# edge_to_remove = ['195-198', '196-198']
# IDAup
# edge_to_remove = ['5-17', '11-12', '39-44','43-47','56-58', '56-57','70-71']
# head
# edge_to_remove = ["37-123", "120-124"]
for edge in edge_to_remove:
    nodes = edge.split('-')
    node1 = int(nodes[0])
    node2 = int(nodes[1])
    print(node1, node2)
    G.remove_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

keywords = ['backbone0', 'backbone3', 'group0', 'group1','group3', 'neck0','neck1','neck2','neck3','neck4','neck5', 
            'transformer0','transformer1','fuser0','fuser1','TRAFFICLANE_2D', 'TRAFFICLANE_BEV', 
            'BEV_OD','MONO_OD','MONO3D','STATIC_OD','SIDE_OD','sequence_dynamic'
#             'seg_static_freespace_head',
#             'seg_speedbump_head',
#             'BevSegHead',
#             'PendingHead',
#             'ArrowHead',
#             'CenterlineHead',
            ]


flag = [0] * len(G.nodes)
lats=0.0
for keyword in keywords:
    lat = 0.0
    for i in range(len(G.nodes)):
        if keyword in G.nodes[i]['name']:
#             print("@@@ ",G.nodes[i]['name'])
            lat += G.nodes[i]['weight']
            flag[i] += 1
    print(keyword, lat)
    lats += lat
print("overal:", lats)

# +

import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan_dir = "/media/models/static/0102_BEV150/trafficlane_1224.trt_profile"
plan = get_plan(plan_dir)


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

# for i in range(len(plan.df)):
#     for out_name in plan.df.iloc[i]['out_names']:
#         for j in range(len(plan.df)):
#             if out_name in plan.df.iloc[j]['inp_names']:
# #                 print(i, j)
#                 G.add_edge(i, j)
with open(plan_dir + '/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

keywords = ['image_backbone', 
            'image_neck', 
            'image_seg_head',
            'map_encoder', 
            'map_guideline_head',
            'lidar_backbone',
            'lidar_fpn',
            'sd_map_encoder',
            'MultiscaleDeformableAttnPlugin',
            'centerline_head',
            'arrow_head',
            'view_transfomer', 
            'bev_feature_encoder',
            'bev_feature_sdmap_fusion',
            'bev_head',
            'bev_seq_fusion',
            'bev_seg_head',
            'bev_',
            'pending_head',
            'Reformatting', 
            'ForeignNode',
            'PWN',
            ]


flag = [0] * len(G.nodes)
lats=0.0
for keyword in keywords:
    lat = 0.0
    for i in range(len(G.nodes)):
        if keyword in G.nodes[i]['name'] and flag[i] == 0:
#             print("@@@ ",G.nodes[i]['name'])
            lat += G.nodes[i]['weight']
            flag[i] += 1
    print(f"{keyword:<30s}  {lat:>5.2f}")
    lats += lat
    
other_value = 0
for i in range(len(G.nodes)):
    if flag[i] == 0:
#         if (G.nodes[i]['weight'] >0.2):
        other_value+=G.nodes[i]['weight']
print(f"others: {other_value:.2f}")
lats +=other_value
print("overal:", lats)


# +
import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan_dir = "/media/models/static/20250116-1848__07_trafficlight_0107.trt_profile"
plan = get_plan(plan_dir)


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

# for i in range(len(plan.df)):
#     for out_name in plan.df.iloc[i]['out_names']:
#         for j in range(len(plan.df)):
#             if out_name in plan.df.iloc[j]['inp_names']:
# #                 print(i, j)
#                 G.add_edge(i, j)
with open(plan_dir + '/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

keywords = ['backbone', 
            'temporal_layer', 
            'neck',
            'head', 
            ]


flag = [0] * len(G.nodes)
lats=0.0
for keyword in keywords:
    lat = 0.0
    for i in range(len(G.nodes)):
        if keyword in G.nodes[i]['name'] and flag[i] == 0:
#             print("@@@ ",G.nodes[i]['name'])
            lat += G.nodes[i]['weight']
            flag[i] += 1
    print(f"{keyword:<30s}  {lat:>5.2f}")
    lats += lat
    
other_value = 0
for i in range(len(G.nodes)):
    if flag[i] == 0:
#         if (G.nodes[i]['weight'] >0.2):
        other_value+=G.nodes[i]['weight']
print(f"others: {other_value:.2f}")
lats +=other_value
print("overal:", lats)
# -




value = 0
for i in range(len(G.nodes)):
    if flag[i] == 0:
#         if (G.nodes[i]['weight'] >0.2):
        print("Not ",G.nodes[i]['name'], flag[i], G.nodes[i]['weight'])
        value+=G.nodes[i]['weight']
#     if flag[i] > 1:
#         print("Multi ",G.nodes[i]['name'], flag[i])
print(value)

# +
import sys
sys.path.append('/media/Projects/MATool')
from trex import *
import networkx as nx

def plan_analizer(plan_dir = None, keywords = []):
    plan = get_plan(plan_dir)
    # 创建有向图对象
    G = nx.DiGraph()
    for i in range(len(plan.df)):
        node_ind = i
    # 添加带有权重的节点
        G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

    # for i in range(len(plan.df)):
    #     for out_name in plan.df.iloc[i]['out_names']:
    #         for j in range(len(plan.df)):
    #             if out_name in plan.df.iloc[j]['inp_names']:
    # #                 print(i, j)
    #                 G.add_edge(i, j)
    with open(plan_dir + '/edge_info.json', 'r') as file:
        edge_to_add = json.load(file)
    for edge in edge_to_add:
        if "X" not in edge:
            nodes = edge.split('-')
            node1 = int(nodes[0])
            node2 = int(nodes[1])
    #         print(node1, node2)
            G.add_edge(node1, node2)

    connected_components = list(nx.weakly_connected_components(G))
    # 计算节点权重的总和
    total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
    print("Total Node Weight:", total_weight)
    print("subgraph num: ", len(connected_components))
    for cc in connected_components:
        print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

    flag = [0] * len(G.nodes)
    lat_dict= {}
    lats=0.0
    for keyword in keywords:
        lat = 0.0
        refmat_lat = 0.0
        for i in range(len(G.nodes)):
            if keyword in G.nodes[i]['name'] and flag[i] == 0:
    #             print("@@@ ",G.nodes[i]['name'])
                lat += G.nodes[i]['weight']
                flag[i] += 1
                if "Reformatting" in G.nodes[i]['name'] or G.nodes[i]['name'].endswith("_input_quantizer/QuantizeLinear"):
                    refmat_lat += 1 #G.nodes[i]['weight']
#                 if keyword == "bev_seg_head":
#                     print(G.nodes[i]['name'])
        print(f"{keyword:<30s}  {lat:>5.2f} reformat {refmat_lat}")
        lat_dict[keyword] = lat
        lats += lat

    other_value = 0
    for i in range(len(G.nodes)):
        if flag[i] == 0:
    #         if (G.nodes[i]['weight'] >0.2):
            other_value+=G.nodes[i]['weight']
    print(f"others: {other_value:.2f}")
    lat_dict["others"] = other_value
    lats_all = lats + other_value
    print("selected:", lats)
    print("overal:", lats_all)
    lat_dict["latency without cudagraph"] = lats
    
    t_lat = 0
    for i in range(len(G.nodes)):
        if "GuideLineHead" in G.nodes[i]['name'] and "TRAFFICLANE_2D" in G.nodes[i]['name']:
            print(G.nodes[i]['name'], G.nodes[i]['weight'])
            t_lat+=G.nodes[i]['weight']
    print(t_lat)
    return lat_dict


# -

plan_dir = "/media/models/static/0102_BEV150/0102_test_calib_ptq_histogram_optqat_sim.trt_profile"
keywords = ['image_backbone', 
                'image_neck', 
                'image_seg_head',
                'map_encoder', 
                'map_guideline_head',
                'lidar_backbone',
                'lidar_fpn',
                'sd_map_encoder',
                'MultiscaleDeformableAttnPlugin',
                'centerline_head',
                'arrow_head',
                'view_transfomer', 
                'bev_feature_encoder',
                'bev_feature_sdmap_fusion',
                'bev_backbone',
                'bev_seq_fusion',
                'bev_fpn',
                'bev_seg_head',
                'bev_occ_head',
                'bev_seg_downsample',
                'bev_head',
                'pending_head',
                'reformat', 
                'ForeignNode',
                'PWN',
                ]
lat_dict_1 = plan_analizer(
    plan_dir="/media/models/static/0102_BEV150/trafficlane_0106_sim.trt_profile", 
    keywords = keywords)
lat_dict_2 = plan_analizer(
    plan_dir="/media/models/static/0102_BEV150/0102_test_calib_ptq_histogram_optqat_sim.trt_profile", 
    keywords = keywords)

c0, c1, c2, c3 = "Module Name","before", "after", "diff"
print(f"{c0:<30s}  {c1:>6s}   {c2:>6s} {c3:>6s}")
for k in lat_dict_1.keys():
    print(f"{k:<30s}  {lat_dict_1[k]:>6.2f}   {lat_dict_2[k]:>6.2f} {lat_dict_2[k]-lat_dict_1[k]:>6.2f}")
c0, c1, c2, c3 = "latency with cuda graph","47.3", "67.8", "20.5"
print(f"{c0:<30s}  {c1:>6s}   {c2:>6s} {c3:>6s}")



keywords = ['backbone0', 
            'backbone3', 
            'group0', 
#             'group1',
#             'group3', 
            'neck0',
#             'neck1',
#             'neck2',
#             'neck3',
#             'neck4',
            'neck5', 
            'transformer0',
#             'transformer1',
            'fuser0',
#             'fuser1',
            'GuideLineHead',
            'TRAFFICLANE_2D', 'TRAFFICLANE_BEV', 
#             'BEV_OD','MONO_OD','MONO3D','STATIC_OD','SIDE_OD','sequence_dynamic'
#             'seg_static_freespace_head',
#             'seg_speedbump_head',
#             'BevSegHead',
#             'PendingHead',
#             'ArrowHead',
#             'CenterlineHead',
            ]
lat_dict_1 = plan_analizer(
    plan_dir="/media/models/sorin/0109_test2/_all_sim_opt_plugin.trt_profile_orin", 
    keywords = keywords)



# +

import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan = get_plan("/media/models/vision+env/v4/unifiedmodel_v4_0511.trt_profile")


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

# for i in range(len(plan.df)):
#     for out_name in plan.df.iloc[i]['out_names']:
#         for j in range(len(plan.df)):
#             if out_name in plan.df.iloc[j]['inp_names']:
# #                 print(i, j)
#                 G.add_edge(i, j)
with open('/media/models/vision+env/v4/unifiedmodel_v4_0511.trt_profile/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

edge_to_remove = [
#                 '5-17', '11-12', '11-38','32-33', 
#                 '32-42','34-35','106-109', '106-110',
#                 '86-87', '103-104', '107-111',
#                 '56-153', '56-161', '120-136',
#                 '107-108', '111-116', '117-124'
#                 '153-163', '120-137'
#                 '219-222', '219-230',
#                 '117-154', '153-155', '219-220'
]
# neck 0
edge_to_remove = ['5-6', '12-66', '12-13','29-30','29-84','31-32', '31-85',
                 '33-94', '45-46', '93-95', '93-100','32-87', '85-88', '85-90',
                 '35-104', '103-117', '103-105']
# neck 1
# edge_to_remove = ['86-87', '103-104', '107-111','111-116','117-124','117-154']
# lss
# edge_to_remove = ['154-162', '153-155', '218-219']
# IDAup
# edge_to_remove = ['5-17', '11-12', '39-44','43-47','56-58', '56-57','70-71']
for edge in edge_to_remove:
    nodes = edge.split('-')
    node1 = int(nodes[0])
    node2 = int(nodes[1])
#     print(node1, node2)
    G.remove_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))
# -

connected_components[1]

connected_components[3]

# # 视觉多任务

# +

import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan_dir = "/media/models/vision+env/v4.4/checkpoint_sim_bevformer256.trt_profile"
plan = get_plan(plan_dir)


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

# for i in range(len(plan.df)):
#     for out_name in plan.df.iloc[i]['out_names']:
#         for j in range(len(plan.df)):
#             if out_name in plan.df.iloc[j]['inp_names']:
# #                 print(i, j)
#                 G.add_edge(i, j)
with open(plan_dir + '/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

# backbone
# edge_to_remove = []
# neck 0
# edge_to_remove = ['5-6', '12-13', '34-35','36-37','36-109','34-108' ,
#                  '12-90', '69-70', '54-165', '164-166','54-178', 
#                  '164-168', '164-170','124-139', '124-132','45-131', '37-114',
#                   '109-115', '109-118'
#                  ]
# neck 1
edge_to_remove = ['87-88', '105-106', '107-112','112-116','122-129','130-167', '144-181']
# 2d
# edge_to_remove = ['158-161', '172-179','189-196']
# # 2d_side
# edge_to_remove = ['144-145', '151-153','155-159']
# # 3d
# edge_to_remove = ['79-311', '79-312','79-310']
# # traffic bev & lss
# edge_to_remove = ['256-257', '54-178']
# # bev_od
# edge_to_remove = ['164-168', '130-167']
# IDAup
# IDAup
# edge_to_remove = ['5-17', '11-12', '39-44','43-47','56-58', '56-57','70-71']
# head
# edge_to_remove = ["37-123", "120-124"]
for edge in edge_to_remove:
    nodes = edge.split('-')
    node1 = int(nodes[0])
    node2 = int(nodes[1])
    print(node1, node2)
    G.remove_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc))

# +
import sys
sys.path.append('/media/Projects/MATool')
from trex import *

import networkx as nx

plan_dir = "/media/models/vision+env/1029_qat/TASK0100000_sim_opt.trt_profile"
plan = get_plan(plan_dir)


# 创建有向图对象
G = nx.DiGraph()
for i in range(len(plan.df)):
    node_ind = i
# 添加带有权重的节点
    G.add_node(node_ind, name=plan.df.iloc[i]['Name'], weight=plan.df.iloc[i]['latency.avg_time'])

# for i in range(len(plan.df)):
#     for out_name in plan.df.iloc[i]['out_names']:
#         for j in range(len(plan.df)):
#             if out_name in plan.df.iloc[j]['inp_names']:
# #                 print(i, j)
#                 G.add_edge(i, j)
with open(plan_dir + '/edge_info.json', 'r') as file:
    edge_to_add = json.load(file)
for edge in edge_to_add:
    if "X" not in edge:
        nodes = edge.split('-')
        node1 = int(nodes[0])
        node2 = int(nodes[1])
#         print(node1, node2)
        G.add_edge(node1, node2)

# backbone
# group 0
edge_to_remove = ['106-107','80-81', '77-81']

for edge in edge_to_remove:
    nodes = edge.split('-')
    node1 = int(nodes[0])
    node2 = int(nodes[1])
    print(node1, node2)
    G.remove_edge(node1, node2)

connected_components = list(nx.weakly_connected_components(G))
# 计算节点权重的总和
total_weight = sum(G.nodes[node]['weight'] for node in G.nodes)
print("Total Node Weight:", total_weight)
print("subgraph num: ", len(connected_components))
for cc in connected_components:
    print(f"包含{list(cc)[0]}号节点的子图耗时：", sum(G.nodes[node_id]['weight'] for node_id in cc), len(list(cc)))
# -

connected_components[0]
