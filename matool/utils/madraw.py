#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""
This script generates an SVG diagram of the input engine graph SVG file.

Note: this script requires graphviz which can be installed manually:
    $ sudo apt-get --yes install graphviz
    $ python3 -m pip install graphviz networkx
"""


import graphviz
from trex import *
from trex import graphing2 as gr2
import argparse
import shutil
import glob


def draw_from_json(engine_json_fname):
    graphviz_is_installed = shutil.which("dot") is not None
    if not graphviz_is_installed:
        print("graphviz is required but it is not installed.\n")
        print("To install on Ubuntu:")
        print("sudo apt --yes install graphviz")
        exit()

    plan = EnginePlan(engine_json_fname)
    formatter = layer_type_formatter
    display_regions = True
    expand_layer_details = False

    graph = to_dot(plan, formatter,
                display_regions=display_regions,
                expand_layer_details=expand_layer_details)
    render_dot(graph, engine_json_fname, 'svg')


def draw_engine(plan, view=False):
    graphviz_is_installed = shutil.which("dot") is not None
    if not graphviz_is_installed:
        print("graphviz is required but it is not installed.\n")
        print("To install on Ubuntu:")
        print("sudo apt --yes install graphviz")
        exit()

    formatter = gr2.layer_type_formatter_simple
    display_regions = True
    expand_layer_details = False

    try:
        print(">>> 尝试进行【结构】和【数据流】可视化")
        graph = to_dot(plan, formatter,
                display_regions=display_regions,
                expand_layer_details=expand_layer_details)
        render_dot(graph, args.save_path + '/graph', 'svg')
        # 数据流可视化 边粗细代表数据流大小，节点height代表latency
        graph2 = gr2.to_dot(plan, formatter,
                    display_layer_names=False,
                    display_regions=False,
                    expand_layer_details=expand_layer_details,
                    min_edge_width=2,
                    max_edge_width=50,
                    min_op_height=20,
                    max_op_height=300,
                    )
        render_dot(graph2, args.save_path + '/graph_2', 'svg')
        print(">>> 创建【结构可视化】和【数据流可视化】成功")
    except:
        try:
            print(">>> 详细数据缺失，尝试进行【简单结构可视化】")
            graph = to_dot(plan, formatter,
                    display_regions=display_regions,
                    expand_layer_details=expand_layer_details)
            render_dot(graph,  args.input, 'svg', view)
            print(">>> 创建【简单结构可视化】成功")
        except:
            print(">>> 创建可视化失败")
    

def get_plan(args):
    if args.input.endswith('.json'):
        args.save_path = os.path.dirname(args.input)
        return EnginePlan(args.input)
    else:
        args.save_path = args.input
        profile_path = os.path.normpath(args.input)
        profile_files = glob.glob(os.path.join(profile_path, "*.profile.json"))
        if len(profile_files) > 1:
            print(f"Error: 多个*.profile.json文件: {profile_files}")
            print(">>> >>> profile文件检测失败")
            return
        elif len(profile_files) == 0:
            print(f"Error: 缺少*.profile.json文件")
            print(">>> >>> profile文件检测失败")
            return

        profile_file = profile_files[0]
        model_name = os.path.basename(profile_file)[:-13]
        graph_file = os.path.join(profile_path, f'{model_name}.graph.json')
        profile_meta_file = os.path.join(profile_path, f'{model_name}.profile.metadata.json')

        plan = EnginePlan(graph_file,
                      profile_file,
                      profile_meta_file)
        return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='samples/radar_bev', help="layerinfo.json 或者profile文件夹")
    args = parser.parse_args()
    # args.input='/media/Projects/ModelBoard/database/offline_models/fisheye/fisheye_20231117'
    plan = get_plan(args)
    draw_engine(plan)