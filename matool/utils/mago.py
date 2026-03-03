#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

from trex import graphing3 as gr3
from trex import graphing2 as gr2
from trex import *

import argparse
import plotly.offline as pyo

import shutil
from markdown2HTML import markdown2html
from parse_trtexec_log import parse_build_log, parse_profiling_log

import glob

from profile2graph import profile2graph
from plan2info_dict import * 

from plan2reformat_PWN import plan2reformat_PWN

from bs4 import BeautifulSoup

mago_dir = os.path.dirname(os.path.abspath(__file__))

fig_dict = {
            '算子数量统计(type+precision)': 1,
            '算子数量统计(subtype+precision)': 0,
            '算子延迟百分占比(type+precision)': 0,
            '算子延迟百分占比(type)': 0,
            '算子分析饼图': 1,
            '算子延迟百分占比(subtype + precision)': 0,
            '算子延迟测算(type)': 1,
            '算子延迟测算(precision)': 0,
            '算子延迟分布直方图': 0,
            '算子subtype延迟测算': 0,
            '算子footprint测算': 1,
            '算子内存效率测算': 1,
            '算子延迟树状图-延迟vs延迟': 0,
            '算子延迟树状图-延迟vs大小': 0,
            '算子参数量统计图': 1,
            '算子输出图大小统计图': 1,
            '算子输出图大小分布直方图': 0,
            '算子数值精度统计图': 0,
            '卷积算子FLOPS测算': 1,
            '卷积算子计算强度测算': 0,
            '卷积算子io测算': 0,
            '卷积算子内存效率': 0,
            '卷积算子计算效率': 1,
            '卷积算子饼图-算子种类': 0,
            '卷积算子饼图-分组参数': 0,
            '卷积算子饼图-卷积核': 0,
            '卷积算子饼图-数值精度': 0,
            '卷积算子延迟统计': 0}


def html_preparation(save_path, open_file=False):
    # html_save_path = save_path + '/source'
    # os.makedirs(html_save_path, exist_ok=True)

    sample_script_path = os.path.abspath(sys.argv[0])
    sample_script_directory = os.path.dirname(sample_script_path)

    file2mainfolder = ['MATool.html']
    folder2mainfolder = ['source']

    for file_name in folder2mainfolder:
        sample_script = sample_script_directory + '/html_template/' + file_name
        target_script = save_path + '/' + file_name
        shutil.copytree(sample_script, target_script, dirs_exist_ok=True)
        # print("模型分析报告source文件夹 ", os.path.abspath(target_script))

    for file_name in file2mainfolder:
        sample_script = sample_script_directory + '/html_template/' + file_name
        target_script = save_path + '/' + file_name
        shutil.copyfile(sample_script, target_script)
    
    print(">>> 模型分析报告生成完毕")
    print(">>> 模型分析报告入口：", 
        os.path.abspath(save_path + '/' + file2mainfolder[0]))
    if (open_file):
        try:
            import webbrowser
            webbrowser.open(os.path.abspath(target_script), new=2)
        except:
            print("模型分析汇总文件打开失败，请手动打开")

def svg2html(svg_file, html_target, html_template=mago_dir + "/html_template/source/SVGviewer.html"):
    # 读取 A.html 文件
    with open(svg_file, 'r', encoding='utf-8') as file:
        svg_content = file.read()

    # 读取 B.html 文件
    with open(html_template, 'r', encoding='utf-8') as file:
        html_content = file.readlines()

    # 使用 BeautifulSoup 解析 A.html 文件
    soup = BeautifulSoup(svg_content, 'html.parser')

    # 获取 <svg> 标签内容
    svg_content = str(soup.find('svg'))

    line_number = 90
    html_content.insert(line_number, svg_content)

    # 将更新后的内容写入到 B.html 文件中
    with open(html_target, 'w', encoding='utf-8') as file:
        file.writelines(html_content)


def profile2structure(plan, path):
    graphviz_is_installed = shutil.which("dot") is not None
    if not graphviz_is_installed:
        print("graphviz is required but it is not installed.\n")
        print("To install on Ubuntu:")
        print("sudo apt --yes install graphviz")
        exit()
    save_path = path + '/structure'
    os.makedirs(save_path, exist_ok=True)

    # formatter = layer_type_formatter if True else precision_formatter
    # graph = to_dot(plan, formatter)

    formatter = gr2.layer_type_formatter_simple
    display_regions = False
    expand_layer_details = False
    DEBUG = False
    # layers_not_to_draw=["shape_call", "wait", "signal"]
    # plan._df = plan.df.drop(plan.df[plan.df['subtype'].isin(layers_not_to_draw)].index)
    # breakpoint()
    if not DEBUG:
        graph = to_dot(plan, formatter,
                    display_regions=display_regions,
                    expand_layer_details=expand_layer_details)
        render_dot(graph, save_path + '/engine', 'svg')
        svg2html(save_path + '/engine.svg', save_path + '/engine.html')

        # try:
        # 数据流可视化 边粗细代表数据流大小，节点height代表latency
        graph2, edge_record = gr2.to_dot(plan, formatter,
                    display_layer_names=False,
                    display_regions= display_regions,
                    expand_layer_details=expand_layer_details,
                    min_edge_width=2,
                    max_edge_width=50,
                    min_op_height=20,
                    max_op_height=300,
                    )

        render_dot(graph2, save_path + '/engine2', 'svg')
        svg2html(save_path + '/engine2.svg', save_path + '/engine2.html')
        with open(path + '/../edge_info.json', 'w') as file:
            json.dump(edge_record, file)
        # except:
        #     print(">>> 创建数据流可视化失败")
        # png_name = render_dot(graph, graph_name, 'png')
    else:
        graph3, edge_record = gr3.to_dot(plan, formatter,
                    display_layer_names=False,
                    display_regions= display_regions,
                    expand_layer_details=expand_layer_details,
                    min_edge_width=2,
                    max_edge_width=50,
                    min_op_height=20,
                    max_op_height=300,
                    )

        render_dot(graph3, save_path + '/engine2', 'svg')
        svg2html(save_path + '/engine2.svg', save_path + '/engine2.html')
        with open(path + '/../edge_info.json', 'w') as file:
            json.dump(edge_record, file)
    

def reorder_df_column(df):
    column = list(df)
    top_column = ['Name', 'type', 'subtype', 
                'latency.avg_time', 'latency.pct_time',
                'output_precision', 'precision',
                ]
    index = 0
    for col in top_column:
        try:
            column.insert(index, column.pop(column.index(col)))
            index += 1
        except:
            pass
    return df.loc[:, column]


    
def main_parse(model, output=None, op_detailed=False):
    print(">>> MATool")
    profile_path = os.path.normpath(model)
    if not os.path.exists(profile_path):
        print(f"Error: 文件目录{profile_path}不存在")
        print(">>> >>> profile文件检测失败")
        return
    
    P_const = 25

    print(">>> >>>", "profile文件目录".ljust(P_const-4), ">>>", profile_path)
    print(">>> >>>", "profile文件检测".ljust(P_const-4), end=" ")

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

    if not os.path.exists(graph_file):
        print(f"Error: 缺少profile文件: {model_name}.graph.json")
        print(">>> >>> profile文件检测失败")
        return

    profile_meta_file = os.path.join(profile_path, f'{model_name}.profile.metadata.json')
    profile_log_file = os.path.join(profile_path, f'{model_name}.profile.log')
    if not os.path.exists(profile_meta_file):
        if os.path.exists(profile_log_file):
            # print("profile log文件存在，生成meta文件")
            profiling_metadata = parse_profiling_log(profile_log_file)
            with open(profile_meta_file, 'w') as fout:
                json.dump(profiling_metadata , fout)
        else:
            profile_meta_file = ''
            print("profile log文件不存在，生成简易版报告")
    print(">>>", "PASSED")

    # 访问命令行参数的值
    output_file = output
    save_path = os.path.join(profile_path, 'report')
    if output_file:
        save_path = output_file
        # print("使用指定输出目录：", save_path)
    if os.path.exists(save_path):
        # pass
        shutil.rmtree(save_path)
        print(">>> >>>", "输出目录已存在".ljust(P_const-7), ">>>", "删除原目录")

    os.makedirs(save_path, exist_ok=True)

    print(">>> >>> 模型分析报告开始生成")
    
    print(">>> >>>", "模型分析报告目录".ljust(P_const-8), ">>>",save_path)

    # plan = EnginePlan(f'{engine_name}.graph.json',
    #                   f'{engine_name}.profile.json',
    #                   f'{engine_name}.profile.metadata.json')

    plan = EnginePlan(graph_file,
                      profile_file,
                      profile_meta_file)

    # print(plan.df.columns)
    # return

    # # 网络结构可视化
    print(">>> >>>", "创建模型结构可视化".ljust(P_const-9), end=" ")
    sys.stdout.flush()
    # breakpoint()
    profile2structure(plan, save_path)
    print(">>> PASSED")
    # 文本分析
    print(">>> >>>", "进行模型数据分析".ljust(P_const-8), end=" ")
    sys.stdout.flush()
    info_dict_basic = plan2info_dict_basic(plan, model_name)
    info_dict_inference = plan2info_dict_inference(plan, model_name)
    info_dict_device = plan2info_dict_device(plan, model_name)
    info_dict_abstract = plan2info_dict_abstract(plan, model_name)
    print(">>> PASSED")
    # 文本报告
    print(">>> >>>", "生成模型分析报告".ljust(P_const-8), end=" ")
    sys.stdout.flush()
    info_dict2report(save_path, info_dict_abstract,
                     report_name='report0', report_title='')
    info_dict2report(save_path, info_dict_basic,
                     report_name='report1', report_title='模型静态属性')
    info_dict2report(save_path, info_dict_inference,
                     report_name='report2', report_title='模型推理指标')
    info_dict2report(save_path, info_dict_device,
                     report_name='report3', report_title='硬件设备信息')
    print(">>> PASSED")
    if op_detailed:
        print(">>> >>>","模型算子详细解析".ljust(P_const-8), end=" ")
        sys.stdout.flush()
        info_dict2report(save_path, {"算子详细信息": reorder_df_column(plan.df)},
                        report_name='report4', report_title='算子详细信息')    
        print(">>> PASSED")
    else:
        print(">>> >>>","模型算子详细解析".ljust(P_const-8), end=" ")
        print(">>> SKIPPED")
    shutil.copyfile(profile_file,
                    save_path + '/reports' + '/report5.json')
    shutil.copyfile(graph_file,
                    save_path + '/reports' + '/report6.json')
    try:
        shutil.copyfile(profile_log_file,
                        save_path + '/reports' + '/report7.json')
    except:
        pass
        # print(f"File {model_name}.profile.log not exists")

    # # 数据绘图
    print(">>> >>>", "绘制分析数据图表".ljust(P_const-8), end=" ")
    profile2graph(fig_dict, plan, save_path)
    print(">>> PASSED")

    #reformatting & PWN 分析报告
    try:
        plan2reformat_PWN(plan,save_path)
    except:
        pass

    # html报告
    html_preparation(save_path,open_file=False)


def main(open_file=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path_default = '../samples/radar_bev/dynamic_model_concat.onnx.engine'
    model_path_default ='../samples/vision_twostage_profile/'
    model_path_default = os.path.join(script_dir, model_path_default)

    # 创建一个ArgumentParser对象
    parser = argparse.ArgumentParser(
        description="该脚本对模型profile进行解析和数据分析")
    # parser.add_argument("--model", "-m", required=True, help="trt模型文件绝对路径")
    parser.add_argument("--model", "-m",
                        default=model_path_default,
                        help="trt模型文件绝对路径")
    parser.add_argument("--output", "-o", default='',
                        help="指定输出文件的绝对路径")
    
    parser.add_argument("--op-detailed", "-d", action='store_true',
                        help="是否生成详细算子报告")

    # parser.add_argument("--verbose", "-v", action="store_true", help="是否启用详细模式")
    args = parser.parse_args()
    main_parse(model=args.model,output=args.output, op_detailed=args.op_detailed)


if __name__ == '__main__':
    main(open_file=True)
