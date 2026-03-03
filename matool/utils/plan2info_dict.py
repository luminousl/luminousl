#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from trex import *

import argparse
import plotly.offline as pyo

import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.markdown2HTML import markdown2html
# from .parse_trtexec_log import parse_build_log, parse_profiling_log

import glob

def textalign(text, length=6, format="left"):
    content = f"{text}"
    if len(content) > length:
        return content
    if format == "left":
        content += "&nbsp;"*(length - len(content))*2
    elif format == "right":
        content = "&nbsp;"*(length - len(content))*2 + content
    elif format == "center":
        content = "&nbsp;"*(length - len(content)) + content + "&nbsp;"*(length - len(content))
    return content

def plan2info_dict_basic(plan, model_name):

    info_dict = {
        "模型名称": model_name,
        "基本指标": {},
    }

    # 模型输入输出
    input_bindings = plan.get_bindings()[0]
    input_bindings = sorted(input_bindings,
                            key=lambda input: input.size_bytes, reverse=True)
    input_attri = {'名称': [], 'shape': [], 'format': []}
    input_df = pd.DataFrame(input_attri)
    for input in input_bindings:
        new_data = {'名称': input.name, 'shape': input.shape,
                    'format': input.format}
        input_df = input_df.append(new_data, ignore_index=True)
    info_dict["模型输入"] = input_df

    output_bindings = plan.get_bindings()[1]
    output_attri = {'名称': [], 'shape': [], 'format': []}
    output_df = pd.DataFrame(output_attri)
    for output in output_bindings:
        new_data = {'名称': output.name, 'shape': output.shape,
                    'format': output.format}
        output_df = output_df.append(new_data, ignore_index=True)
    info_dict["模型输出"] = output_df

    # 算子种类统计
    layer_types = group_count(plan.df, 'type')
    info_dict["算子种类统计"] = layer_types

    # 基本指标
    plan_summary = summary_dict(plan)
    # breakpoint()

    info_dict["基本指标"]["模型深度"] = f"{plan_summary['Layers']}层"
    info_dict["基本指标"]["输入数目"] = input_df.shape[0]
    info_dict["基本指标"]["输出数目"] = output_df.shape[0]
    info_dict["基本指标"]["总计权重大小"] = f"{plan_summary['Weights']}"
    info_dict["基本指标"]["总计特征大小"] = f"{plan_summary['Activations']}"

    return info_dict

def plan2info_dict_abstract(plan, model_name):

    info_dict = {
        "模型分析摘要": {}
    }
    footnote = []
    info_dict["备注"]=footnote
    # 模型输入输出
    input_bindings = plan.get_bindings()[0]
    input_bindings = sorted(input_bindings,
                            key=lambda input: input.size_bytes, reverse=True)
    input_attri = {'名称': [], 'shape': [], 'format': []}
    input_df = pd.DataFrame(input_attri)
    for input in input_bindings:
        new_data = {'名称': input.name, 'shape': input.shape,
                    'format': input.format}
        input_df = input_df.append(new_data, ignore_index=True)
    info_dict["模型输入"] = input_df

    output_bindings = plan.get_bindings()[1]
    output_attri = {'名称': [], 'shape': [], 'format': []}
    output_df = pd.DataFrame(output_attri)
    for output in output_bindings:
        new_data = {'名称': output.name, 'shape': output.shape,
                    'format': output.format}
        output_df = output_df.append(new_data, ignore_index=True)
    info_dict["模型输出"] = output_df

    # 算子种类统计
    layer_types = group_count(plan.df, 'type')
    info_dict["算子种类统计"] = layer_types

    # 基本指标
    plan_summary = summary_dict(plan)

    info_dict["模型分析摘要"]["模型名称"] = model_name
    # info_dict["模型分析报告摘要"]["总计特征大小"] = f"{plan_summary['Activations']}"
    # if plan.performance_summary:
    #     info_dict['模型分析摘要'][
    #         '模型吞吐量'] = f"{plan.performance_summary['Throughput']} qps"
    #     info_dict['模型分析摘要'][
    #         '平均延迟'] = f"{plan.performance_summary['Latency'][2]} ms"
    lat = plan.df['latency.pct_time']
    lat_norm = lat / (100 / len(lat))
    mem = plan.df['total_io_size_bytes']
    mem_norm = mem / (sum(mem) / len(mem))


    df_t = group_count_multi(plan.df, ['type', 'precision'])
    # breakpoint()
    df_conv = df_t[df_t['type'] =='Convolution']
    df_deconv = df_t[df_t['type'] =='Deconvolution']
    df_conv_deconv = pd.merge(df_conv, df_deconv, how='outer')
    conv_num = df_conv_deconv['count'].sum()
    conv_int8_num = df_conv_deconv[df_conv_deconv['precision'] == "INT8"]['count'].sum()

    df_pw = df_t[df_t['type'] =='PointWise']
    pw_num = df_pw['count'].sum()
    pw_int8_num = df_pw[df_pw['precision'] == "INT8"]['count'].sum()

    df_reformat = df_t[df_t['type'] =='Reformat']
    reformat_num = df_reformat['count'].sum()
    reformat_latency = plan.df[plan.df['type']=="Reformat"]['latency.avg_time'].sum()
    all_latency = plan.df['latency.avg_time'].sum()
    int8_num = plan.df[plan.df['precision'] == 'INT8']['Name'].count()
    all_num = plan.df['Name'].count()
    # breakpoint()
    try:
        info_dict['模型分析摘要']['平均延迟' + "-"*18] = textalign(f"{plan.performance_summary['Latency'][2]:.1f}") + " ms"
        info_dict['模型分析摘要']['模型吞吐量' + "-"*15] = textalign(f"{plan.performance_summary['Throughput']:.1f}") + " qps"
    except:
        pass
    info_dict["模型分析摘要"]["算子总数" + "-"*18] = f"{textalign(plan_summary['Layers'])}"
    info_dict["模型分析摘要"]["未量化算子" + "-"*15] = f"{textalign(all_num - int8_num)}[数量占 {1.0*(all_num - int8_num)/all_num:2.1%}]"
    info_dict['模型分析摘要']['卷积算子' + "-"*18] = f"{textalign(conv_num)}[未量化{textalign(conv_num - conv_int8_num, length=4, format='right')}个]"
    info_dict['模型分析摘要']['元素级算子' + "-"*15] = f"{textalign(pw_num)}[未量化{textalign(pw_num - pw_int8_num, length=4, format='right')}个]"
    info_dict['模型分析摘要']['reformat算子'  + "-"*13] = f"{textalign(reformat_num)}[耗时占 {reformat_latency/all_latency:2.1%}]"

    
    info_dict["模型分析摘要"]["<注意>较高计算延迟算子"] = f"{np.sum(lat_norm>2.5)}个, 占比{round(np.sum(lat_norm>2.5)/len(lat) * 100, 1)}%"
    info_dict["模型分析摘要"]["<注意>较高内存操作算子"] = f"{np.sum(mem_norm>2.5)}个, 占比{round(np.sum(mem_norm>2.5)/len(mem) * 100, 1)}%"
    footnote.append("[1]: 较高计算延迟算子指计算延迟大于2.5倍算子平均计算延迟")
    footnote.append("[2]: 较高内存操作算子指内存操作大于2.5倍算子平均内存操作")


    # info_dict["模型分析摘要"]["输入数目"] = input_df.shape[0]
    # info_dict["模型分析摘要"]["输出数目"] = output_df.shape[0]
    FP32_num = np.sum(np.sum(plan.df['precision']=='FP32'))
    FP16_num = np.sum(np.sum(plan.df['precision']=='FP16'))
    INT8_num = np.sum(np.sum(plan.df['precision']=='INT8'))
    precision_info = f"FP32数： {FP32_num} 层; FP16数： {FP16_num}层; INT8数：{INT8_num}层"
    info_dict["模型分析摘要"]["数值精度"] = precision_info


    info_dict["模型分析摘要"]["模型权重大小"] = f"{plan_summary['Weights']}"
    return info_dict


def plan2info_dict_inference(plan, model_name):

    info_dict = {
        "模型名称": model_name,
        "推理指标": {},
    }
    footnote = []

    plan_summary = summary_dict(plan)

    # 模型延迟
    if plan.performance_summary:
        df_t = group_count_multi(plan.df, ['type', 'precision'])
        # breakpoint()
        df_conv = df_t[df_t['type'] =='Convolution']
        df_deconv = df_t[df_t['type'] =='Deconvolution']
        df_conv_deconv = pd.merge(df_conv, df_deconv, how='outer')
        conv_num = df_conv_deconv['count'].sum()
        conv_int8_num = df_conv_deconv[df_conv_deconv['precision'] == "INT8"]['count'].sum()

        df_pw = df_t[df_t['type'] =='PointWise']
        pw_num = df_pw['count'].sum()
        pw_int8_num = df_pw[df_pw['precision'] == "INT8"]['count'].sum()

        df_reformat = df_t[df_t['type'] =='Reformat']
        reformat_num = df_reformat['count'].sum()
        reformat_latency = plan.df[plan.df['type']=="Reformat"]['latency.avg_time'].sum()
        all_latency = plan.df['latency.avg_time'].sum()
        int8_num = plan.df[plan.df['precision'] == 'INT8']['Name'].count()
        all_num = plan.df['Name'].count()
        info_dict['推理指标']['平均延迟' + "-"*18] = textalign(f"{plan.performance_summary['Latency'][2]:.1f}") + " ms"
        info_dict['推理指标']['模型吞吐量' + "-"*15] = textalign(f"{plan.performance_summary['Throughput']:.1f}") + " qps"
        info_dict["推理指标"]["算子总数" + "-"*18] = f"{textalign(plan_summary['Layers'])}"
        info_dict["推理指标"]["未量化算子" + "-"*15] = f"{textalign(all_num - int8_num)}[数量占 {1.0*(all_num - int8_num)/all_num:2.1%}]"
        info_dict['推理指标']['卷积算子' + "-"*18] = f"{textalign(conv_num)}[{textalign(conv_num - conv_int8_num, length=3)}未量化]"
        info_dict['推理指标']['元素级算子' + "-"*15] = f"{textalign(pw_num)}[{textalign(pw_num - pw_int8_num, length=3)}未量化]"
        info_dict['推理指标']['reformat算子'  + "-"*13] = f"{textalign(reformat_num)}[耗时占比 {reformat_latency/all_latency:.1%}]"
        # breakpoint()


        # info_dict['推理指标'][
        #     'GPU使用时间[1]'] = f"{plan.performance_summary['Total GPU Compute Time']} s"
        # info_dict['推理指标'][
        #     '主机运行时间[2]'] = f"{plan.performance_summary['Total Host Walltime']} s"
        # footnote.append("[1]: GPU使用时间除了模型运行延迟外，还包括数据在主机及设备间传输、GPU任务管理等额外时间开销")
        # footnote.append("[2]: 主机运行时间除了GPU使用时间外，还包括数据、模型在主机的内存读取等非GPU时间开销")
        # 模型延迟表格
        datalist = ['Latency', 'Enqueue Time', 'H2D Latency', 'GPU Compute Time',
                    'D2H Latency']
        model_latency_dict = {k: v for k, v in plan.performance_summary.items() if
                              k in datalist}
        model_latency_df = pd.DataFrame(model_latency_dict).T.round(4)
        model_latency_df.columns = ['最小值', '最大值', '均值', '中位数', '90%',
                                    '95%', '99%']
        info_dict["备注"]=footnote
        info_dict['模型推理延迟/ms'] = model_latency_df

        lat = plan.df['latency.pct_time']
        lat_norm = lat / (100 / len(lat))
        mem = plan.df['total_io_size_bytes']
        mem_norm = mem / (sum(mem) / len(mem))

        # info_dict["推理指标"]["<警告>特高计算延迟算子[3]"] = f"{np.sum(lat_norm>5)}个, 占比{round(np.sum(lat_norm>5)/len(lat) * 100, 1)}%"
        # info_dict["推理指标"]["<警告>特高内存操作算子[4]"] = f"{np.sum(mem_norm>5)}个, 占比{round(np.sum(mem_norm>5)/len(mem) * 100, 1)}%" 
        info_dict["推理指标"]["<预警>较高计算延迟算子[1]"] = f"{np.sum(lat_norm>2.5)}个, 占比{round(np.sum(lat_norm>2.5)/len(lat) * 100, 1)}%"
        info_dict["推理指标"]["<预警>较高内存操作算子[2]"] = f"{np.sum(mem_norm>2.5)}个, 占比{round(np.sum(mem_norm>2.5)/len(mem) * 100, 1)}%"
        # info_dict["推理指标"]["<注意>极低计算延迟算子"] = f"{np.sum(lat_norm<0.2)}个, 占比{round(np.sum(lat_norm<0.2)/len(lat) * 100, 1)}%"
        # info_dict["推理指标"]["<注意>极低内存操作算子"] = f"{np.sum(mem_norm<0.2)}个, 占比{round(np.sum(mem_norm<0.2)/len(mem) * 100, 1)}%"
        footnote.append("[1]: 较高计算延迟算子指计算延迟大于2.5倍算子平均计算延迟")
        footnote.append("[2]: 较高内存操作算子指内存操作大于2.5倍算子平均内存操作")


    # 算子延迟top5
    try:
        top_5_indices = plan.df['latency.pct_time'].nlargest(5).index
        top_5_rows = plan.df.loc[
            top_5_indices, ['Name', 'type', 'latency.avg_time', 'latency.pct_time']]
        top_5_rows = top_5_rows.rename(columns={'latency.pct_time': '延迟占比%',
                                                'latency.avg_time': '平均延迟ms'})
        top_5_rows['延迟占比%'] = top_5_rows['延迟占比%'].round(2)
        top_5_rows['平均延迟ms'] = top_5_rows['平均延迟ms'].round(4)
        info_dict["算子延迟top5"] = top_5_rows

        # 内存读写top5
        top_5_indices = plan.df['total_footprint_bytes'].nlargest(5).index
        top_5_rows = plan.df.loc[
            top_5_indices, ['Name', 'type', 'total_footprint_bytes']]
        top_5_rows = top_5_rows.rename(
            columns={'total_footprint_bytes': '内存读写'})
        info_dict["内存读写top5"] = top_5_rows
    except:
        pass
    return info_dict


def plan2info_dict_device(plan, model_name):

    info_dict = {
        "模型名称": model_name,
        "设备信息": {}
    }
    # 设备信息
    # device_info = pd.DataFrame(list(plan.device_properties.items()), columns=['', ''])
    device_info = plan.device_properties
    try:
        if device_info:
            info_dict["设备信息"]['设备名称'] = device_info['Selected Device']
            info_dict["设备信息"]['计算版本'] = device_info['Compute Capability']
            info_dict["设备信息"]['处理器数量(SMs)'] = device_info['SMs']
            info_dict["设备信息"]['计算时钟速率/GHz'] = device_info[
                'Compute Clock Rate']
            info_dict["设备信息"]['设备全局内存'] = device_info['Device Global Memory']
            info_dict["设备信息"]['每个处理器的共享内存'] = device_info[
                'Shared Memory per SM']
            info_dict["设备信息"]['内存总线宽度/bit'] = device_info['Memory Bus Width']
            info_dict["设备信息"]['内存时钟速率/GHz'] = device_info['Memory Clock Rate']
    except:
        pass
    return info_dict

def info_dict2report(save_path, info_dict,
                     report_name='',
                     report_title='Model Analysis Report', add_reports2path=True, remove_md=False):
    if add_reports2path:
        report_save_path = save_path + '/reports'
    else:
        report_save_path = save_path
    os.makedirs(report_save_path, exist_ok=True)

    md_path = report_save_path + f'/{report_name}.md'
    # print("md报告生成中:" , report_name)
    with open(md_path, 'w') as md_file:
        if report_name:
            md_file.write(f'# {report_title}\n\n')
        for key, value in info_dict.items():
            md_file.write(f'## {key}\n\n')
            if isinstance(value, pd.DataFrame):
                content = value.to_markdown(index=True)
                md_file.write(content.replace('_', '\\_').replace('||', '\|\|'))
                md_file.write('\n')
            elif isinstance(value, dict):
                for k, v in value.items():
                    content = f'- {k}: **{v}**\n'
                    md_file.write(content.replace('_', '\\_').replace('||', '\|\|'))
                md_file.write('\n')
            elif isinstance(value, list):
                for v in value:
                    content = f'- {v}\n'
                    md_file.write(content.replace('_', '\\_').replace('||', '\|\|'))
                md_file.write('\n')
            else:
                md_file.write((f'- **{value}**\n').replace('_', '\\_').replace('||', '\|\|'))
                md_file.write('\n')

    # markdown to html
    html_path = report_save_path + f'/{report_name}.html'
    # print("html报告生成中:" , report_name)
    markdown2html(md_path, html_path)
    if remove_md:
        os.remove(md_path)
    