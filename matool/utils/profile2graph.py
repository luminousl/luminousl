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
from markdown2HTML import markdown2html

import glob

merged_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR)
merged_colormap.update(layer_colormap)
merged_colormap.update(precision_colormap)

def save_to_html(fig, prefix_path, filename):
    pyo.plot(fig, filename=prefix_path + '/' + filename + '.html',
             auto_open=False)

def profile2graph(fig_dict, plan, path):
    save_path = path + '/graphs'
    os.makedirs(save_path, exist_ok=True)

    df = clean_for_display(plan.df)

    for index, layer in plan.df.iterrows():
        latency = plan.df.loc[index, 'latency.avg_time']
        nb_bytes = plan.df.loc[index, 'total_footprint_bytes']
        nb_MB = nb_bytes / ( 8 * 1024**2)

        plan.df.loc[index, 'attr.total_footprint_MB'] = nb_MB
        plan.df.loc[index, 'attr.memory_efficiency'] = nb_MB / latency


    # layer_types = group_count(plan.df, 'type')
    fig_title = '算子数量统计(type+precision)'  # Layer Count By Type
    if fig_dict[fig_title]:
        df_t = group_count_multi(plan.df, ['type', 'precision'])

        fig = px_bar(
                df=df_t,
                title=fig_title + "<BR>【颜色->数值精度】",
                values_col='count',
                names_col='type',
                orientation='h',
                color='precision',
                colormap=precision_colormap,
                show_axis_ticks=(True, True),
                # showlegend=True,
                do_show=False)

        save_to_html(fig, save_path, fig_title)

    fig_title = '算子数量统计(subtype+precision)'  # Layer Count By Type
    if fig_dict[fig_title]:
        df_t = group_count_multi(plan.df, ['subtype', 'precision'])
        
        fig = px_bar(
                df=df_t,
                title=fig_title + "<BR>【颜色->数值精度】",
                values_col='count',
                names_col='subtype',
                orientation='h',
                color='precision',
                colormap=precision_colormap,
                show_axis_ticks=(True, True),
                # showlegend=True,
                do_show=False)

        save_to_html(fig, save_path, fig_title)

    fig_title = '算子延迟百分占比(type+precision)'
    if fig_dict[fig_title]:
        df_t = group_sum_attr_multi(plan.df,
                              grouping_attr=['type', 'precision'],
                              reduced_attr=['latency.avg_time', 'latency.pct_time'])

        # print(df_t)
        fig = px_bar(
                df=df_t,
                title=fig_title + "<BR>【颜色->数值精度】",
                values_col='latency.pct_time',
                names_col='type',
                orientation='h',
                color='precision',
                colormap=precision_colormap,
                show_axis_ticks=(True, True),
                hover_data=['latency.avg_time'])
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子延迟百分占比(type)'
    if fig_dict[fig_title]:
        df_t = group_sum_attr_multi(plan.df,
                              grouping_attr=['type'],
                              reduced_attr=['latency.avg_time', 'latency.pct_time'])

        # print(df_t)
        fig = px_bar(
                df=df_t,
                title=fig_title + "<BR>【颜色->数值精度】",
                values_col='latency.pct_time',
                names_col='type',
                orientation='h',
                color='type',
                colormap=layer_colormap,
                show_axis_ticks=(True, True),
                hover_data=['latency.avg_time'])
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子延迟百分占比(subtype + precision)'
    if fig_dict[fig_title]:
        df_t = group_sum_attr_multi(plan.df,
                              grouping_attr=['subtype', 'precision'],
                              reduced_attr=['latency.avg_time', 'latency.pct_time'])

        # print(df_t)
        fig = px_bar(
                df=df_t,
                title="算子subtype延迟百分占比<BR>【颜色->数值精度】",
                values_col='latency.pct_time',
                names_col='subtype',
                orientation='h',
                color='precision',
                colormap=precision_colormap,
                show_axis_ticks=(True, True),
                hover_data=['latency.avg_time'])
        save_to_html(fig, save_path, fig_title)


    fig_title = '算子延迟测算(type)'  # Latency Budget Per Layer
    if fig_dict[fig_title]:
        fig = plotly_bar(
            df=plan.df,
            title=fig_title + "<BR>平均延迟ms【颜色->算子种类】",
            values_col="latency.avg_time",
            names_col="Name",
            orientation='v',
            color='type',
            use_slider=False,
            xaxis_title=' ',
            do_show=False,
            colormap=layer_colormap,
            # showlegend=True,
            hover_data=['latency.pct_time', 'type', 'subtype', 'precision']
        )
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子延迟测算(precision)'  # Latency Budget Per Layer
    if fig_dict[fig_title]:
        fig = plotly_bar(
            df=plan.df,
            title=fig_title + "<BR>平均延迟ms【颜色->数值精度】)",
            values_col="latency.avg_time",
            names_col="Name",
            orientation='v',
            color='precision',
            use_slider=False,
            xaxis_title=' ',
            do_show=False,
            colormap=precision_colormap,
            # showlegend=True,
            hover_data=['latency.pct_time', 'type', 'subtype', 'precision']
        )
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子延迟分布直方图'  # Layer Latency Distribution
    if fig_dict[fig_title]:
        fig = plotly_hist(
            df=plan.df,
            title=fig_title + "<BR>Layer Latency Distribution",
            values_col="latency.pct_time",
            xaxis_title="Latency (ms)",
            color='type',
            colormap=layer_colormap,
            do_show=False)
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子subtype延迟测算'  # Latency Budget Per Layer Type
    if fig_dict[fig_title]:
        time_pct_by_type = plan.df.groupby(["type", "subtype"]).sum()[["latency.pct_time","latency.avg_time"]].reset_index()

        # display_df(time_pct_by_type)
        fig = plotly_bar(
            df=time_pct_by_type,
            title=fig_title + "<BR>Latency Budget Per Layer Type",
            values_col="latency.avg_time",
            names_col="subtype",
            orientation='v',
            color='type',
            colormap=layer_colormap,
            xaxis_title=' ',
            show_axis_ticks=(True, True),
            hover_data=['latency.pct_time','type'],
            do_show=False)
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子footprint测算'  # Latency Budget Per Layer
    if fig_dict[fig_title]:
        fig = plotly_bar(
            df=plan.df,
            title=fig_title + "<BR>单位: MB【颜色->算子种类】",
            values_col="attr.total_footprint_MB",
            names_col="Name",
            orientation='v',
            color='type',
            use_slider=False,
            xaxis_title=' ',
            do_show=False,
            colormap=layer_colormap,
            # showlegend=True,
            hover_data=['latency.avg_time', 'latency.pct_time', 'type', 'subtype', 'precision']
        )
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子内存效率测算'
        # Compute operations per ms (assuming one time read/write penalty)
    if fig_dict[fig_title]:
        fig = plotly_bar(
            df=plan.df,
            title=fig_title + "<BR>单位: MB/ms【颜色->算子种类】",
            values_col="attr.memory_efficiency",
            names_col="Name",
            orientation='v',
            color='type',
            use_slider=False,
            xaxis_title=' ',
            do_show=False,
            colormap=layer_colormap,
            # showlegend=True,
            hover_data=['latency.avg_time', 'latency.pct_time', 'type', 'subtype', 'precision']
        )
        save_to_html(fig, save_path, fig_title)

    try:
        fig_title = '算子延迟树状图-延迟vs延迟'  # Treemap Of Layer Latencies (Size & Color Indicate Latency)
        if fig_dict[fig_title]:
            fig = px.treemap(
                plan.df,
                path=['type', 'Name'],
                values='latency.pct_time',
                title=fig_title + "<BR>Treemap Of Layer Latencies (Size & Color Indicate Latency)",
                color='latency.pct_time')
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            # fig.show()
            save_to_html(fig, save_path, fig_title)

        fig_title = '算子延迟树状图-延迟vs大小'  # Treemap Of Layer Latencies (Size Indicates Latency. Color Indicates Activations Size)
        if fig_dict[fig_title]:
            fig = px.treemap(
                plan.df,
                path=['type', 'Name'],
                values='latency.pct_time',
                title=fig_title + "<BR>Treemap Of Layer Latencies (Size Indicates Latency. Color Indicates Activations Size)",
                color='total_io_size_bytes')
            fig.update_traces(root_color="white")
            fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            # fig.show()
            save_to_html(fig, save_path, fig_title)
    except:
        pass

    fig_title = '算子参数量统计图'  # Weights Sizes Per Layer
    if fig_dict[fig_title]:
        fig = plotly_bar2(
            plan.df,
            fig_title + "<BR>Weights Sizes Per Layer",
            "weights_size", "Name",
            color='type',
            colormap=layer_colormap,
            xaxis_title=' ',
            do_show=False)
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子输出图大小统计图'  # Activations Sizes Per Layer
    if fig_dict[fig_title]:
        fig = plotly_bar2(
            plan.df,
            fig_title + "<BR>Activations Sizes Per Layer",
            "total_io_size_bytes",
            "Name",
            color='type',
            colormap=layer_colormap,
            xaxis_title=' ',
            do_show=False)
        save_to_html(fig, save_path, fig_title)

    fig_title = '算子输出图大小分布直方图'  # Layer Activations Sizes Distribution
    if fig_dict[fig_title]:
        fig = plotly_hist(
            plan.df,
            fig_title + "<BR>Layer Activations Sizes Distribution",
            "total_io_size_bytes",
            "Size (bytes)",
            color='type',
            colormap=layer_colormap,
            do_show=False)

        plan.df["total_io_size_bytes"].describe()
        save_to_html(fig, save_path, fig_title)


    fig_title = '算子分析饼图'
    if fig_dict[fig_title]:
        subfig_title_1 = '算子精度统计'
        subfig_title_2 = '算子延迟占比'
        charts = []
        df_t1 = group_count_multi(plan.df, ['precision'])
        charts.append((df_t1,
                       subfig_title_1 + '<BR>Layer Count By Precision', 'count',
                       'precision'))
        df_t2 = group_sum_attr_multi(plan.df,
                                      grouping_attr=['type'],
                                      reduced_attr=['latency.avg_time', 'latency.pct_time'])
        charts.append((df_t2,
                       subfig_title_2 + '<BR>Latency Budget By Precision',
                       'latency.pct_time', 'type'))
        fig = plotly_pie2(fig_title, charts,
                          colormap=merged_colormap,
                          do_show=False)
        save_to_html(fig, save_path, fig_title)

    ### Convolutions
    try:
        convs = plan.get_layers_by_type('Convolution')

        fig_title = '卷积算子FLOPS测算'
        if fig_dict[fig_title]:
            fig = plotly_bar2(
                convs,
                fig_title + "<BR>Convolution macs【颜色->计算延迟】",
                "attr.macs", "Name",
                color='latency.pct_time',
                # colormap=precision_colormap,
                xaxis_title=' ',
                do_show=False)
            save_to_html(fig, save_path, fig_title)

        fig_title = '卷积算子计算强度测算'
        if fig_dict[fig_title]:
            fig = plotly_bar2(
                convs,
                fig_title + "<BR>Convolution arithmetic intensity 【颜色->输出大小】",
                "attr.arithmetic_intensity", "Name",
                color='latency.pct_time',
                # colormap=precision_colormap,
                xaxis_title=' ',
                do_show=False)
            save_to_html(fig, save_path, fig_title)

        fig_title = '卷积算子io测算'
        if fig_dict[fig_title]:
            fig = plotly_bar2(
                convs,
                fig_title + "<BR>Convolution Data Sizes 【颜色->计算延迟】",
                "total_io_size_bytes",
                "Name",
                color='latency.pct_time',
                xaxis_title=' ',
                do_show=False)
            save_to_html(fig, save_path, fig_title)

        fig_title = '卷积算子内存效率'
        # Memory accesses per ms (assuming one time read/write penalty)
        if fig_dict[fig_title]:
            fig = plotly_bar2(
                convs,
                fig_title + "<BR>Convolution Memory Efficiency<BR>【颜色->计算延迟】",
                "attr.memory_efficiency",
                "Name",
                color='latency.pct_time',
                xaxis_title=' ',
                do_show=False)
            save_to_html(fig, save_path, fig_title)

        fig_title = '卷积算子计算效率'
        # Compute operations per ms (assuming one time read/write penalty)
        if fig_dict[fig_title]:
            fig = plotly_bar2(
                convs,
                fig_title + "<BR>Convolution Compute Efficiency<BR>【颜色->计算延迟】",
                "attr.compute_efficiency",
                "Name",
                color='latency.pct_time',
                xaxis_title=' ',
                do_show=False)
            save_to_html(fig, save_path, fig_title)

        ### Statistics

        convs = plan.get_layers_by_type('Convolution')

        fig_title = '卷积算子饼图-算子种类'
        if fig_dict[fig_title]:
            charts = []
            convs_count_by_type = group_count(convs, 'subtype')
            charts.append((convs_count_by_type, 'Count', 'count', 'subtype'))

            convs_time_pct_by_type = group_sum_attr(convs, grouping_attr='subtype',
                                                    reduced_attr='latency.pct_time')
            charts.append((convs_time_pct_by_type, '% Latency Budget',
                           'latency.pct_time', 'subtype'))
            fig = plotly_pie2(fig_title + " Convolutions Statistics (Subtype)",
                              charts,
                              do_show=False)
            save_to_html(fig, save_path, fig_title)

        fig_title = '卷积算子饼图-分组参数'
        if fig_dict[fig_title]:
            charts = []
            convs_count_by_group_size = group_count(convs, 'attr.groups')
            charts.append(
                (convs_count_by_group_size, 'Count', 'count', 'attr.groups'))

            convs_time_pct_by_grp_size = group_sum_attr(convs,
                                                        grouping_attr='attr.groups',
                                                        reduced_attr='latency.pct_time')
            charts.append((convs_time_pct_by_grp_size, '% Latency Budget',
                           'latency.pct_time', 'attr.groups'))
            fig = plotly_pie2(
                fig_title + " Convolutions Statistics (Number of Groups)", charts,
                do_show=False)
            save_to_html(fig, save_path, fig_title)

        fig_title = '卷积算子饼图-卷积核'
        if fig_dict[fig_title]:
            charts = []
            convs_count_by_kernel_shape = group_count(convs, 'attr.kernel')
            charts.append(
                (convs_count_by_kernel_shape, 'Count', 'count', 'attr.kernel'))

            convs_time_pct_by_kernel_shape = group_sum_attr(convs,
                                                            grouping_attr='attr.kernel',
                                                            reduced_attr='latency.pct_time')
            charts.append((convs_time_pct_by_kernel_shape, '% Latency Budget',
                           'latency.pct_time', 'attr.kernel'))
            fig = plotly_pie2(fig_title + " Convolutions Statistics (Kernel Size)",
                              charts,
                              do_show=False)
            save_to_html(fig, save_path, fig_title)

        fig_title = '卷积算子饼图-数值精度'
        if fig_dict[fig_title]:
            charts = []
            convs_count_by_precision = group_count(convs, 'precision')
            charts.append((convs_count_by_precision, 'Count', 'count', 'precision'))

            convs_time_pct_by_precision = group_sum_attr(convs,
                                                         grouping_attr='precision',
                                                         reduced_attr='latency.pct_time')
            charts.append((convs_time_pct_by_precision, '% Latency Budget',
                           'latency.pct_time', 'precision'))

            fig = plotly_pie2(fig_title + " Convolutions Statistics (Precision)",
                              charts, colormap=precision_colormap,
                              do_show=False)
            save_to_html(fig, save_path, fig_title)

        fig_title = '卷积算子延迟统计'
        # Memory accesses per ms (assuming one time read/write penalty)
        if fig_dict[fig_title]:
            convs = plan.get_layers_by_type('Convolution')
            fig = plotly_bar2(
                convs,
                fig_title + "<BR>Convolution latency【颜色->计算延迟】",
                "latency.pct_time",
                "Name",
                color='latency.pct_time',
                xaxis_title=' ',
                do_show=False)
            save_to_html(fig, save_path, fig_title)
    except:
        pass



