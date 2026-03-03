
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from trex import *
from trex import graphing2 as gr2

import argparse
import plotly.offline as pyo

import shutil
from parse_trtexec_log import parse_build_log, parse_profiling_log

import glob

from profile2graph import profile2graph
from plan2info_dict import *

profile_path =  '../samples/radar_bev_profiles/'
profile_files = glob.glob(os.path.join(profile_path, "*.profile.json"))
profile_file = profile_files[0]
model_name = os.path.basename(profile_file)[:-13]
profile_meta_file = os.path.join(profile_path, f'{model_name}.profile.metadata.json')
graph_file = os.path.join(profile_path, f'{model_name}.graph.json') 


plan = EnginePlan(graph_file,
                      profile_file,
                      profile_meta_file)

formatter = layer_type_formatter
display_regions = False
expand_layer_details = False

graph = gr2.to_dot(plan, formatter,
                display_layer_names=False,
                display_regions=display_regions,
                expand_layer_details=expand_layer_details,
                min_edge_width=2,
                max_edge_width=50,
                min_op_height=30,
                max_op_height=200,
                )

save_path = './'
graph_name = save_path + '/tmp/test'
svg_name = render_dot(graph, graph_name, 'svg')



# for index, layer in plan.df.iterrows():
#         latency = plan.df.loc[index, 'latency.avg_time']
#         nb_bytes = plan.df.loc[index, 'total_footprint_bytes']
#         nb_MB = nb_bytes / ( 8 * 1024**2)
#         plan.df.loc[index, 'attr.total_footprint_MB'] = nb_MB
#         plan.df.loc[index, 'attr.memory_efficiency'] = (nb_MB / 1024) / (latency / 1000)


# subfig_title_1 = '算子数量统计'
# subfig_title_2 = '算子延迟占比'
# charts = []
# df_t1 = group_count_multi(plan.df, ['type'])
# charts.append((df_t1,
#                subfig_title_1 + '<BR>Layer Count By Precision', 'count',
#                'precision'))

# layers_time_pct_by_precision = group_sum_attr(plan.df,
#                                               grouping_attr='precision',
#                                               reduced_attr='latency.pct_time')
# df_t2 = group_sum_attr_multi(plan.df,
#                               grouping_attr=['type'],
#                               reduced_attr=['latency.avg_time', 'latency.pct_time'])
# # display(layers_time_pct_by_precision)

# charts.append((df_t2,
#                subfig_title_2 + '<BR>Latency Budget By Precision',
#                'latency.pct_time', 'type'))
# # charts.append((layers_time_pct_by_precision, subfig_title_2 + '<BR>Latency Budget By Precision 2', 'latency.pct_time', 'precision'))
# # print(layer_colormap)
# # print(precision_colormap)
# merged_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR)
# merged_colormap.update(layer_colormap)
# merged_colormap.update(precision_colormap)
# # print(merged_colormap)
# fig = plotly_pie2( " Precision Statistics", charts,
#                   colormap=merged_colormap,
#                   do_show=True)


# fig = plotly_bar(
#     df=plan.df,
#     title="<BR>平均延迟ms(颜色->算子种类)",
#     values_col="latency.avg_time",
#     names_col="type",
#     orientation='v',
#     color='type',
#     use_slider=False,
#     xaxis_title=' ',
#     do_show=True,
#     colormap=layer_colormap,
#     # showlegend=True,
#     hover_data=['latency.pct_time', 'type', 'subtype', 'precision']
# )

# df_t = group_count_multi(plan.df, ['subtype', 'precision'])

# df_t = group_sum_attr_multi(plan.df,
#                           grouping_attr=['subtype', 'precision'],
#                           reduced_attr=['latency.avg_time', 'latency.pct_time'])

# # print(df_t)
# time_pct_by_type = plan.df.groupby(["type", "subtype"]).sum()[["latency.pct_time","latency.avg_time"]].reset_index()

# # display_df(time_pct_by_type)
# fig = plotly_bar(
#     df=time_pct_by_type,
#     title="<BR>Latency Budget Per Layer Type",
#     values_col="latency.avg_time",
#     names_col="subtype",
#     orientation='v',
#     color='type',
#     colormap=layer_colormap,
#     xaxis_title=' ',
#     show_axis_ticks=(True, False),
#     hover_data=['latency.pct_time','type'],
#     do_show=True)

# import plotly.express as px

# fig = px.bar(df_t, x="count", y="type", color="precision", text="count", orientation="h")
# fig.show()
