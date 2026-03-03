# +
import sys
# sys.path.append('/media/Projects/ModelToolbox')
# sys.path.append('/media/Projects/MATool/utils')
sys.path.append('/media/Projects/MATool')

from trex import *
import plotly.express as px
import os
import yaml
import pandas as pd
from utils.plan2info_dict import info_dict2report
import random

def get_engine_plan(profile_path):
    all_files = os.listdir(profile_path)
#     print(all_files)
    file_graph = [file for file in all_files if file.endswith("graph.json")]
    if len(file_graph) == 1:
        engine_name = os.path.join(profile_path, file_graph[0][:-11])
        plan = EnginePlan(f'{engine_name}.graph.json', 
                  f'{engine_name}.profile.json', 
                  f'{engine_name}.profile.metadata.json')
        return plan
    else:
        print(f"Multiple or No *.graph.json Path in {profile_path}")
    return None


# plan = get_engine_plan("/media/Projects/ModelBoard/database/online_models/20240711_orinX_6090_27.11U/multitask_profile")

def get_info_from_plan(plan):
    info_dict = {}
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
    info_dict['Latency'] = round(plan.performance_summary['Latency'][2], 1)
    info_dict['△Latency'] = round((random.random()-0.5), 1)
    info_dict['DLA'] = True if plan.df['type'].isin(['DLA']).any() else False
    info_dict["#OP"] = all_num
    info_dict["#float_OP"] = all_num - int8_num
    info_dict["%float_OP"] = 1.0*(all_num - int8_num)/all_num
    info_dict['#CONV'] = conv_num
    info_dict['#float_CONV'] = conv_num - conv_int8_num
    info_dict['#PWN'] = pw_num
    info_dict['#float_PWN'] = pw_num - pw_int8_num
    info_dict['#REFORMAT'] = reformat_num
    info_dict['%REFORMAT'] = 1.0*reformat_latency/all_latency
    return info_dict

def info2str(info_dict):
    str_info_dict = {}
    str_info_dict['Lat'] = f"{info_dict['Latency']}"
    str_info_dict['△Lat'] = f"<p align='left' style='color:red'><b>↑+{info_dict['△Latency']}</b></p>" \
        if info_dict['△Latency'] > 0 \
        else f"<p align='left' style='color:green'><b>↓{info_dict['△Latency']}</b></p>" \
        if info_dict['△Latency'] < 0 \
        else None
    str_info_dict['DLA'] = '✅' if info_dict['DLA'] else None
    str_info_dict["#OP"] = f"{info_dict['#OP']}"
    str_info_dict["#f_OP"] = f"<p align='right'>{info_dict['#float_OP']}</p>" \
        if info_dict['DLA'] or info_dict['%float_OP'] < 0.4 \
        else f"<p align='right' style='color:red'><b>{info_dict['#float_OP']}</b></p>"
    str_info_dict["%f_OP"] = f"<p align='right'>{info_dict['%float_OP']:2.1%}</p>" \
        if info_dict['DLA'] or info_dict['%float_OP'] < 0.4 \
        else f"<p align='right' style='color:red'><b>{info_dict['%float_OP']:2.1%}</b></p>"
    str_info_dict['#CONV'] = f"{info_dict['#CONV']}"
    str_info_dict['#f_CONV'] = f"<p align='right'>{info_dict['#float_CONV']}</p>" \
        if info_dict['#float_CONV']==0 or info_dict['#CONV'] / info_dict['#float_CONV'] > 5 \
        else f"<p align='right' style='color:red'><b>{info_dict['#float_CONV']}</b></p>"
    str_info_dict['#PWN'] = f"<p align='right'>{info_dict['#PWN']}</p>" \
        if info_dict['#PWN']==0 or info_dict['DLA'] or info_dict['#CONV'] / info_dict['#PWN'] > 3 \
        else f"<p align='right' style='color:red'><b>{info_dict['#PWN']}</b></p>"
    str_info_dict['#f_PWN'] = f"<p align='right'>{info_dict['#float_PWN']}</p>" \
        if info_dict['#float_PWN']==0 or info_dict['DLA'] or info_dict['#CONV'] / info_dict['#float_PWN'] > 6 \
        else f"<p align='right' style='color:red'><b>{info_dict['#float_PWN']}</b></p>"
    str_info_dict['#REF'] = f"<p align='right'>{info_dict['#REFORMAT']}</p>" \
        if info_dict['DLA'] or info_dict['#OP']/info_dict['#REFORMAT'] > 4 \
        else f"<p align='right' style='color:red'><b>{info_dict['#REFORMAT']}</b></p>"
    str_info_dict['%REF'] = f"<p align='right'>{info_dict['%REFORMAT']:.1%}</p>" \
        if info_dict['%REFORMAT'] < 0.09 \
        else f"<p align='right' style='color:red'><b>{info_dict['%REFORMAT']:.1%}</b></p>"
        
    return str_info_dict

yaml_path = "/media/Projects/zpilot_profiler/tools/config.yaml"
with open(yaml_path,'r') as f:
    config=yaml.safe_load(f)

info_dicts = {}
profile_directory = "/media/Projects/ModelBoard/database/online_models/20240711_orinX_6090_27.11U/"
for group, scenes in config.items():
    for scene in scenes.keys():
        plan_path = f"{profile_directory}/{scene}_profile"
        if os.path.exists(plan_path):
            try:
                plan = get_engine_plan(plan_path)
                info_dicts[scene] = get_info_from_plan(plan)
            except:
                pass

str_info_dicts = {k: info2str(v) for k,v in info_dicts.items()}


            
df = pd.DataFrame(str_info_dicts).T

info_dict2report('tmp/', {"27.11U": df}, report_name='report_test', report_title='', add_reports2path=False)

# +

import plotly.graph_objects as go
import plotly.offline as pyo
# 创建数据
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]

# 创建折线图
fig = go.Figure(data=go.Scatter(x=x, y=y))

# 设置图表布局
fig.update_layout(title='Simple Line Chart', xaxis_title='X Axis', yaxis_title='Y Axis')

# 显示图表
pyo.plot(fig, filename='tmp/test.html', auto_open=False)
# -

df = pd.DataFrame(info_dicts).T
df_lat1 = df.loc[:, ['Latency']]
df_lat2 = df.loc[:, ['Latency']]
df_lat3 = df.loc[:, ['Latency']]
df_lat1['version'] = 27.11
df_lat2['version'] = 26.1
df_lat3['version'] = 29.1

df_fig = pd.concat([df_lat1, df_lat2, df_lat3]).reset_index()
df_fig.columns = ["ModelName", "Latency", "VersionNumber"]

df_fig

# +
import plotly.express as px 
  
df = px.data.tips() 
  
plot = px.line(df_fig, x = 'VersionNumber',  
               y = 'Latency', color='ModelName', markers=True, line_dash='ModelName') 

layout = go.Layout(
        title={
            'text': "版本模型耗时变化",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'bottom',
            'font':{'size': 24}
            },
#         width=size[0],
#         height=size[1],
        xaxis={
            'visible': True,
            'showticklabels': True,
            'showline': True,
#             'title': x_title,
#             'gridcolor': "Black",
        },
        yaxis={
            'visible': True,
            'showticklabels': True,
            'showline': True,
#             'title': y_title,
#             'gridcolor': y_grid,
            'tickformat': "%{y:$.2f}"},
        plot_bgcolor='rgba(0,0,0,0)',
        legend={
        'yanchor': "top",
        'y': 0.8,
        'xanchor': "right",
        'x': 1.08,
        'font':{'size': 20}
        },
)


plot.update_layout(layout)
pyo.plot(plot, filename='tmp/test.html', auto_open=False)
# -

help(px.line)

help(px.scatter)
