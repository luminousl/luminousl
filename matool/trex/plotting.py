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
This file contains pyplot plotting wrappers.
"""


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd
import math
from .notebook import display_df
from .misc import stack_dataframes
import numpy as np


NVDA_GREEN = '#76b900'
UNKNOWN_KEY_COLOR = 'gray'
GRID_COLOR = 'rgba(114, 179, 24, 0.3)'


# pallete = px.colors.qualitative.G10
# https://medialab.github.io/iwanthue/
default_pallete = [
    "#a11350",
    "#008619",
    "#4064ec",
    "#ffb519",
    "#8f1a8e",
    "#b2b200",
    "#64b0ff",
    "#e46d00",
    "#02d2ba",
    "#ef393d",
    "#f1b0f7",
    "#7e4401",
    UNKNOWN_KEY_COLOR]


# Set a color for each precision datatype.
precision_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, {
    'INT8':  NVDA_GREEN,
    'FP32':  'red',
    'FP16':  'orange',
    'INT32': 'lightgray',
})


# Set a color for each layer type.
layer_colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, {
    # https://htmlcolorcodes.com/
    "Convolution":    "#4682B4", # SteelBlue
    "Deconvolution":  "#7B68EE", # MediumSlateBlue
    "ConvActPool":    "#6495ED", # CornflowerBlue
    "MatrixMultiply": "#1E90FF", # DodgerBlue
    "Reformat":       "#00FFFF", # Cyan
    "Shuffle":        "#BC8F8F", # RosyBrown
    "Slice":          "#FFA500", # Orange
    "Scale":          "#8FBC8B", # DarkSeaGreen
    "Quantize":       "#6B8E23", # OliveDrab
    "Pooling":        "#3CB371", # MediumSeaGreen
    "PluginV2":       "#C71585", # MediumVioletRed
    "PointWise":      "#9ACD32", # YellowGreen
    "ElementWise":    "#9ACD32", # YellowGreen
    "Reduce":         "#90EE90", # LightGreen
    "SoftMax":        "#DA70D6", # Orchid
    "Myelin":         "#800080", # Purple
    "correlation":    "#4682B4",
    "maxpool":        "#3CB371",
    "avgpool":        "#3CB371",
    "gemm":           "#1E90FF",
    "kgen":           "#800080",
    "deconv":         "#7B68EE",
})


def _categorical_colormap(df: pd.DataFrame, color_col:str):
    # Protect against index-out-of-range
    max_idx = len(default_pallete) - 1
    colormap = {category:
        default_pallete[min(i,max_idx)] for i,category in enumerate(set(df[color_col]))}
    return colormap


def _create_categorical_marker(
    df: pd.DataFrame,
    color_col: str,
    colormap: Dict=None
):
    if not colormap:
        colormap = _categorical_colormap(df, color_col)
    # Make colormap robust to unknown keys
    colormap = defaultdict(lambda: UNKNOWN_KEY_COLOR, colormap)
    color_list = [colormap[key] for key in df[color_col]]
    marker = dict(color=color_list)
    return marker


def trex_base_layout(fig, gridcolor=None):
    "White background with colored grid"
    gridcolor = gridcolor or GRID_COLOR
    fig.update_layout({
        'xaxis': {'gridcolor': gridcolor},
        'yaxis': {'gridcolor': gridcolor},
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })


def create_layout(
    title: str,
    size:Tuple,
    x_title: str,
    y_title: str,
    orientation: str,
    show_axis_ticks: Tuple[bool]=(True,True),
    legend_title: str = None
):
    y_grid = None if orientation == 'h' else GRID_COLOR
    x_grid = None if orientation == 'v' else GRID_COLOR

    if orientation == 'h':
        x_title, y_title = y_title, x_title

    top_right = {
        'yanchor': "top",
        'y': 0.99,
        'xanchor': "right",
        'x': 0.99}

    layout = go.Layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'bottom',
            'font':{'size': 24}
            },
        width=size[0],
        height=size[1],
        xaxis={
            'visible': True,
            'showticklabels': show_axis_ticks[0],
            'title': x_title,
            'gridcolor': x_grid,},
        yaxis={
            'visible': True,
            'showticklabels': show_axis_ticks[1],
            'title': y_title,
            'gridcolor': y_grid,
            'tickformat': "%{y:$.2f}"},
        plot_bgcolor='rgba(0,0,0,0)',
        legend={
        'yanchor': "top",
        'y': 0.99,
        'xanchor': "right",
        'x': 0.99,
        'font':{'size': 20}
        },
    )


    return layout

def px_bar(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    names_col: str,
    orientation: str='v',
    color: str=None,
    size: Tuple=(None, None),
    use_slider: bool=False,
    colormap: Dict=None,
    show_axis_ticks:Tuple=(False, True),
    showlegend: bool=False,
    xaxis_title: str=None,
    yaxis_title: str=None,
    text: str=None,
    hover_data: List=None,
    do_show: bool=False,
):
    def categorical_color(df, color) -> bool:
        if df[color].dtype in [float, int]:
            return False
        return True

    layout = create_layout(
        title,
        size,
        x_title=xaxis_title or names_col,
        y_title=yaxis_title or values_col,
        orientation=orientation,
        show_axis_ticks=show_axis_ticks,)

    if orientation == 'v':
        x, y = (names_col, values_col)
    else:
        x, y = (values_col, names_col)

    if text:
        bartext = text
    else:
        bartext = df[values_col]
    fig = go.Figure(layout=layout)

    fig = px.bar(data_frame=df, 
        x=df[x], y=df[y],
        color=color,
        orientation=orientation,
        text=bartext,
        title=title,
        text_auto=".3f",
        hover_data=hover_data,
        color_discrete_map=colormap,
      )
    fig.update_layout(layout)

    if use_slider:
        fig.update_xaxes(rangeslider_visible=True)
    if do_show:
        fig.show()

    return fig

def plotly_bar(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    names_col: str,
    orientation: str='v',
    color: str=None,
    size: Tuple=(None, None),
    use_slider: bool=False,
    colormap: Dict=None,
    show_axis_ticks:Tuple=(False, True),
    showlegend: bool=False,
    xaxis_title: str=None,
    yaxis_title: str=None,
    hover_data: List=None,
    do_show: bool=True,
):
    def categorical_color(df, color) -> bool:
        if df[color].dtype in [float, int]:
            return False
        return True

    def add_bar(df, name, color, colormap, hover_data):
        if orientation == 'v':
            x, y = (names_col, values_col)
            hover_txt = f"{x}: " + "%{x}" \
                            + f"<br>{y}: " + "%{y:.4f}"
        else:
            x, y = (values_col, names_col)
            hover_txt = f"{x}: " + "%{x:.4f}" + f"<br>{y}: " + "%{y}"
        customdata_list = []
        customdata_index = 0
        if hover_data:
            for customdata_name in hover_data:
                hover_txt += f"<br>{customdata_name}: " + "%{customdata[" + str(customdata_index)+ "]}"
                customdata_list.append(df[customdata_name])
                customdata_index += 1
        customdata = np.stack(customdata_list, axis=-1) if customdata_list else []

        is_categorical = False
        colorbar = None
        if color is not None:
            assert isinstance(color, str)
            if not categorical_color(df, color):
                colorbar = dict(title=color)
                color = df[color]
            else:
                is_categorical = True

        if not is_categorical:
            marker = dict(color=color, colorbar=colorbar)
        else:
            marker = _create_categorical_marker(df, color, colormap)
        texttemplate = "%{value:.4f}" if df[values_col].dtype == float else '%{value}'
        
        
        bar = go.Bar(
            x=df[x], y=df[y],
            orientation=orientation,
            marker=marker,
            text=df[values_col],
            texttemplate=texttemplate,
            hovertemplate=hover_txt,
            name=name,
            customdata=customdata,
            showlegend=False)
        return bar

    layout = create_layout(
        title,
        size,
        x_title=xaxis_title or names_col,
        y_title=yaxis_title or values_col,
        orientation=orientation,
        show_axis_ticks=show_axis_ticks,)
    
    def add_bar_lengend(df, name, color, colormap, hover_data):
        if orientation == 'v':
            x, y = (names_col, values_col)
            hover_txt = f"{x}: " + "%{x}" + f"<br>{y}: " + "%{y:.4f}"
        else:
            x, y = (values_col, names_col)
            hover_txt = f"{x}: " + "%{x:.4f}" + f"<br>{y}: " + "%{y}"

        customdata_list = []
        customdata_index = 0
        if hover_data:
            for customdata_name in hover_data:
                hover_txt += f"<br>{customdata_name}: " + "%{customdata[" + str(customdata_index)+ "]}"
                customdata_list.append(df[customdata_name])
                customdata_index += 1
        
        customdata = np.stack(customdata_list, axis=-1) if customdata_list else []
        is_categorical = False
        colorbar = None
        if color is not None:
            assert isinstance(color, str)
            if not categorical_color(df, color):
                colorbar = dict(title=color)
                color = df[color]
            else:
                is_categorical = True

        if not is_categorical:
            marker = dict(color=color, colorbar=colorbar)
        else:
            marker = _create_categorical_marker(df, color, colormap)
        texttemplate = "%{value:.4f}" if df[values_col].dtype == float else '%{value}'
        
        type_bars = {}
        for i in range(len(df[x])):
            current_type = df['type'][i]

            # If the type is encountered for the first time, create a new dictionary entry
            if current_type not in type_bars:
                type_bars[current_type] = {
                    'x': [],
                    'y': [],
                    'text': [],
                    'name': current_type,
                    'marker': {'color': []},
                    'hovertemplate': hover_txt,
                    'showlegend': True,
                    'legendgroup': current_type
                }

            # Append data for the current bar to the corresponding type entry
            type_bars[current_type]['x'].append(df[x][i])
            type_bars[current_type]['y'].append(df[y][i])
            # type_bars[current_type]['text'].append(df[values_col][i])
            type_bars[current_type]['text'].append(df['type'][i])
            type_bars[current_type]['marker']['color'].append(marker['color'][i])
            type_bars[current_type]['customdata'] = customdata[i]

        # Convert the dictionary entries to go.Bar traces
        bar_traces = [go.Bar(**data) for data in type_bars.values()]

        return bar_traces

    add_bar = add_bar_lengend if showlegend else add_bar
    
    fig = go.Figure(layout=layout)

    if not isinstance(df, Dict):
        df_dict = {None: df}
    else:
        df_dict = df
    for name, df in df_dict.items():
        bar = add_bar(df, name, color, colormap, hover_data=hover_data)
        fig.add_traces(bar)

    if use_slider:
        fig.update_xaxes(rangeslider_visible=True)
    if do_show:
        fig.show()

    return fig

def plotly_bar6(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    names_col: str,
    orientation: str='v',
    color: str=None,
    size: Tuple=(None, None),
    use_slider: bool=False,
    colormap: Dict=None,
    show_axis_ticks:Tuple=(False, True),
    showlegend: bool=False,
    xaxis_title: str=None,
    yaxis_title: str=None,
    hover_data: List=None,
    do_show: bool=False,
):
    def categorical_color(df, color) -> bool:
        if df[color].dtype in [float, int]:
            return False
        return True

    def add_bar(df, name, color, colormap, showlegend, hover_data):
        if orientation == 'v':
            x, y = (names_col, values_col)
            hover_txt = f"{x}: " + "%{x}" \
                            + f"<br>{y}: " + "%{y:.4f}"
        else:
            x, y = (values_col, names_col)
            hover_txt = f"{x}: " + "%{x:.4f}" + f"<br>{y}: " + "%{y}"
        customdata_list = []
        customdata_index = 0
        if hover_data:
            for customdata_name in hover_data:
                hover_txt += f"<br>{customdata_name}: " + "%{customdata[" + str(customdata_index)+ "]}"
                customdata_list.append(df[customdata_name])
                customdata_index += 1

        is_categorical = False
        colorbar = None
        if color is not None:
            assert isinstance(color, str)
            if not categorical_color(df, color):
                colorbar = dict(title=color)
                color = df[color]
            else:
                is_categorical = True

        if not is_categorical:
            marker = dict(color=color, colorbar=colorbar)
        else:
            marker = _create_categorical_marker(df, color, colormap)
        texttemplate = "%{value:.4f}" if df[values_col].dtype == float else '%{value}'
        
        
        bar = go.Bar(
            x=df[x], y=df[y],
            orientation=orientation,
            marker=marker,
            text=df[values_col],
            texttemplate=texttemplate,
            hovertemplate=hover_txt,
            name=name,
            customdata=np.stack(customdata_list, axis=-1),
            showlegend=showlegend)
        return bar

    layout = create_layout(
        title,
        size,
        x_title=xaxis_title or names_col,
        y_title=yaxis_title or values_col,
        orientation=orientation,
        show_axis_ticks=show_axis_ticks,)
    
    
    fig = go.Figure(layout=layout)

    if not isinstance(df, Dict):
        df_dict = {None: df}
    else:
        df_dict = df
    for name, df in df_dict.items():
        bar = add_bar(df, name, color, colormap, showlegend=showlegend, hover_data=hover_data)
        fig.add_traces(bar)

    if use_slider:
        fig.update_xaxes(rangeslider_visible=True)
    if do_show:
        fig.show()

    return fig


def plotly_bar2(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    names_col: str,
    orientation: str='v',
    color: str=None,
    size: Tuple=(None, None),
    use_slider: bool=False,
    colormap: Dict=None,
    show_axis_ticks:Tuple=(False, True),
    showlegend: bool=False,
    xaxis_title: str=None,
    yaxis_title: str=None,
    do_show: bool=True,
):
    def categorical_color(df, color) -> bool:
        if df[color].dtype in [float, int]:
            return False
        return True

    def add_bar(df, name, color, colormap, showlegend):
        if orientation == 'v':
            x, y = (names_col, values_col)
            hover_txt = f"{x}: " + "%{x}"  + f"<br>{y}: " + "%{y:.4f}" 
        else:
            x, y = (values_col, names_col)
            hover_txt = f"{x}: " + "%{x:.4f}" + f"<br>{y}: " + "%{y}"

        is_categorical = False
        colorbar = None
        if color is not None:
            assert isinstance(color, str)
            if not categorical_color(df, color):
                colorbar = dict(title=color)
                color = df[color]
            else:
                is_categorical = True

        if not is_categorical:
            marker = dict(color=color, colorbar=colorbar)
        else:
            marker = _create_categorical_marker(df, color, colormap)
        texttemplate = "%{value:.4f}" if df[values_col].dtype == float else '%{value}'
        
        
        bar = go.Bar(
            x=df[x], y=df[y],
            orientation=orientation,
            marker=marker,
            text=df[values_col],
            texttemplate=texttemplate,
            hovertemplate=hover_txt,
            name=name,
            # customdata=np.stack((df['type'], df['latency.pct_time']),axis=-1),
            showlegend=showlegend)
        return bar

    layout = create_layout(
        title,
        size,
        x_title=xaxis_title or names_col,
        y_title=yaxis_title or values_col,
        orientation=orientation,
        show_axis_ticks=show_axis_ticks,)
    
    
    fig = go.Figure(layout=layout)

    if not isinstance(df, Dict):
        df_dict = {None: df}
    else:
        df_dict = df
    for name, df in df_dict.items():
        bar = add_bar(df, name, color, colormap, showlegend=showlegend)
        fig.add_traces(bar)

    if use_slider:
        fig.update_xaxes(rangeslider_visible=True)
    if do_show:
        fig.show()

    return fig


def plotly_bar3(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    names_col: str,
    orientation: str='v',
    color: str=None,
    size: Tuple=(None, None),
    use_slider: bool=False,
    colormap: Dict=None,
    show_axis_ticks:Tuple=(False, True),
    showlegend: bool=False,
    xaxis_title: str=None,
    yaxis_title: str=None,
    do_show: bool=True,
    xstandoff: int=150,
):
    def categorical_color(df, color) -> bool:
        if df[color].dtype in [float, int]:
            return False
        return True

    def add_bar(df, name, color, colormap, showlegend):
        if orientation == 'v':
            x, y = (names_col, values_col)
            hover_txt = f"{x}: " + "%{x}" + f"<br>{y}: " + "%{y:.4f}"
        else:
            x, y = (values_col, names_col)
            hover_txt = f"{x}: " + "%{x:.4f}" + f"<br>{y}: " + "%{y}"

        is_categorical = False
        colorbar = None
        if color is not None:
            assert isinstance(color, str)
            if not categorical_color(df, color):
                colorbar = dict(title=color)
                color = df[color]
            else:
                is_categorical = True

        if not is_categorical:
            marker = dict(color=color, colorbar=colorbar)
        else:
            marker = _create_categorical_marker(df, color, colormap)

        texttemplate = "%{value:.4f}" if df[values_col].dtype == float else '%{value}'
        bar = go.Bar(
            x=df[x], y=df[y],
            orientation=orientation,
            marker=marker,
            text=df[values_col],
            texttemplate=texttemplate,
            hovertemplate=hover_txt,
            name=name,
            showlegend=showlegend)
        return bar

    layout = create_layout(
        title,
        size,
        x_title=xaxis_title or names_col,
        y_title=yaxis_title or values_col,
        orientation=orientation,
        show_axis_ticks=show_axis_ticks,)
    
#     # 将x轴标题移到下方
#     layout.update_layout(
#         xaxis_title=dict(
#             y=0.0,  # 设置x轴标题的垂直位置为0.0，表示相对于图表底部
#             xanchor='center',  # 设置水平位置相对于标题的锚点为中心
#         )
#     )
    
    fig = go.Figure(layout=layout)
    
    # 将x轴标题移到下方
    fig.update_xaxes(
        title=dict(
            text=xaxis_title or names_col,
            standoff=xstandoff,  # 设置标题与x轴的距离，以避免重叠
        )
    )

    if not isinstance(df, Dict):
        df_dict = {None: df}
    else:
        df_dict = df

    for name, df in df_dict.items():
        bar = add_bar(df, name, color, colormap, showlegend=showlegend)
        fig.add_traces(bar)

    if use_slider:
        fig.update_xaxes(rangeslider_visible=True)
    if do_show:
        fig.show()
    return fig

def plotly_bar4(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    names_col: str,
    orientation: str='v',
    color: str=None,
    size: Tuple=(None, None),
    use_slider: bool=False,
    colormap: Dict=None,
    show_axis_ticks:Tuple=(False, True),
    showlegend: bool=False,
    xaxis_title: str=None,
    yaxis_title: str=None,
    do_show: bool=True,
):
    def categorical_color(df, color) -> bool:
        if df[color].dtype in [float, int]:
            return False
        return True

    def add_bar(df, name, color, colormap, showlegend):
        if orientation == 'v':
            x, y = (names_col, values_col)
            hover_txt = f"{x}: " + "%{x}" + f"<br>{y}: " + "%{y:.4f}" + "<br>Type: " + "%{text}"
        else:
            x, y = (values_col, names_col)
            hover_txt = f"{x}: " + "%{x:.4f}" + f"<br>{y}: " + "%{y}" + "<br>Type: " + "%{text}"

        is_categorical = False
        colorbar = None
        if color is not None:
            assert isinstance(color, str)
            if not categorical_color(df, color):
                colorbar = dict(title=color)
                color = df[color]
            else:
                is_categorical = True

        if not is_categorical:
            marker = dict(color=color, colorbar=colorbar)
        else:
            marker = _create_categorical_marker(df, color, colormap)
        texttemplate = "%{value:.4f}" if df[values_col].dtype == float else '%{value}'
        
        type_bars = {}
        for i in range(len(df[x])):
            current_type = df['type'][i]

            # If the type is encountered for the first time, create a new dictionary entry
            if current_type not in type_bars:
                type_bars[current_type] = {
                    'x': [],
                    'y': [],
                    'text': [],
                    'name': current_type,
                    'marker': {'color': []},
                    'hovertemplate': hover_txt,
                    'showlegend': showlegend,
                    'legendgroup': current_type,
                    'customdata': [current_type]
                }

            # Append data for the current bar to the corresponding type entry
            type_bars[current_type]['x'].append(df[x][i])
            type_bars[current_type]['y'].append(df[y][i])
            # type_bars[current_type]['text'].append(df[values_col][i])
            type_bars[current_type]['text'].append(df['type'][i])
            type_bars[current_type]['marker']['color'].append(marker['color'][i])

        # Convert the dictionary entries to go.Bar traces
        bar_traces = [go.Bar(**data) for data in type_bars.values()]

        return bar_traces

    layout = create_layout(
        title,
        size,
        x_title=xaxis_title or names_col,
        y_title=yaxis_title or values_col,
        orientation=orientation,
        show_axis_ticks=show_axis_ticks,)
    
    
    fig = go.Figure(layout=layout)

    if not isinstance(df, Dict):
        df_dict = {None: df}
    else:
        df_dict = df
    for name, df in df_dict.items():
        bar = add_bar(df, name, color, colormap, showlegend=showlegend)
        fig.add_traces(bar)

    if use_slider:
        fig.update_xaxes(rangeslider_visible=True)
    if do_show:
        fig.show()

    return fig

def rotate_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df


def stacked_tabular_df(
    bar_names: List[str],
    df_list: List[pd.DataFrame],
    names_col: str,
    values_col: str,
    empty_symbol: object=0,
):
    stacked = stack_dataframes(df_list, names_col, values_col, empty_symbol)
    df = pd.DataFrame.from_dict(stacked, orient='index', columns=bar_names)
    df[values_col] = stacked.keys()
    df = rotate_columns(df)
    return df


def stacked_tabular(*args, **kwargs):
    df = stacked_tabular_df(*args, **kwargs)
    display_df(df)


def stacked_bars(
    title: str,
    bar_names: List[str],
    df_list: List[pd.DataFrame],
    names_col: str,
    values_col: str,
    empty_symbol: object=0,
    colormap: Dict=None,
    display_tbl: bool=True,
    fig: go.Figure=None,
    xaxis_title: str=None,
    yaxis_title: str=None,
):
    """Stack a list of dataframes.
    Each df in df_list has a `names` column and a 'values` column.
    This function returns a dictionary indexed by each name in the
    set of all names from all dfs in df_list.
    For each `name` key the dictionary value is a list of all values from all dfs.
    If a df[name] does not exist, we create an `empty_symbol` entry for it.
    """

    stacked = stack_dataframes(df_list, names_col, values_col, empty_symbol)
    if colormap:
        bars = [go.Bar(name=k, x=bar_names, y=v, marker_color=colormap[k], text=v) for k,v in stacked.items()]
    else:
        bars = [go.Bar(name=k, x=bar_names, y=v, text=v) for k,v in stacked.items()]

    display_bars = True
    if fig is None:
        fig = go.Figure(data=bars)
        trex_base_layout(fig)
    else:
        fig.add_traces(bars)
        display_bars = False
    fig.update_layout(title=title, title_x=0.5, font_size=15,)
    fig.update_layout(barmode='stack')
    fig.update_layout(showlegend=colormap is not None)
    fig.update_traces(texttemplate='%{text:.4f}')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout({
        'yaxis_title': yaxis_title or values_col,
        'xaxis_title': xaxis_title})
    if display_bars:
        fig.show()
    if display_tbl:
        stacked_tabular(bar_names, df_list, names_col, values_col, empty_symbol)
    return fig




def plotly_hist(
    df: pd.DataFrame,
    title: str,
    values_col: str,
    xaxis_title: str,
    color: str,
    colormap=None,
    do_show: bool=True,
):
    """Draw a histogram diagram."""

    fig = px.histogram(
        df, x=values_col, title=title, nbins=len(df),
        histfunc="count",
        color=color,
        template="simple_white",
        color_discrete_map=colormap)
    fig.update_layout(xaxis={"title": xaxis_title}, bargap=0.05)
    if do_show:
        fig.show()
    return fig


def plotly_pie(
    df: pd.DataFrame,
    title: str,
    values: str,
    names: str,
    colormap: Dict[str,str]=None,
    do_show: bool=True,
):
    """Draw a pie diagram."""

    fig = go.Figure(data=[go.Pie(
        labels=df[names], values=df[values], hole=.0)])

    pie_pallette = default_pallete
    if colormap:
        pie_pallette = [colormap[key] for key in df[names]]

    marker = dict(colors=pie_pallette, line=dict(color=NVDA_GREEN, width=1))
    fig.update_traces(marker=marker)
    fig.update_traces(
        hoverinfo='label+percent', textinfo='value', textfont_size=20,)
    fig.update_layout(title=title, title_x=0.5, font_size=20,)
    if do_show:
        fig.show()
    return fig


def plotly_pie2(
    title: str,
    charts: list,
    max_cols: int=3,
    colormap: Dict[str,str]=None,
    do_show: bool=True,
):
    """Draw a pie diagram."""

    main_title = title
    n_charts = len(charts)
    n_cols = min(max_cols, n_charts)
    n_rows = math.ceil(n_charts/n_cols)
    specs = [[{'type':'domain'}] * n_cols] * n_rows

    subtitles = [chart[1] for chart in charts]
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=specs,
                        subplot_titles=subtitles)

    pie_pallette = None
    for i, chart in enumerate(charts):
        df, title, values, names = chart
        row, col = i // n_cols, i % n_cols
        texttemplate = "%{value:.1f}" if df[values].dtype == float else '%{value}'
        if colormap is not None:
            pie_pallette = [colormap[key] for key in df[names]]
        marker = dict(colors=pie_pallette, line=dict(color=NVDA_GREEN, width=1))

        fig.add_trace(go.Pie(
            labels=df[names],
            values=df[values],
            name=title,
            marker=marker,
            texttemplate=texttemplate,), row+1, col+1)

    if pie_pallette is None:
        pie_pallette = default_pallete

    fig.update_traces(
        hoverinfo='label+percent',
        textinfo='value',
        textfont_size=20,
        hole=0.4,
        textposition='inside',)
    fig.update_layout(title=main_title, title_x=0.5, font_size=20,)
    if do_show:
        fig.show()
    return fig
