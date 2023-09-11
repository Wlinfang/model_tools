import os

import numpy as np
import pandas as pd
from typing import List, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_univar(df: pd.DataFrame, x: str, y: str, title='',
                group_cols='',
                is_show=False) -> go.Figure:
    """
    单变量 折线图
    :param df:
    :param x:
    :param y:
    :param title:
    :param is_show:
    :return:
    """
    df[x] = df[x].astype(str)
    data = [
        go.Scatter(x=df[x], y=df[y], name=y)
    ]

    layout = go.Layout(
        title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
        legend=dict(yanchor="top", y=1.2, xanchor="right", x=1),
        xaxis=dict(title=x, tickangle=-45),
        yaxis=dict(title=y, zeroline=True, ),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout()
    if is_show:
        fig.show()
    return fig


def plot_univar_with_bar(df: pd.DataFrame, x: str, y_rate: str, y_cnt: str,
                         group_col='',
                         title='',
                         is_show=False) -> go.Figure:
    """
    单变量分布图:柱形图+折线图的联合分布
    :param y_rate  折线图
    :param y_cnt 柱形图
    :param group_col 分组
    :param title:图片名字
    """
    df[x] = df[x].astype(str)
    data = [
        go.Bar(x=df[x], y=df[y_cnt], name=y_cnt),
        go.Scatter(x=df[x], y=df[y_rate], name=y_rate, yaxis='y2')
    ]
    layout = go.Layout(
        barmode='group',
        bargap=0.4,  # 组间距离
        bargroupgap=0.2,  # 组内距离
        title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
        legend=dict(yanchor="top", y=1.2, xanchor="right", x=1),
        xaxis=dict(title=x, tickangle=-45),
        yaxis=dict(title=y_cnt, zeroline=True, ),
        yaxis2=dict(title=y_rate,
                    anchor='x', overlaying='y', side='right',
                    zeroline=True, )
    )
    fig = go.Figure(data=data, layout=layout)
    if is_show:
        fig.show()
    return fig


def plot_univar_with_pdp(df, x, y1_cnt, y1_rate, y2, title='', is_show=False) -> go.Figure:
    """
    单变量分布图  变量x 同 y_true 的关系  变量 x 同 y_pred 的关系
    :param df:
    :param x:
    :param y1_cnt:
    :param y1_rate:
    :param y2:
    :return:
    """
    # 2 行 1 列
    fig = make_subplots(rows=2, cols=1, subplot_titles=('univar', 'php'))
    data = [
        go.Bar(x=df[x], y=df[y1_cnt], name=y1_cnt),
        go.Scatter(x=df[x], y=df[y1_rate], name=y1_rate, yaxis='y2')
    ]
    fig.add_trace(
        data=data,
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df[x], y=df[y2], name=y2, mode='lines+markers'),
        row=2, col=1
    )
    fig.update_layout(
        title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
        legend=dict(yanchor="top", y=1.2, xanchor="right", x=1),
        xaxis=dict(tickangle=-45),
        yaxis2=dict(title=y1_rate, anchor='x', overlaying='y', side='right', zeroline=True),
        # 第二个子图的横坐标轴
        xaxis2=dict(tickangle=-45),
        yaxis3=dict(title=y2, zeroline=True),
    )
    if is_show:
        fig.show()
    return fig


def plot_liftvar(df: pd.DataFrame, x1: str, y1: str, x2: str, y2: str,
                 title='', f1_title='', f2_title='', is_show=True) -> go.Figure:
    """
    适用于(x1,y1) (x2,y2) 一行2个子图的情况，2个子图均为折线图
    :param title 整个图的名称； f1_title 第一个子图的名称；f2_title 第二个子图的名称
    :return:fig
    """
    # 一行 两列
    fig = make_subplots(rows=1, cols=2, subplot_titles=('', ''))
    # traces
    fig.add_trace(
        go.Scatter(x=df[x1], y=df[y1], name=f1_title, mode='lines+markers'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df[x2], y=df[y2], name=f2_title, mode='lines+markers'),
        row=1, col=2
    )
    fig.update_layout(
        title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
        legend=dict(yanchor="top", y=1.2, xanchor="right", x=1),
        xaxis=dict(tickangle=-45),
        # 第二个子图的横坐标轴
        xaxis2=dict(tickangle=-45),
    )
    if is_show:
        fig.show()
    return fig


def plot_liftvars(xs1: List[Union[list, pd.Series, np.array]],
                  ys1: List[Union[list, pd.Series, np.array]],
                  xs2: List[Union[list, pd.Series, np.array]],
                  ys2: List[Union[list, pd.Series, np.array]],
                  title,
                  fig_titles, is_show=False
                  ):
    """
    适用于多个数据集的情况，比如 训练集、测试集、验证集;2个子图
    n = len(xs1) = len(xs2) = len(ys1) = len(ys2) 表示有 n 个数据集
    :param (xs1[0],ys1[0]) (xs2[0],ys2[0]) 相当于 plot_liftvar 一个数据集的lift 图
    :param fig_titles : 每个数据集的title len(fig_titles) = n
    :return:
    """
    # 一行 两列
    fig = make_subplots(rows=1, cols=2, subplot_titles=('', ''))
    for x1, y1, x2, y2, f_title in zip(xs1, ys1, xs2, ys2, fig_titles):
        # traces
        fig.add_trace(
            go.Scatter(x=x1, y=y1, name=f_title, mode='lines+markers'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x2, y=y2, name=f_title, mode='lines+markers'),
            row=1, col=1
        )
    fig.update_layout(
        title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
        legend=dict(yanchor="top", y=1.2, xanchor="right", x=1),
        xaxis=dict(tickangle=-45),
        # 第二个子图的横坐标轴
        xaxis2=dict(tickangle=-45),
    )
    if is_show:
        fig.show()
    return fig


def save_fig_tohtml(file_name: str, fig: go.Figure):
    """
    将图片保存到 file_name 中，append 追加形式保存
    :param file_name:
    :param fig:
    :return:
    """
    # 判断文件名为 .html
    if not os.path.isfile(file_name):
        raise ValueError(f'{file_name} is error')
    if os.path.splitext(file_name)[-1] != '.html':
        raise ValueError(f'{file_name} must be a html ')

    with open(file_name, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))


def show_dash(figs):
    from dash import Dash, dcc, html
    """
    dash 展示
    """
    app = Dash()
    graphs = [dcc.Graph(figure=fig) for fig in figs]
    app.layout = html.Div(graphs)
    app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

#
# def plot_corr(df, feature_name_list):
#     """
#     热力图：皮尔逊系数计算
#     """
#     plt.figure(figsize=(8, 6))
#     corr = np.corrcoef(df[feature_name_list], rowvar=False)
#     # 以corr的形状生成一个全为0的矩阵
#     mask = np.zeros_like(corr)
#     # 将mask的对角线及以上设置为True
#     # 这部分就是对应要被遮掉的部分
#     mask[np.triu_indices_from(mask, k=1)] = True
#     with sn.axes_style("white"):
#         sn.heatmap(corr, mask=mask, fmt='.2f', annot=True, cmap="RdBu_r")
#     plt.title('person corr')
#     plt.show()

#
# def rgb_to_hex(rgb):
#     """
#     RGB格式颜色转换为16进制颜色格式
#     rgb 格式: '100, 100,100'  or (100, 100,100)
#     16进制格式:'#0F0F0F'
#     """
#     if isinstance(rgb, str):
#         RGB = rgb.split(',')  # 将RGB格式划分开来
#     elif isinstance(rgb, tuple):
#         RGB = list(rgb)
#     elif isinstance(rgb, list):
#         RGB = rgb
#     else:
#         raise ValueError('rgb format error ')
#     if len(rgb) != 3:
#         raise ValueError('rgb must having 3 value')
#     color = '#'
#     for i in RGB:
#         num = int(i)
#         # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
#         color += str(hex(num))[-2:].replace('x', '0').upper()
#     return color
#
#
# def hex_to_rgb(hex: str) -> tuple:
#     '''
#     16进制颜色格式颜色转换为RGB格式
#     RGB 格式: '100, 100,100'  or (100, 100,100)
#     hex:16进制格式:'#0F0F0F'
#     返回：(100, 100,100) 两种格式的颜色
#     '''
#     r = int(hex[1:3], 16)
#     g = int(hex[3:5], 16)
#     b = int(hex[5:7], 16)
#     return tuple([r, g, b])
#
#
# def gradient_color(color_list, color_sum=700, is_show=True):
#     '''
#     生成渐变色
#     color_list：list，16进制的颜色list
#     color_sum：颜色个数
#     '''
#     # 主颜色个数
#     color_center_count = len(color_list)
#     if color_sum < color_center_count * 3:
#         color_sum = color_center_count * 3
#     # 两个颜色之中的个数
#     color_sub_count = int(color_sum / (color_center_count - 1))
#     color_index_start = 0
#     color_map = []
#     for color_index_end in range(1, color_center_count):
#         color_rgb_start = hex_to_rgb(color_list[color_index_start])[1]
#         color_rgb_end = hex_to_rgb(color_list[color_index_end])[1]
#         r_step = (color_rgb_end[0] - color_rgb_start[0]) / color_sub_count
#         g_step = (color_rgb_end[1] - color_rgb_start[1]) / color_sub_count
#         b_step = (color_rgb_end[2] - color_rgb_start[2]) / color_sub_count
#         # 生成中间渐变色
#         now_color = color_rgb_start
#         color_map.append(rgb_to_hex(now_color))
#         for color_index in range(1, color_sub_count):
#             now_color = [now_color[0] + r_step, now_color[1] + g_step, now_color[2] + b_step]
#             color_map.append(rgb_to_hex(now_color))
#         color_index_start = color_index_end
#
#     # if is_show:
#     #     sn.palplot(color_map)
#     return color_map
#
#
# def gradient_color_by_palette(color_sum=10, is_show=True):
#     '''
#     利用调色板自定义颜色变化
#     '''
#     colors = sn.color_palette("Blues", n_colors=color_sum)
#     # colors 中返回的rgb 在 0-1之间
#     df = pd.DataFrame(np.array(colors) * 256, columns=['R', 'G', 'B'])
#     df['hex'] = df.apply(lambda x: rgb_to_hex([x.R, x.G, x.B]), axis=1)
#     # if is_show:
#     #     sn.palplot(df.hex)
#     return df
#
#
# def draw_table(ax, df: pd.DataFrame, cols: list):
#     '''
#     画表格图像
#     ax:matplotlib.axes
#     '''
#     ax.set_axis_off()
#     table = ax.table(df[cols].values, colLabels=cols, loc='center')
#     table.auto_set_column_width(True)
#     table.auto_set_font_size(True)
#     # fig.savefit(bbox_inches='tight')
#     # key, cell in table.get_celld().items()
#     return table
