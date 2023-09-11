import os

import numpy as np
import pandas as pd

import plotly.graph_objects as go


def plot_univar(df: pd.DataFrame, x: str, y_rate: str, y_cnt: str, title='',
                x_title='', y_rate_title='', y_cnt_title='', is_show=True):
    """
    单变量分布图:样本量+每组的比例
    :param title:
    :param 如果 x_title y_rate_title y_cnt_title 为空，则默认使用x,y_rate,y_cnt
    """
    df[x] = df[x].astype(str)
    data = [
        go.Bar(x=df[x], y=df[y_cnt], name=y_cnt_title or y_cnt),
        go.Scatter(x=df[x], y=df[y_rate], name=y_rate_title or y_rate, yaxis='y2')
    ]
    layout = go.Layout(
        title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
        xaxis=dict(title=x_title or x, tickangle=-45),
        yaxis=dict(title=y_cnt_title or y_cnt, zeroline=True, ),
        yaxis2=dict(title=y_rate_title or y_rate,
                    anchor='x', overlaying='y', side='right',
                    zeroline=True, )
    )
    fig = go.Figure(data=data, layout=layout)
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
