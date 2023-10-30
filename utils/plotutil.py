import os

import numpy as np
import pandas as pd
from typing import List, Union
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from model_tools.mymdl import mdlutil


def plot_hist(df, x, group_cols=[], feature_grid=[], n_bin=20, title='', is_show=False):
    """
    概率密度图 显示数量+比例图，等宽分布
    左y: 比例；；右y:数量
    """
    gp = mdlutil.histvar(df, x, feature_grid, cut_type=2, n_bin=n_bin, group_cols=group_cols)
    fig = plot_line_with_bar(gp, 'lbl', y_line='cnt_rate', y_bar='cnt', group_cols=group_cols, title=title,
                             is_show=is_show)
    return fig, gp


def plot_scatter(df, x: str, y: str, group_cols=[], is_show=False) -> go.Figure:
    """
    趋势线：x相对于y的点图+趋势线，适用于x,y均为连续性变量
    返回 fig
    """
    if group_cols is None or len(group_cols) == 0:
        df['group_cols_str'] = ''
    else:
        df['group_cols_str'] = df[group_cols].apply(
            lambda xx: '::'.join(['{}={}'.format(k, v) for k, v in zip(group_cols, xx)]),
            axis=1)
    fig = px.scatter(df, x=x, y=y, trendline="ols", color='group_cols_str')
    if is_show:
        fig.show()
    return fig


def plot_univar(df: pd.DataFrame, x: str, y: str, group_cols=[], feature_grid=[], cut_type=1, n_bin=10, title='',
                is_show=False):
    """
    x相对于y的分布图 对x进行分桶，对y区间进行取均值
    返回 fig,gp
    """
    # 统计
    gp = mdlutil.univar(df, x, y, feature_grid, cut_type, n_bin, group_cols)
    fig = plot_line_with_bar(gp, x='lbl', y_line='avg', y_bar='cnt', group_cols=group_cols, title=title, is_show=False)
    fig.update_layout(
        yaxis=dict(title=y),
        xaxis=dict(title=x)
    )
    if is_show:
        fig.show()
    return fig, gp


def plot_univar_and_pdp(df, x, y_true, y_pred, group_cols=[], feature_grid=[], cut_type=1, n_bin=10, title='',
                        is_show=False):
    """
    适用于二分类：2个子图，一个是 x:y_true   一个是 x:y_pred
    返回 fig,gp
    """
    # 对x 进行分组
    df = mdlutil.get_bin(df, x, cut_type=1, n_bin=n_bin, feature_grid=feature_grid)
    if group_cols is None or len(group_cols) == 0:
        # 每组求y_true & y_pred 的均值
        gp = df.groupby(['lbl']).agg(
            cnt=(y_true, 'count'),
            cnt_bad=(y_true, 'sum'),
            rate_bad=(y_true, 'mean'),
            score_avg=(y_pred, 'mean')
        ).reset_index()
        gp['group_cols_str'] = ''
    else:
        gp = df.groupby(group_cols + ['lbl']).agg(
            cnt=(y_true, 'count'),
            cnt_bad=(y_true, 'sum'),
            rate_bad=(y_true, 'mean'),
            score_avg=(y_pred, 'mean')
        ).reset_index()
        gp['group_cols_str'] = gp[group_cols].apply(
            lambda xx: '::'.join(['{}={}'.format(k, v) for k, v in zip(group_cols, xx)]),
            axis=1)
    gp['rate_bad'] = np.round(gp['rate_bad'], 3)
    gp['score_avg'] = np.round(gp['score_avg'], 6)
    gp['lbl'] = gp['lbl'].astype(str)
    # 画图 x-y_true
    fig = make_subplots(rows=2, cols=1, subplot_titles=('univar-' + title, 'pdp-' + title))
    gcs = gp['group_cols_str'].unique()
    colors = px.colors.qualitative.Dark24
    for ix in range(0, len(gcs), 1):
        gc = gcs[ix]
        color = colors[ix]
        tmp = gp[gp['group_cols_str'] == gc]
        fig.add_trace(
            go.Scatter(x=tmp['lbl'], y=tmp['rate_bad'], mode='lines+markers',
                       name=gc, line=dict(color=color),
                       hovertemplate=gc + '<br><br>lbl=%{x}<br>rate_bad=%{y}<extra></extra>',
                       legendgroup=gc, showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=tmp['lbl'], y=tmp['cnt'], name=gc, opacity=0.5, marker=dict(color=color),
                   hovertemplate=gc + '<br><br>lbl=%{x}<br>cnt=%{y}<extra></extra>',
                   yaxis='y2', legendgroup=gc, showlegend=False),
            row=1, col=1
        )
        # x:y_pred
        fig.add_trace(
            go.Scatter(x=tmp['lbl'], y=tmp['score_avg'], mode='lines+markers',
                       name=gc, line=dict(color=color),
                       hovertemplate=gc + '<br><br>lbl=%{x}<br>score_avg=%{y}<extra></extra>',
                       legendgroup=gc, showlegend=True),
            row=2, col=1
        )
    fig.update_traces(row=1, col=1, secondary_y=True, overwrite=True,
                      xaxis=dict(tickangle=-30),
                      yaxis=dict(title=y_true, side='left'),
                      yaxis2=dict(title='cnt', anchor='x', overlaying='y', zeroline=True, side='right'), )
    # fig.update_traces(row=2,col=1,secondary_y=False,xaxis=dict(tickangle=-30),yaxis=dict(title=y_pred),)
    fig.update_yaxes(
        matches=None,
    )
    fig.update_layout(
        title=dict(y=0.9, x=0.5, xanchor='center', yanchor='top'),
        # 横向图例
        legend=dict(orientation='h', yanchor="bottom", y=-0.4, xanchor="left", x=0),
        width=900,
        height=900 * 0.62
    )
    if is_show:
        fig.show()
    return fig, gp


def plot_line(df, x: str, y_line: str, group_cols=[], title='', is_show=False) -> go.Figure:
    """
    折线图
    返回 fig
    """
    if group_cols is None or len(group_cols) == 0:
        df['group_cols_str'] = ''
    else:
        df['group_cols_str'] = df[group_cols].apply(
            lambda xx: '::'.join(['{}={}'.format(k, v) for k, v in zip(group_cols, xx)]),
            axis=1)
    df[x] = df[x].astype(str)
    fig = px.line(df, x=x, y=y_line, line_group='group_cols_str', markers=True, width=900, height=900 * 0.62)
    fig.update_layout(
        title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
        xaxis=dict(title=x, tickangle=-15),
        yaxis=dict(title=y_line, zeroline=True),
        legend=dict(yanchor="bottom", y=-0.4, xanchor="right", x=1, orientation='h'),
    )
    if is_show:
        fig.show()
    return fig


def plot_line_with_bar(df, x: str, y_line: str, y_bar: str, group_cols=[], title='', is_show=False) -> go.Figure:
    """
    数据分布显示：x:区间 左y 折线；右y:柱形图
    返回 fig
    """
    if group_cols is None or len(group_cols) == 0:
        df['group_cols_str'] = ''
    else:
        df['group_cols_str'] = df[group_cols].apply(
            lambda xx: '::'.join(['{}={}'.format(k, v) for k, v in zip(group_cols, xx)]),
            axis=1)
    df[x] = df[x].astype(str)
    # 左y line     # 右y bar
    gcs = df['group_cols_str'].unique()
    colors = px.colors.qualitative.Dark24
    data = []
    for ix in range(0, len(gcs), 1):
        gc = gcs[ix]
        color = colors[ix]
        tmp = df[df['group_cols_str'] == gc]

        t1 = go.Scatter(x=tmp[x], y=tmp[y_line], mode='lines+markers',
                        name=gc, line=dict(color=color),
                        hovertemplate=gc + '<br><br>' + x + '=%{x}<br>' + y_line + '=%{y}<extra></extra>',
                        legendgroup=gc, showlegend=False)

        t2 = go.Bar(x=tmp[x], y=tmp[y_bar], name=gc, opacity=0.5, marker=dict(color=color),
                    hovertemplate=gc + '<br><br>' + x + '=%{x}<br>' + y_bar + '=%{y}<extra></extra>',
                    yaxis='y2', legendgroup=gc, showlegend=True)

        data.extend([t1, t2])
    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
        xaxis=dict(title=x, tickangle=-15),
        yaxis=dict(title=y_line, zeroline=True),
        yaxis2=dict(title=y_bar, anchor='x', overlaying='y', zeroline=True, side='right'),
        legend=dict(yanchor="bottom", y=-0.4, xanchor="right", x=1, orientation='h'),
        # bargap=0.8,  # 组间距离
        bargroupgap=0.05,  # 组内距离
        width=900,
        height=900 * 0.618
    )
    if is_show:
        fig.show()
    return fig


# def plot_liftvar(df: pd.DataFrame, x1: str, y1: str, x2: str, y2: str,
#                  title='', f1_title='', f2_title='',
#                  group_col=None,
#                  is_show=True) -> go.Figure:
#     """
#     适用于 子图(x1,y1) 子图(x2,y2) 一行2个子图的情况，2个子图均为折线图
#     :param title 整个图的名称； f1_title 第一个子图的名称；f2_title 第二个子图的名称
#     :param group_col:str 分组
#     :return:fig
#     """
#     if df is None:
#         return None
#     df[x1] = df[x1].astype(str)
#     if x2 != x1:
#         df[x2] = df[x2].astype(str)
#     # 一行 两列
#     fig = make_subplots(rows=1, cols=2, subplot_titles=(f1_title, f2_title))
#     if pd.isna(group_col) or len(group_col) == 0:
#         # y_true
#         fig.add_trace(
#             go.Scatter(x=df[x1], y=df[y1], mode='lines+markers'),
#             row=1, col=1
#         )
#         # y_pred
#         fig.add_trace(
#             go.Scatter(x=df[x2], y=df[y2], mode='lines+markers'),
#             row=1, col=2
#         )
#     else:
#         gcs = df[group_col].unique()
#         colors = px.colors.qualitative.Dark24
#         for ix in range(0, len(gcs), 1):
#             gc = gcs[ix]
#             color = colors[ix]
#             tmp = df[df[group_col] == gc]
#             # 单变量折线图
#             fig.add_trace(
#                 go.Scatter(x=tmp[x1], y=tmp[y1],
#                            legendgroup='group', name=gc,
#                            hovertext=gc, line=dict(color=color)),
#                 row=1, col=1
#             )
#             # lift 图
#             fig.add_trace(
#                 go.Scatter(x=tmp[x2], y=tmp[y2], legendgroup='group',
#                            showlegend=False, hovertext=gc, line=dict(color=color)),
#                 row=1, col=2
#             )
#     fig.update_yaxes(
#         matches=None,
#     )
#     fig.update_layout(
#         title=dict(text=title, y=0.9, x=0.5, xanchor='center', yanchor='top'),
#         # 横向图例
#         legend=dict(orientation='h', yanchor="bottom", y=-0.4, xanchor="left", x=0),
#         width=1000,
#         height=1000 * 0.618,
#         xaxis=dict(tickangle=-30),
#         # 第二个子图的横坐标轴
#         xaxis2=dict(tickangle=-30),
#         yaxis=dict(title=y1),
#         yaxis2=dict(title=y2)
#     )
#     if is_show:
#         fig.show()
#     return fig


def plot_score_liftvar(df, x_score: str, target, group_cols=[], n_bin=10, feature_grid=[], is_show=False):
    """
    分组lift
    数据为明细数据
    :return fig,gp_lift
    """
    gp = mdlutil.binary_liftvar(df, x=x_score, y=target, group_cols=group_cols, n_bin=n_bin, feature_grid=feature_grid)
    # 分组计算 auc,ks,gini
    if group_cols is not None and len(group_cols) > 0:
        gp_auc = df[df[x_score].notna()].groupby(group_cols).apply(
            lambda x: mdlutil.evaluate_binary_classier(x[target], x[x_score], is_show=False))
        gp_auc = gp_auc.reset_index().rename(columns={0: 'value'})
        gp_auc.loc[:, ['cnt', 'auc', 'ks', 'gini']] = gp_auc['value'].apply(pd.Series,
                                                                            index=['cnt', 'auc', 'ks', 'gini'])
        gp_auc.drop(['value'], axis=1, inplace=True)
        gp_auc['group_cols_str'] = gp_auc[group_cols].apply(lambda x: ':'.join(x), axis=1)
        gp['group_cols_str'] = gp[group_cols].apply(lambda x: ':'.join(x), axis=1)

    else:
        cnt, auc, ks, gini = mdlutil.evaluate_binary_classier(df[df[x_score].notna()][target],
                                                              df[df[x_score].notna()][x_score])
        gp_auc = pd.DataFrame([[cnt, auc, ks, gini]], columns=['cnt', 'auc', 'ks', 'gini'], index=['all'])
        print(gp_auc)
        gp['group_cols_str'] = ''
        gp_auc['group_cols_str'] = ''
    gp_auc['auc_info'] = gp_auc.apply(
        lambda x: 'cnt:{} auc:{} ks:{}'.format(x['cnt'], x['auc'], x['ks']), axis=1)
    gp = gp.merge(gp_auc[['group_cols_str', 'auc_info']], on='group_cols_str', how='left')
    gp['legend_name'] = gp.apply(
        lambda x: '{}::{}'.format(x['group_cols_str'], x['auc_info']), axis=1)
    # lift: rate_bad lift
    t1 = gp[['legend_name', 'lbl', 'lbl_index', 'rate_bad']].rename(columns={'rate_bad': 'value'})
    t1['key'] = 'rate_bad'
    t2 = gp[['legend_name', 'lbl', 'lbl_index', 'lift_bad']].rename(columns={'lift_bad': 'value'})
    t2['key'] = 'lift_bad'
    t = pd.concat([t1, t2])
    t['lbl'] = t['lbl'].astype(str)
    fig = px.line(t, x='lbl', y='value', color='legend_name',
                  line_group='legend_name', facet_col='key',
                  orientation='h', facet_col_wrap=2, markers=True, facet_col_spacing=0.05,
                  width=1000, height=1000 * 0.62, title=x_score)

    fig.update_yaxes(
        matches=None, showticklabels=True
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        # 横向图例
        title=dict(text=x_score),
        legend=dict(orientation='h', yanchor="bottom", y=-0.5, xanchor="left", x=0),
        xaxis=dict(tickangle=-30),
        xaxis2=dict(tickangle=-30),
    )
    if is_show:
        fig.show()
    gp.drop(['group_cols_str', 'auc_info', 'legend_name'], axis=1, inplace=True)
    return fig, gp


def plot_scores_liftvar(df, x_scores: list, target, n_bin=10, is_show=False):
    """
    多个模型分的lift图
    数据为明细数据
    :return fig,gp_lift
    """
    # 计算每个模型分的lift
    data = []
    for x in x_scores:
        gp = mdlutil.binary_liftvar(df, x, target, n_bin=n_bin)
        t = df[df[x].notna()]
        cnt, auc, ks, gini = mdlutil.evaluate_binary_classier(t[target], t[x])
        gp['model_name'] = '{}::{}::auc={}::ks={}'.format(x, int(cnt), auc, ks)
        data.append(gp)
    gp = pd.concat(data)
    # lift: rate_bad lift
    t1 = gp[['model_name', 'lbl', 'lbl_index', 'rate_bad']].rename(columns={'rate_bad': 'value'})
    t1['key'] = 'rate_bad'
    t2 = gp[['model_name', 'lbl', 'lbl_index', 'lift_bad']].rename(columns={'lift_bad': 'value'})
    t2['key'] = 'lift_bad'
    t = pd.concat([t1, t2])
    fig = px.line(t, x='lbl_index', y='value', color='model_name', line_group='model_name', facet_col='key',
                  orientation='h', facet_col_wrap=1,
                  width=900, height=900 * 0.62)
    fig.update_yaxes(
        matches=None,
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(
        # 横向图例
        # legend=dict(orientation='h', yanchor="bottom", y=-0.4, xanchor="left", x=0),
        xaxis=dict(tickangle=-30),
    )
    if is_show:
        fig.show()
    return fig, gp


def plot_train_test_liftvar(df_train, df_test, x: str, target: str, group_cols=[], n_bin=10, feature_grid=[],
                            is_show=False):
    """
    基于训练集分桶，lift 表现
    返回 fig,gp
    """
    if len(feature_grid) == 0:
        feature_grid = mdlutil.get_feature_grid(df_train[x], cut_type=1, n_bin=n_bin)
    group_cols.append('sample_type')
    cols = [x, target] + group_cols
    df_train['sample_type'] = 'train'
    df_test['sample_type'] = 'test'
    t = pd.concat([df_train[cols], df_test[cols]])
    return plot_score_liftvar(t, x, target, group_cols, feature_grid=feature_grid, is_show=is_show)


def plot_corr_heatmap(df, feature_cols: list) -> pd.DataFrame:
    """
    相关性矩阵热力图,并返回相关性数据
    """
    if df is None:
        return None
    df_corr = df[feature_cols].corr()
    # 下三角
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    # 将上三角置为空
    df_corr = df_corr.mask(mask)

    x = list(df_corr.columns)
    y = list(df_corr.index)
    z = np.round(np.array(df_corr), decimals=2)
    fig = px.imshow(z, x=x, y=y, zmin=-1, zmax=1,
                    color_continuous_scale='RdBu', aspect="auto", title='pearson corr',
                    width=1000, height=1000 * 0.62)
    fig.update_traces(text=z, texttemplate="%{text}")
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        xaxis=dict(tickangle=-90),
    )
    fig.show()
    return df_corr


def plot_scatter_matrix(df, feature_cols, group_feature_name=None):
    """
    散点图:特征之间的散点图 seaborn 实现
    :return:
    """
    if len(feature_cols) > 9:
        print('特征太多了')
        return
    # pd.plotting.scatter_matrix(df[feature_cols], color='red',
    #                            hist_kwds={'bins': 30, 'color': 'red'})
    sns.set_theme(style="ticks")
    if pd.isna(group_feature_name):
        sns.pairplot(df[feature_cols])
    else:
        sns.pairplot(df[feature_cols + [group_feature_name]], hue=group_feature_name)


def plot_target_scatter_matrix(df, feature_cols, target, group_feature_name=None, is_show=True) -> go.Figure:
    """
    特征同目标变量之间的散点图
    """
    if group_feature_name is None:
        ix_cols = [target]
        df = df[feature_cols + [target]]
    else:
        ix_cols = [target, group_feature_name]
        df = df[feature_cols + [target, group_feature_name]]
    gp = df.set_index(ix_cols).stack().reset_index().rename(columns={'level_2': 'feature_name', 0: 'feature_value'})
    fig = px.scatter(gp, x='feature_value', y=target, color=group_feature_name, facet_col='feature_name',
                     facet_col_wrap=1, facet_row_spacing=0.1, width=1000, height=1000 * 0.62)
    fig.update_yaxes(matches=None)
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
    if os.path.splitext(file_name)[-1] != '.html':
        raise ValueError(f'{file_name} must be a html ')

    with open(file_name, 'a') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs=True))


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
