"""
主要用于统计分析
"""
import pandas as pd
import numpy as np


def ctribtion_rate_one(df_base, df_test, x, sub_group_cols: list, group_cols=[]) -> pd.DataFrame:
    """
    贡献率公式1，加法模式 case: uv,pv
    贡献度 = 维度值绝对DIFF / 大盘绝对DIFF。
    Y = ∑X_i【子项】   △Y= (Y_test - Y_base ) / Y_base
    某子项的贡献率 = △Y_i/△Y = (X_test_i - X_base_i) / Y_base / △Y = (X_test_i - X_base_i) / (Y_test - Y_base )
    :param group_cols 整体 可以为空；sub_group_cols 子项
    """
    df_base[x] = pd.to_numeric(df_base[x], errors='coerce')
    df_test[x] = pd.to_numeric(df_test[x], errors='coerce')

    if sub_group_cols is None or len(sub_group_cols) == 0:
        raise ValueError('sub_group_cols must not be empty')
    gp_base = df_base.groupby(sub_group_cols).agg(sub_sum=(x, 'sum')).reset_index()
    gp_test = df_test.groupby(sub_group_cols).agg(sub_sum=(x, 'sum')).reset_index()
    # 总体，
    if group_cols is None or len(group_cols) == 0:
        gp_base['all_sum'] = df_base[x].sum()
        gp_test['all_sum'] = df_test[x].sum()
    else:
        if len(set(group_cols) - set(sub_group_cols)) != 0:
            raise ValueError('group_cols is sub set of sub_group_cols')
        t = df_base.groupby(group_cols).agg(all_sum=(x, 'sum')).reset_index()
        gp_base = gp_base.merge(t, on=group_cols, how='left')
        # test
        t = df_test.groupby(group_cols).agg(all_sum=(x, 'sum')).reset_index()
        gp_test = gp_test.merge(t, on=group_cols, how='left')
    gp = gp_base.merge(gp_test, on=sub_group_cols, how='outer', suffiexs=('_base', '_test'))
    gp['贡献率'] = np.round((gp.sub_sum_test - gp.sub_sum_base) / (gp.all_sum_test - gp.all_sum_base), 3)
    return gp


def ctribtion_rate_two(df_base, df_test, x, y, x_method='sum', y_method='count', group_cols=[])->pd.DataFrame:
    """
    适用于比例类指标：考虑指标变化 + 子项样本占比变化
    case: R = 放款率 = 放款数/ 授信数 ; 放款数=安卓放款数+IOS 放款数  授信数=安卓授信数+IOS 授信数
    样本占比  P = 安卓授信数 / 授信数   S = 安卓放款数/放款数
    指标变化测算：A_i=(R_test_{ios} - R_base_{ios}) * P_base_{ios} : 假设 样本占比不变的情况下，指标变化率的贡献
    占比变化测算：B_i=(P_test_{ios}-P_base_{ios}) * (R_test_{ios} - R_base)
    总体贡献度 C_i = A_i + B_i
    diff R = R_test - R_base =  ∑C_i
    证明过程：P_test_{ios} * R_test_{ios} +P_test_{and} * R_test_{and}  = R_test
    """
    df_base[x] = pd.to_numeric(df_base[x], errors='coerce')
    df_base[y] = pd.to_numeric(df_base[y], errors='coerce')

    df_test[x] = pd.to_numeric(df_test[x], errors='coerce')
    df_test[y] = pd.to_numeric(df_test[y], errors='coerce')

    # 总的变化
    if x_method == 'sum':
        x_base_total = df_base[x].sum()
        x_test_total = df_test[x].sum()
    elif x_method == 'count':
        x_base_total = df_base[x].shape[0]
        x_test_total = df_test[x].shape[0]
    else:
        raise ValueError('x_method not support,only sum or count')

    if y_method == 'sum':
        y_base_total = df_base[y].sum()
        y_test_total = df_test[y].sum()
    elif y_method == 'count':
        y_base_total = df_base[y].shape[0]
        y_test_total = df_test[y].shape[0]
    else:
        raise ValueError('y_method not support,only sum or count')

    r = np.round(x_test_total / y_test_total - x_base_total / y_base_total, 3)

    # 维度变化
    gp_base = df_base.groupby(group_cols).agg(x_sub=(x, x_method), y_sub=(y, y_method)).reset_index()
    gp_base['r'] = np.round(gp_base.x_sub / gp_base.y_sub, 3)
    # 样本占比
    gp_base['y_over_ally'] = np.round(gp_base.y_sub / y_base_total, 3)
    gp_test = df_test.groupby(group_cols).agg(x_sub=(x, x_method), y_sub=(y, y_method)).reset_index()
    gp_test['r'] = np.round(gp_test.x_sub / gp_test.y_sub, 3)
    gp_test['y_over_ally'] = np.round(gp_test.y_sub / y_test_total, 3)

    # 合并
    gp = gp_base.merge(gp_test, on=group_cols, suffixes=('_base', '_test'), how='outer')
    gp = gp.fillna(0)
    gp['贡献率_指标变化'] = np.round((gp.r_test - gp.r_base) * gp.y_over_ally_base / r, 3)
    gp['贡献率_占比变化'] = np.round(
        (gp.y_over_ally_test - gp.y_over_ally_base) * (gp.r_test - x_base_total / y_base_total) / r, 3)
    gp['贡献率'] = gp['贡献率_指标变化'] + gp['贡献率_占比变化']
    cols = group_cols + ['x_sub_base', 'y_sub_base', 'r_base', 'y_over_ally_base', 'x_sub_test', 'y_sub_test', 'r_test',
                         'y_over_ally_test', '贡献率_指标变化', '贡献率_占比变化', '贡献率']
    gp = gp[cols]
    gp = gp.rename(columns={'x_sub_base': 'x_base', 'y_sub_base': 'y_base', 'y_over_ally_base': '占比_base','r_base':'指标_base',
                            'x_sub_test': 'x_test', 'y_sub_test': 'y_test', 'y_over_ally_test': '占比_test','r_test':'指标_test'})
    return gp


def ctribtion_rate_lmdi():
    """
    适用于乘法链路：case: Y【新增成交用户数】=N[新增用户量] × A[用户激活率] × S[激活用户3日留存率]  × P[留存用户购买率]
    ln y_test / y_base = ln N_test/N_base + ln A_test/A_base + ln S_test/S_base + ln P_test/P_base
    1 = (ln N_test/N_base + ln A_test/A_base + ln S_test/S_base + ln P_test/P_base) / ln y_test / y_base
    """
    pass
