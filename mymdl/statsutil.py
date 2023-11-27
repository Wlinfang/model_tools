"""
主要用于统计分析
"""
import pandas as pd
import numpy as np


def ctribution_rate_one(df, x, sub_group_cols=[], group_cols=[]) -> pd.DataFrame:
    """
    贡献率公式1，适用于总额的变化的情况
    case:本月销售总额100w,其中电器50w，则电器的贡献度 50%
    :param sub_group_cols 子维度； group_cols 总体维度
    """
    df[x] = pd.to_numeric(df[x], errors='coerce')
    if sub_group_cols is None or len(sub_group_cols) == 0:
        raise ValueError('sub_group_cols must not be empty')
    gp = df.groupby(sub_group_cols).agg(sub_sum=(x, 'sum')).reset_index()
    # 总体，
    if group_cols is None or len(group_cols) == 0:
        gp['all_sum'] = df[x].sum()
    else:
        if len(set(group_cols) - set(sub_group_cols)) != 0:
            raise ValueError('group_cols is sub set of sub_group_cols')
        t = df.groupby(group_cols).agg(all_sum=(x, 'sum')).reset_index()
        gp = gp.merge(t, on=group_cols, how='left')
    gp['贡献率'] = np.round(gp.sub_sum / gp.all_sum, 3)
    return gp


def ctribution_rate_two(df_base, df_test, x, group_cols=[]):
    """
    贡献率公式2、适用于增长变化率
    case:本月总额F；上月总额E； 本月环比上月变化 Y=(F-E)/E；
    其中电器的贡献率=(Fi-Ei)/E / (F-E)/E = (Fi-Ei)/(F-E)
    """
    df_test[x] = pd.to_numeric(df_test[x], errors='coerce')
    df_base[x] = pd.to_numeric(df_base[x], errors='coerce')
    # 总的变化
    y = df_test[x].sum() - df_base[x].sum()
    # 维度变化
    gp_base = df_base.groupby(group_cols).agg(sub_sum=(x, 'sum')).reset_index()
    gp_test = df_test.groupby(group_cols).agg(sub_sum=(x, 'sum')).reset_index()
    gp = gp_base.merge(gp_test, on=group_cols, suffiexs=('_base', '_test'), how='outer')
    gp['贡献率'] = np.round((gp['sub_sum_test'] - gp['sub_sum_base']) / y, 3)
    gp['变化率'] = np.round(y / df_base[x].sum(), 3)
    return gp

def ctribution_rate_three(df):
    """
    贡献率公式3,适用于比率的差异的贡献

    """
