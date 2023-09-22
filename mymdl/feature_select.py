import pandas as pd
import numpy as np
from typing import Tuple
from model_tools.mymdl import metricutil


def filter_corr(df, feature_cols, threold):
    """
    筛选高相关性的特征
    :param df:
    :param feature_cols:
    :param threold: (0~1)
    :return:
    """
    if df is None:
        return None
    if threold > 1 or threold < 0:
        return None
    df_corr = df[feature_cols].corr()
    # 下三角
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    # 将上三角置为空
    df_corr = df_corr.mask(mask)
    mask = np.array((df_corr < threold) | ((df_corr > -threold) & (df_corr < 0)))
    df_corr = df_corr.mask(mask)
    data = []
    for f in df_corr.columns:
        t = df_corr[f]
        t = t[t.notna()]
        t = t.reset_index()
        t.columns = ['f1', 'corr']
        t['f2'] = f
        data.append(t)
    df_corr = pd.concat(data)
    df_corr.sort_values(['f1', 'corr'], ascending=True, inplace=True)
    cols = ['f1', 'f2', 'corr']
    return df_corr[cols]


def filter_corr_iv(df, feature_cols, target, corr_threold=0.8, iv_threold=0.02) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    高相关性iv 过滤
    :param df:
    :param feature_cols:
    :param target:目标变量
    :param corr_threold:相关性阈值 0~1之间
    :param iv_threold iv 阈值
    :param df_corr  df_iv
    """
    df_corr = filter_corr(df, feature_cols, corr_threold)
    # 计算iv 值
    iv_dict = {}
    for f in feature_cols:
        iv_value = metricutil.iv(df, f, target, cut_type=1, n_bin=10)
        if iv_value < iv_threold:
            continue
        iv_dict[f] = iv_value
    df_iv = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['iv'])
    df_iv = df_iv.reset_index()
    # 高相关性的特征剔除
    df_corr = df_corr.merge(df_iv.rename(columns={'index': 'f1', 'iv': 'f1_iv'}), on='f1', how='left')
    df_corr = df_corr.merge(df_iv.rename(columns={'index': 'f2', 'iv': 'f2_iv'}), on='f2', how='left')
    return df_corr, df_iv
