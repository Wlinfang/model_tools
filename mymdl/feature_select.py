import pandas as pd
import numpy as np
from typing import Tuple, List
from model_tools.mymdl import metricutil, mdlutil

import logging


def filter_miss(df, feature_cols, threold=0.9) -> List:
    """
    过滤缺失值超过 threold 的特征
    """
    gp = mdlutil.describe_df(df, feature_cols)
    gp['miss_rate_float'] = gp['miss_rate'].str.replace('%', '')
    gp['miss_rate_float'] = gp['miss_rate_float'].astype(float)
    gp['miss_rate_float'] = np.round(gp['miss_rate_float'] / 100, 2)
    # 缺失率高的特征剔除
    miss_feature_cols = gp[gp['miss_rate_float'] > threold].index.tolist()
    logging.info('缺失率高的特征数为%s' % len(miss_feature_cols))
    return list(set(feature_cols) - set(miss_feature_cols))


def filter_onevalue(df, feature_cols) -> List:
    """
    过滤掉取值1个的特征，返回新的可用的特征列表
    """
    gp = mdlutil.describe_df(df, feature_cols)
    # 剔除取值只有1个的特征
    nunique_one_feature_cols = gp[gp['nunique'] == 1].index.tolist()
    return list(set(feature_cols) - set(nunique_one_feature_cols))


def filter_freq(df, feature_cols, threold=0.8) -> List:
    """
    剔除众数占比高的特征,返回新的特征列表
    """
    gp = mdlutil.describe_df(df, feature_cols)
    gp['freq_rate'] = np.round(gp['freq_count'] / gp['count'], 2)
    drop_cols = gp[gp['freq_rate'] > threold].index.tolist()
    return set(feature_cols)-set(drop_cols)




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
