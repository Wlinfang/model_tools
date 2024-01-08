import pandas as pd
import numpy as np
from typing import Tuple, List, Union
from sklearn.feature_selection import f_classif, SelectKBest
from scipy import stats
from ..mymdl import statsutil


def filter_avona_classier(df, feature_cols: list, target: str) -> list:
    """
    适用于连续特征值&分类目标变量
    方差分析 F检验，返回过滤后的有效的特征列表
    F-value ::is a measure of the statistical significance of the difference between the means of two or more groups.
    """
    skb = SelectKBest(f_classif, k='all')
    skb.fit(df[feature_cols], df[target])
    df_pvalue = pd.DataFrame(skb.pvalues_, index=skb.feature_names_in_, columns=['pvalue'])
    # 过滤掉 pvalue
    return df_pvalue[df_pvalue.pvalue < 0.05].index.tolist()


def filter_all_miss(df, feature_cols: List) -> pd.DataFrame:
    """
    1、删除行全为空的
    2、删除列全为空的
    3、返回可用的样本
    """
    # 行为空
    t = df[feature_cols].isnull().T.all()
    df = df[~df.index.isin(t[t == True].index)]
    # 列为空的
    t = df[feature_cols].isnull().all()
    cols = t[t == True].index.tolist()
    df = df.drop(cols, axis=1)
    return df


def filter_miss_freq(df, feature_cols: List, miss_threold=0.9, freq_threold=0.8) -> List:
    """
    过滤缺失值超过 miss_threold 的特征
    过滤众数占比高超过 freq_threold 的特征
    :return 返回可用的特征
    """
    gp = statsutil.describe_df(df, feature_cols)
    gp['miss_rate_float'] = gp['miss_rate'].str.replace('%', '')
    gp['miss_rate_float'] = gp['miss_rate_float'].astype(float)
    gp['miss_rate_float'] = np.round(gp['miss_rate_float'] / 100, 2)
    # 缺失率高的特征剔除
    miss_feature_cols = gp[gp['miss_rate_float'] > miss_threold].index.tolist()
    gp['freq_rate'] = np.round(gp['freq_count'] / gp['count'], 2)
    drop_cols = gp[gp['freq_rate'] > freq_threold].index.tolist()
    return list(set(feature_cols) - set(miss_feature_cols) - set(drop_cols))


def filter_iv(df, feature_cols: list, target: str, iv_threold=0.02) -> pd.DataFrame:
    """
    1、过滤掉 < iv_threold 的特征
    返回可用的特征的iv 信息
    """
    # 计算iv 值
    iv_dict = {}
    for f in feature_cols:
        iv_value = statsutil.iv(df, f, target, n_bin=10)
        if iv_value < iv_threold:
            continue
        iv_dict[f] = iv_value
    df_iv = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['iv'])
    df_iv = df_iv.reset_index()
    df_iv.sort_values('iv', ascending=False, inplace=True)
    return df_iv


def filter_corr_iv(df, feature_cols: list, target: str, corr_threold=0.8, iv_threold=0.02) -> pd.DataFrame:
    """
    1、过滤掉 < iv_threold 的特征
    2、相关性高[ >abs(corr_threold) ]的特征，保留高iv的特征   corr_threold (0,1)
    3、返回可用的特征的iv 信息
    """
    # 计算iv 值
    df_iv = filter_iv(df, feature_cols, target, iv_threold)
    # 高iv的特征
    feature_cols = df_iv['index'].unique()
    # 计算 corr
    df_corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(df_corr, dtype=bool))
    df_corr = df_corr.mask(mask)
    # 提取高corr 的 特征
    mask = np.array((df_corr < corr_threold) & (df_corr > -corr_threold) | (df_corr.isna()))
    df_corr = df_corr.mask(mask)
    # 根据corr 从大到小排序，然后
    df_corr = df_corr.stack().reset_index().rename(columns={'level_0': 'f1', 'level_1': 'f2', 0: 'corr'})
    return_cols = []
    # 剔除的特征
    tmp_cols = []
    for f in feature_cols:
        if f in tmp_cols:
            continue
        # 相关特征且iv低的
        t = list(
            set(df_corr[(df_corr['f1'] == f)]['f2'].unique()) | set(df_corr[(df_corr['f2'] == f)]['f1'].unique()))
        df_corr = df_corr[~((df_corr['f1'].isin(t)) | (df_corr['f2'].isin(t)))]
        return_cols.append(f)
        tmp_cols.extend(t)
    return df_iv[df_iv['index'].isin(return_cols)]


def filter_corr_target(df, feature_cols: list, target: str, threld=0.1, corr_method='pearsonr') -> list:
    """
    过滤掉 feature 同 targe 不相关的特征 拒绝掉  （-threld，threld） 的特征
    计算指标 pearsonr & spearmanr=(变量排序 + pearsonr) & kendalltau(有序性)
    :param corr_method: pearsonr 、spearmanr 、kendalltau
    :return:返回 可用的特征列表
    0~0.2 无相关或者积弱
    0.2~0.4 弱相关
    0.4~0.6 中等相关
    0.6~0.8 强相关
    0.8~1  极强相关
    """
    data = []
    for f in feature_cols:
        t = df[df[f].notna()]
        if t.shape[0] < 5:
            continue
        if corr_method == 'pearsonr':
            s1, p1 = stats.pearsonr(t[f], t[target])
        elif corr_method == 'spearmanr':
            s1, p1 = stats.spearmanr(df[f], df[target], nan_policy='omit')
        elif corr_method == 'kendalltau':
            s1, p1 = stats.kendalltau(df[f], df[target], nan_policy='omit')
        else:
            s1, p1 = None, None
        data.append([f, s1, p1])
    df_corr = pd.DataFrame(data, columns=['feature_name', 'corr_value', 'pvalue'])
    # 拒绝
    df_corr = df_corr[~((df_corr['corr_value'] < threld) & (df_corr['corr_value'] > -threld))]
    return df_corr.feature_name.tolist()


def filter_psi(df_base, df_test, feature_names: list, target: str, psi_threld=0.3) -> pd.DataFrame:
    """
    1、过滤掉 psi > psi_threld 的 特征
    返回可用的特征的 psi,iv
    """
    data = []
    for f in feature_names:
        p = statsutil.psi(df_base[f], df_test[f], n_bin=10)
        if target in df_base.columns and len(df_base[target].nunique()) == 2:
            iv_base = statsutil.iv(df_base, f, target)
            iv_test = statsutil.iv(df_test, f, target)
            data.append([f, p, iv_base, iv_test])
        else:
            data.append([f, p])
    if np.array(data).ndim[1]==2:
        df_iv = pd.DataFrame(data, columns=['feature_name', 'psi'])
    else:
        df_iv = pd.DataFrame(data, columns=['feature_name', 'psi', 'iv_base', 'iv_test'])
    # 过滤掉 psi > psi_threld
    df_iv = df_iv[~(df_iv['psi'] > psi_threld)]
    return df_iv
