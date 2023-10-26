import pandas as pd
import numpy as np
from typing import Tuple, List, Union
from model_tools.mymdl import mdlutil
from sklearn.feature_selection import f_classif, SelectKBest
from scipy import stats


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
    剔除样本特征全为空的样本
    """
    t = df[feature_cols].isnull().T.all()
    df = df[~df.index.isin(t[t == True].index)]
    return df


def filter_miss_freq(df, feature_cols:List, miss_threold=0.9, freq_threold=0.8) -> List:
    """
    过滤缺失值超过 miss_threold 的特征
    过滤众数占比高超过 freq_threold 的特征
    :return 返回可用的特征
    """
    gp = mdlutil.describe_df(df, feature_cols)
    gp['miss_rate_float'] = gp['miss_rate'].str.replace('%', '')
    gp['miss_rate_float'] = gp['miss_rate_float'].astype(float)
    gp['miss_rate_float'] = np.round(gp['miss_rate_float'] / 100, 2)
    # 缺失率高的特征剔除
    miss_feature_cols = gp[gp['miss_rate_float'] > miss_threold].index.tolist()
    gp['freq_rate'] = np.round(gp['freq_count'] / gp['count'], 2)
    drop_cols = gp[gp['freq_rate'] > freq_threold].index.tolist()
    return list(set(feature_cols) - set(miss_feature_cols) - set(drop_cols))


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

    # 计算iv 值
    iv_dict = {}
    for f in feature_cols:
        iv_value = mdlutil.iv(df, f, target, cut_type=1, n_bin=10)
        if iv_value < iv_threold:
            continue
        iv_dict[f] = iv_value
    df_iv = pd.DataFrame.from_dict(iv_dict, orient='index', columns=['iv'])
    df_iv = df_iv.reset_index()
    # 高iv的特征
    feature_cols = df_iv['index'].unique()
    df_corr = filter_corr(df, feature_cols, corr_threold)
    # 高相关性的特征剔除
    df_corr = df_corr.merge(df_iv.rename(columns={'index': 'f1', 'iv': 'f1_iv'}), on='f1', how='left')
    df_corr = df_corr.merge(df_iv.rename(columns={'index': 'f2', 'iv': 'f2_iv'}), on='f2', how='left')
    return df_corr, df_iv


def filter_corr_target(df, feature_cols, target, threld=0.1, corr_method='pearsonr') -> list:
    """
    过滤掉 feature 同 targe 不相关的特征 拒绝掉  （-threld，threld） 的特征
    计算指标 pearsonr & spearmanr=(变量排序 + pearsonr) & kendalltau(有序性)
    :param corr_method: pearsonr 、spearmanr 、kendalltau
    :return:返回 有相关性的特征列表
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


def psi(data_base: Union[list, np.array, pd.Series],
        data_test: Union[list, np.array, pd.Series], feature_grid=[], n_bin=10):
    """
    支持数值型&类别型，计算的时候，剔除空值
    :param data_base: 基准值
    :param data_test: 观察值
    :param feature_grid 分组参数 如果未指定，则按照等频 n_bin 分组
    :return psi   >0.25  分布差异明显
    """
    if (data_base is None) or (data_test is None):
        raise ValueError('data error')
    # 类型转换
    data_base = np.array(data_base)
    data_test = np.array(data_test)
    data_base = pd.DataFrame(data_base, columns=['data'])
    data_test = pd.DataFrame(data_test, columns=['data'])
    data_base['data'] = pd.to_numeric(data_base['data'], errors='ignore')
    data_test['data'] = pd.to_numeric(data_test['data'], errors='ignore')
    # 去除空值
    data_base = data_base[data_base['data'].notna()]
    data_test = data_test[data_test['data'].notna()]
    if not pd.api.types.is_numeric_dtype(data_base):
        # 非数字
        data_base['lbl'] = data_base['data']
        data_test['lbl'] = data_test['data']
    else:
        # 对 data_base 进行分组
        if feature_grid:
            feature_grid = mdlutil.get_feature_grid(data_base, cut_type=1, n_bin=n_bin)
        data_base = mdlutil.get_bin(data_base, 'data', feature_grid=feature_grid)
        data_test = mdlutil.get_bin(data_test, 'data', feature_grid=feature_grid)
    # 统计每个区间的分布
    gp_base = data_base.groupby('lbl')['data'].count().reset_index()
    gp_test = data_test.groupby('lbl')['data'].count().reset_index()
    gp_base['rate'] = np.round(gp_base['data'] / data_base.shape[0], 3)
    gp_test['rate'] = np.round(gp_test['data'] / data_test.shape[0], 3)
    gp = gp_base.merge(gp_test, on='lbl', how='outer', suffixes=('_base', '_test'))
    # psi 分组计算求和，分组公式=(base_rate-pre_rate) * ln(base_rate/pre_rate)
    gp[['rate_base', 'rate_test']].fillna(0, inplace=True)
    eps = np.finfo(np.float32).eps
    gp['psi'] = (gp.rate_base - gp.rate_test) * np.log((gp.rate_base + eps) / (gp.rate_test + eps))
    return np.round(gp['psi'].sum(), 2)


def filter_features_psi(df_base, df_test, feature_names, threold=0.3) -> List:
    """
    拒绝不稳定的特征，>threold 的特征，返回 <= threold的特征
    """
    left_feature_names = []
    for f in feature_names:
        p = psi(df_base[f], df_test[f], n_bin=10)
        if p <= threold:
            left_feature_names.append(f)
    return left_feature_names


def filter_psi_iv(df_base, df_test, feature_names, target, psi_threld=0.3, iv_threld=0.2) -> pd.DataFrame:
    """
    计算稳定性和iv
    拒绝掉 psi > psi_threld 的 数据
    拒绝掉 abs(iv_base-iv_test) > iv_threld
    """
    data = []
    for f in feature_names:
        p = psi(df_base[f], df_test[f], n_bin=10)
        iv_base = mdlutil.iv(df_base, f, target)
        iv_test = mdlutil.iv(df_test, f, target)
        data.append([f, p, iv_base, iv_test])
    df_iv = pd.DataFrame(data, columns=['feature_name', 'psi', 'iv_base', 'iv_test'])
    df_iv['diff_iv'] = np.abs(df_iv['iv_base'] - df_iv['iv_test'])
    df_iv = df_iv[~((df_iv.diff_iv > iv_threld) | (df_iv['psi'] > psi_threld))]
    return df_iv
