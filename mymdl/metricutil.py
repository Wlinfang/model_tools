import pandas as pd
import numpy as np
from typing import Union, Tuple
from scipy import stats
import logging

from model_tools.utils import toolutil
def get_feature_grid(values: Union[list, np.array],
                     cut_type=1, n_bin=10, default_values=[]) -> list:
    """
    计算分桶
    :param values: 要进行分桶的数据
    :param cut_type: 1 : 等频; 2:等宽 3、
    :return
    """
    if (values is None) or (len(values) < 1) or (n_bin <= 0):
        logging.info('param values is %s', values)
        return None
    values = np.array(values)
    # 限制为 1d
    if values.ndim > 1:
        logging.info('param values is %s', values)
        return None
    # 去除空值、空字符串
    values = toolutil.del_none(values)
    # 去除默认值
    values = values[~np.isin(values, default_values)]
    # 非数值型数据，直接返回
    if values.dtype.kind not in ['i', 'u', 'f', 'c']:
        return np.unique(values)
    n = np.unique(values).size
    if n == 0:
        return None
    # 对values 进行排序
    vs_sort = np.sort(values)
    if n <= n_bin:
        f = [-np.Inf]
        f.extend(list(np.unique(vs_sort.tolist())))
    else:
        if cut_type == 1:
            # 等频
            bin_index = [i / n_bin for i in range(0, n_bin + 1)]
            f = np.sort(np.unique(np.quantile(vs_sort, bin_index)))
        elif cut_type == 2:
            # 等宽
            mi, mx = values.min(), values.max()
            bin_index = [mi + (mx - mi) * i / n_bin for i in range(0, n_bin + 1)]
            f = np.sort(np.unique(bin_index))
        else:
            return None
        # 包括无穷大，样本集中数据可能有些最小值，最大值不全
        f[0] = -np.Inf
        f[-1] = np.inf
    return np.round(f, 3)


def get_bin(df: pd.DataFrame, feature_name: str, cut_type=1,
            n_bin=10, feature_grid=[], default_values=[]):
    """
    分组；默认值+缺失值为分为1组
    :param cut_type 分组类型： 1 ：等频分组；2：等宽分组
    :param n_bin 分组个数，默认10个组；如果加入缺失值，则分为11组
    :param feature_grid 优先按照 feature_grid 分组；如果未指定，按照cut_type
    :return df 含有字段 lbl  lbl_index lbl_left
    """
    if (df is None) or (df.shape[0] == 0):
        logging.error('data is None ')
        return None
    df[feature_name] = pd.to_numeric(df[feature_name], errors='ignore')
    # 分为空和非空
    t1 = df[(df[feature_name].notna()) & (~df[feature_name].isin(default_values))].copy()
    t2 = df[~df.index.isin(t1.index)].copy()

    if not pd.api.types.is_numeric_dtype(df[feature_name]):
        # 非数字型
        t1['lbl'] = t1[feature_name]
    if feature_grid is None or len(feature_grid) == 0:
        # 如果未指定
        feature_grid = get_feature_grid(df[feature_name], cut_type, n_bin
                                        , default_values)
        if len(feature_grid) == 0:
            logging.error('feature_grid is None ')
            return None
    if pd.api.types.is_numeric_dtype(df[feature_name]):
        # 数字型 左闭右开
        t1['lbl'] = pd.cut(t1[feature_name], feature_grid, include_lowest=True,
                           right=False, precision=4)

    t1['lbl'] = t1['lbl'].astype('category')
    # 则为缺失值
    t1 = pd.concat([t1, t2], axis=0)
    # 填充空值
    t1['lbl'] = t1['lbl'].cat.add_categories('miss_data')
    t1['lbl'] = t1['lbl'].fillna('miss_data')
    cats = t1['lbl'].cat.categories.tolist()
    cats = [i.left if isinstance(i, pd.Interval) else i for i in cats]
    t1['lbl_index'] = t1['lbl'].cat.codes
    t1['lbl_left'] = t1.lbl_index.apply(lambda x: cats[x])
    # 更新空值，空值为最大
    t1.loc[t1.lbl_index == -1, 'lbl_index'] = t1.lbl_index.max() + 1
    t1['lbl_left'] = t1['lbl_left'].astype(str)
    return t1


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
            feature_grid = get_feature_grid(data_base, cut_type=1, n_bin=n_bin)
        data_base = get_bin(data_base, 'data', feature_grid=feature_grid)
        data_test = get_bin(data_test, 'data', feature_grid=feature_grid)
    # 统计每个区间的分布
    gp_base = data_base.groupby('lbl')['data'].count().reset_index()
    gp_test = data_test.groupby('lbl')['data'].count().reset_index()
    gp_base['rate'] = np.round(gp_base['data'] / data_base.shape[0], 3)
    gp_test['rate'] = np.round(gp_test['data'] / data_test.shape[0], 3)
    gp = gp_base.merge(gp_test, on='lbl', how='outer', suffixes=('_base', '_test'))
    # psi 分组计算求和，分组公式=(base_rate-pre_rate) * ln(base_rate/pre_rate)
    gp[['rate_base', 'rate_test']].fillna(0, inplace=True)
    eps = np.finfo(np.float32).eps
    gp['psi'] = (gp.base_rate - gp.test_rate) * np.log((gp.base_rate + eps) / (gp.test_rate + eps))
    return np.round(gp['psi'].sum(), 2)


def woe(df: pd.DataFrame, x: str, y: str, feature_grid=[], cut_type=1, n_bin=10):
    """
    woe = ln (坏人比例) / (好人比例)
    坏人比例 =  组内的坏人数/ 总坏人数
    好人比例 = 组内的好人数/ 总好人数
    缺失值的woe 可不满足单调性，因为缺失值尤其逻辑含义
    如果相邻分箱的woe值相同，则合并为1个分箱
    当一个分箱内只有bad 或者 good时，修正公式公式计算中，加入 eps
    如果训练集woe满足单调性；but 验证集或测试集上不满足，则分箱不合理或者这个特征不稳定，时效性差
    :param df
    :param y 二值变量  定义 坏=1  好=0
    """
    # 好人比例 = rate_good_over_allgood
    # 坏人比例 = rate_bad_over_allbad
    df = get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    group_cols = ['lbl', 'lbl_index', 'lbl_left']
    # 分组对y 进行计算
    gp = pd.pivot_table(df, values=y, index=group_cols,
                        sort='lbl_index', aggfunc=['count', 'sum'],
                        fill_value=0, margins=True, observed=True)
    gp.columns = ['cnt', 'cnt_bad']
    gp['cnt_good'] = gp['cnt'] - gp['cnt_bad']
    gp['rate_bad'] = np.round(gp['cnt_bad'] / gp['cnt'], 3)
    # 坏样本占整体坏样本比例
    gp['rate_bad_over_allbad'] = np.round(gp['cnt_bad'] / gp.loc['All', 'cnt_bad'].values[0], 3)
    # 好样本占整体好样本比例
    gp['rate_good_over_allgood'] = np.round(gp['cnt_good'] / gp.loc['All', 'cnt_good'].values[0], 3)
    # 极小值
    eps = np.finfo(np.float32).eps
    gp['woe'] = np.round(np.log((gp['rate_bad_over_allbad'] + eps) / (gp['rate_good_over_allgood'] + eps)), 3)
    gp['iv_bin'] = np.round((gp['rate_bad_over_allbad'] - gp['rate_good_over_allgood']) * gp['woe'], 4)
    gp.loc['All', ['woe', 'iv_bin']] = None
    gp.reset_index(inplace=True)
    return gp


def iv(df: pd.DataFrame, x: str, y: str, feature_grid=[], cut_type=1, n_bin=10):
    """
    对 x 进行分组，对于每个组内的iv_i 求和
    iv_i =  (坏人比例-好人比例) * woe_i
    iv = sum(iv_i)
    坏人比例 =  组内的坏人数/ 总坏人数
    好人比例 = 组内的好人数/ 总好人数
    :return   <0.02 (无用特征)  0.02~0.1（弱特征） 0.1~0.3（中价值）0.3~0.5（高价值）>0.5(数据不真实)
    """
    gp = woe(df, x, y, feature_grid, cut_type, n_bin)
    return np.round(np.sum(gp['iv_bin']), 2)

def corr_target(df, feature_cols, target) -> list:
    """
    过滤掉 feature 同 targe 不相关的特征
    计算指标 pearsonr & spearmanr=(变量排序 + pearsonr) & kendalltau(有序性)
    :return:返回 有相关性的特征列表
    0~0.2 无相关或者积弱
    0.2~0.4 弱相关
    0.4~0.6 中等相关
    0.6~0.8 强相关
    0.8~1  极强相关
    """
    data = []
    for f in feature_cols:
        s1, p1 = stats.pearsonr(df[f], df[target])
        s2, p2 = stats.spearmanr(df[f], df[target], nan_policy='omit')
        s3, p3 = stats.kendalltau(df[f], df[target], nan_policy='omit')
        data.append([f, s1, p1, s2, p2, s3, p3])
    df_corr = pd.DataFrame(data, columns=['feature_name', 'pearsonr', 'pearsonr_pvalue',
                                          'spearmanr', 'spearmanr_pvalue',
                                          'kendall', 'kendall_pvalue'])

    return df_corr