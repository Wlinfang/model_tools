import pandas as pd
import numpy as np
from typing import Union, Tuple

from sklearn import metrics
from scipy import stats
import plotly.graph_objects as go

from model_tools.utils import toolutil
import logging

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.precision', 3)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)


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
                           right=False, precision=4, duplicates='drop')

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


def describe_df(df, feature_names: list) -> pd.DataFrame:
    """
    描述性分析
    :param feature_names  特征名称列表
    """
    # 提取数字型数据
    col_num_list = df[feature_names].select_dtypes(include=np.number).columns.tolist()
    # 字段类型
    df_num = df[feature_names].dtypes.to_frame().rename(columns={0: 'dtype'})
    # count,miss_rate
    tmp = df[feature_names].agg(['count']).T
    tmp['miss_rate'] = np.round(1 - tmp['count'] / df.shape[0], 3)
    tmp['miss_rate'] = tmp['miss_rate'].transform(lambda x: format(x, '.2%'))
    df_num = df_num.merge(tmp, how='outer', left_index=True, right_index=True)
    # nunique
    tmp = df[feature_names].agg(['nunique']).T
    df_num = df_num.merge(tmp, how='outer', left_index=True, right_index=True)
    # freq,freq_count
    tmp = []
    for f in feature_names:
        # 统计出现频数
        t = df[f].value_counts(dropna=True)
        if (t is not None) and (t.shape[0] > 0):
            t = t.head(1)
            freq, freq_count = t.index[0], t.values[0]
            tmp.append([f, freq, freq_count])
    tmp = pd.DataFrame(tmp, columns=['index', 'freq', 'freq_count'])
    tmp.set_index('index', inplace=True)
    df_num = df_num.merge(tmp, how='outer', left_index=True, right_index=True)

    # max,min,mean,%25,%50,%75,
    if col_num_list:
        # 数字型
        # min ,max,mean
        tmp = df[col_num_list].agg(['min', 'max', 'mean', 'std']).T
        df_num = df_num.merge(tmp, how='outer', left_index=True, right_index=True)
        # %25,%50,%75
        tmp = df[col_num_list].quantile([0.25, 0.5, 0.75], interpolation='midpoint').T \
            .rename(columns={0.25: '25%', 0.5: '50%', 0.75: '75%'})
        df_num = df_num.merge(tmp, how='outer', left_index=True, right_index=True)
    return df_num


def histvar(df, x, feature_grid=[], cut_type=2, n_bin=10, group_cols=[]):
    """
    变量x的分布图，默认等宽分布统计
    """
    df = get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    cls_cols = group_cols
    if len(group_cols) == 0:
        cls_cols = ['lbl']
    else:
        # 如果是 cls_cols.extend，则会更新 group_cols
        cls_cols = cls_cols + ['lbl']
        gp_all = df.groupby(group_cols).size().reset_index().rename(columns={0: 'cnt_all'})
    gp = df.groupby(cls_cols).size().reset_index().rename(columns={0: 'cnt'})
    gp['cnt'] = gp['cnt'].astype(int)
    if len(group_cols) == 0:
        gp['cnt_all'] = df.shape[0]
    else:
        gp = gp.merge(gp_all, on=group_cols, how='inner')
    gp['cnt_rate'] = np.round(gp['cnt'] / gp['cnt_all'], 3)
    return gp


def univar(df: pd.DataFrame, x: str, y: str, feature_grid=[],
           cut_type=1, n_bin=10, group_cols=[]) -> pd.DataFrame:
    """
    变量累计分布 对x 进行分组，然后累计计算每组y的均值和数量
    :param x df 中的字段名称
    :param feature_grid cut_type n_bin
    :param group_cols 分组统计
    :return cnt sum avg accum_cnt,accum_sum,accum_avg
    """
    if df is None:
        return None
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    cls_cols = group_cols
    if len(group_cols) == 0:
        cls_cols = ['lbl_index', 'lbl']
    else:
        # 如果是 cls_cols.extend，则会更新 group_cols
        cls_cols = cls_cols + ['lbl_index', 'lbl']
    gp = pd.pivot_table(df, values=y, index=cls_cols,
                        sort='lbl_index', aggfunc=['count', 'sum', 'mean'],
                        fill_value=0, margins=False, observed=True)
    gp.columns = ['cnt', 'sum', 'avg']
    gp['avg'] = np.round(gp['avg'], 3)
    if group_cols is None or len(group_cols) == 0:
        gp['accum_cnt'] = gp['cnt'].cumsum()
        gp['accum_sum'] = gp['sum'].cumsum()
        gp['accum_avg'] = np.round(gp['accum_sum'] / gp['accum_cnt'], 3)
    else:
        gp['accum_cnt'] = gp.groupby(group_cols)['cnt'].cumsum()
        gp['accum_sum'] = gp.groupby(group_cols)['sum'].cumsum()
        gp['accum_avg'] = np.round(gp['accum_sum'] / gp['accum_cnt'], 3)
    gp = gp.reset_index()
    return gp


def binary_liftvar(df: pd.DataFrame, x: str, y: str, feature_grid=[],
                   cut_type=1, n_bin=10, group_cols=[]) -> pd.DataFrame:
    """
    变量lift 分布，适用于y值二分类,对 x 变量进行分组
    :param y  定义 坏=1   好=0
    :param group_cols 分组统计
    :param feature_grid cut_type[1:等频分布 2:等宽分布] n_bin 分组的参数
    :return 'cnt', 'cnt_bad', 'rate_bad', 'accum_cnt', 'accum_cnt_bad', 'accum_rate_bad', 'accum_lift_bad'
    """
    if df is None:
        return None
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    cls_cols = group_cols
    if group_cols is None or len(group_cols) == 0:
        cls_cols = ['lbl', 'lbl_index', 'lbl_left']
    else:
        # extend : 会 更新 group_cols的数据
        cls_cols = cls_cols + ['lbl', 'lbl_index', 'lbl_left']

    # 分组对y 进行计算
    gp = pd.pivot_table(df, values=y, index=cls_cols,
                        sort='lbl_index', aggfunc=['count', 'sum'],
                        margins=False, observed=True)
    gp.columns = ['cnt', 'cnt_bad']
    # gp['cnt_good'] = gp['cnt'] - gp['cnt_bad']
    gp['rate_bad'] = np.round(gp['cnt_bad'] / gp['cnt'], 3)

    if group_cols is None or len(group_cols) == 0:
        # 累计
        gp['accum_cnt'] = gp['cnt'].cumsum()
        gp['accum_cnt_bad'] = gp['cnt_bad'].cumsum()
        # gp['accum_cnt_good'] = gp['cnt_good'].cumsum()
        gp['accum_rate_bad'] = np.round(gp['accum_cnt_bad'] / gp['accum_cnt'], 2)

        # 坏样本占整体坏样本比例
        # gp['accum_rate_bad_over_allbad'] = np.round(gp['accum_cnt_bad'] / gp['cnt_bad'].sum(), 3)
        # 好样本占整体好样本比例
        # gp['accum_rate_good_over_allgood'] = np.round(gp['accum_cnt_good'] / gp['cnt_good'].sum(), 3)
        # lift = accum_rate_bad / bad_rate
        # 整体bad_rate
        all_bad_rate = np.round(gp['cnt_bad'].sum() / gp['cnt'].sum(), 4)
        gp['lift_bad'] = np.round(gp['accum_rate_bad'] / all_bad_rate, 3)
    else:
        gp['accum_cnt'] = gp.groupby(group_cols)['cnt'].cumsum()
        gp['accum_cnt_bad'] = gp.groupby(group_cols)['cnt_bad'].cumsum()
        # gp['accum_cnt_good'] = gp.groupby(group_cols)['cnt_good'].cumsum()
        gp['accum_rate_bad'] = np.round(gp['accum_cnt_bad'] / gp['accum_cnt'], 2)
        tmp = gp.groupby(group_cols).agg(all_cnt_bad=('cnt_bad', 'sum'),
                                         all_cnt=('cnt', 'sum')).reset_index()
        tmp['all_rate_bad'] = np.round(tmp['all_cnt_bad'] / tmp['all_cnt'], 3)
        # index = group_cols + ['lbl', 'lbl_index', 'lbl_left']
        # tmp = index = group_cols 需要对其，否则丢失字段
        gp = gp.reset_index()
        gp = gp.merge(tmp, on=group_cols, how='left')

        # # 坏样本占整体坏样本比例
        # gp['accum_rate_bad_over_allbad'] = np.round(gp['accum_cnt_bad'] / gp['all_cnt_bad'], 3)
        # # 好样本占整体好样本比例
        # gp['accum_rate_good_over_allgood'] = np.round(gp['accum_cnt_good'] / gp['all_cnt_good'], 3)
        # lift = bad_over_allbad_rate / bad_rate
        # gp['accum_lift_bad'] = np.round(gp['accum_rate_bad_over_allbad'] / gp['all_rate_bad'], 3)
        gp['lift_bad'] = np.round(gp['accum_rate_bad'] / gp['all_rate_bad'], 3)
        # 删除
        gp.drop(['all_cnt_bad', 'all_cnt', 'all_rate_bad'], axis=1, inplace=True)
    gp.reset_index(inplace=True)
    cols = cls_cols + ['cnt', 'cnt_bad', 'rate_bad', 'accum_cnt', 'accum_cnt_bad', 'accum_rate_bad', 'lift_bad']
    return gp[cols]


def twoscores_binary_liftvar(df, f1_score, f2_score, target, f1_grid=[], f2_grid=[], n_bin=10) -> pd.DataFrame:
    """
    评估2个模型分的联合的lift 变化
    :param f1 f2 ::: feature_name
    :param target:
    :param show_flat 数据展示模型，True 展开；False 堆叠
    :return:'cnt', 'rate_bad', 'accum_cnt_rate', 'accum_rate_bad', 'accum_lift_bad'
    """
    cols = [target, f1_score, f2_score]
    df = df[cols]
    df = get_bin(df, f1_score, feature_grid=f1_grid, cut_type=1, n_bin=n_bin)
    df.rename(columns={'lbl': '%s_lbl' % f1_score, 'lbl_index': '%s_lbl_index' % f1_score}, inplace=True)
    df = get_bin(df, f2_score, feature_grid=f2_grid, cut_type=1, n_bin=n_bin)
    df.rename(columns={'lbl': '%s_lbl' % f2_score, 'lbl_index': '%s_lbl_index' % f2_score}, inplace=True)
    ix = '%s_lbl' % f1_score
    column = '%s_lbl' % f2_score
    gp = pd.pivot_table(df, values=[target], index=ix,
                        columns=column, fill_value=0,
                        aggfunc=('count', 'sum', 'mean'), observed=True, )
    gp = gp[target]
    # 所有的数量
    all_cnt = df.shape[0]
    all_bad_cnt = df[target].sum()
    all_bad_rate = np.round(all_bad_cnt / all_cnt, 3)
    # 分组数量计算
    t_cnt = gp['count']
    t_cnt = t_cnt.stack().reset_index().rename(columns={0: 'cnt'})
    # 每组坏比例
    t_bad_rate = gp['mean']
    t_bad_rate = t_bad_rate.stack().reset_index().rename(columns={0: 'rate_bad'})
    gp_out = t_cnt.merge(t_bad_rate, on=[ix, column], how='outer')
    # 累计总量
    t_accum_cnt = gp['count'].cumsum(axis=1).cumsum(axis=0)
    # 累计总量占比
    t_accum_rate = np.round(t_accum_cnt / all_cnt, 3)
    t_accum_rate = t_accum_rate.stack().reset_index().rename(columns={0: 'accum_cnt_rate'})
    gp_out = gp_out.merge(t_accum_rate, on=[ix, column], how='outer')
    # 累计所有坏的比例
    t_accum_bad = gp['sum'].cumsum(axis=1).cumsum(axis=0)
    t_accum_rate_bad = np.round(t_accum_bad / t_accum_cnt, 3)
    # lift
    t_accum_lift = np.round(t_accum_rate_bad / all_bad_rate, 3)
    # 累计坏比例
    t_accum_rate_bad = t_accum_rate_bad.stack().reset_index().rename(columns={0: 'accum_rate_bad'})
    gp_out = gp_out.merge(t_accum_rate_bad, on=[ix, column], how='outer')

    t_accum_lift = t_accum_lift.stack().reset_index().rename(columns={0: 'accum_lift_bad'})
    gp_out = gp_out.merge(t_accum_lift, on=[ix, column], how='outer')

    # if not show_flat:
    # 堆叠模式
    # gp_out = gp_out.set_index([ix, column])
    # gp_out = gp_out.stack().reset_index().rename(columns={'level_2': 'key', 0: 'value'})
    # gp_out = pd.pivot_table(gp_out, values=['value'], index=ix,
    #                         columns=[column, 'key'],
    #                         aggfunc=np.mean, sort=False)
    # gp_out = gp_out['value']
    gp_out = pd.pivot_table(gp_out,
                            values=['cnt', 'rate_bad', 'accum_cnt_rate', 'accum_rate_bad', 'accum_lift_bad'],
                            index=ix,
                            columns=column,
                            aggfunc=np.mean, sort=False)
    return gp_out


def multiscores_binary_liftvar(df, model_scores: list, target, n_bin=10) -> pd.DataFrame:
    """
    评估多个模型分的联合的lift 变化,适用于二分类目标变量的评估
    :param model_scores:模型分
    :param target:目标变量，二分类 0，1 变量
    :param show_flat 数据展示模型，True 展开；False 堆叠
    :return:'cnt', 'rate_bad', 'accum_cnt_rate', 'accum_rate_bad', 'accum_lift_bad'
    """
    cols = [target] + model_scores
    df = df[cols]
    # 每个模型分分桶
    for score in model_scores:
        df = get_bin(df, score, cut_type=1, n_bin=n_bin)
        df.rename(columns={'lbl': '%s_lbl' % score, 'lbl_index': '%s_lbl_index' % score}, inplace=True)
    ixes = ['%s_lbl' % f for f in model_scores]
    # 空值不计入
    gp = pd.pivot_table(df, values=[target], index=ixes,
                        aggfunc=('count', 'sum', 'mean'), observed=True, )
    gp = gp[target]
    gp.rename(columns={'count': 'cnt', 'sum': 'cnt_bad', 'mean': 'rate_bad'}, inplace=True)
    # 所有的数量
    all_cnt = df.shape[0]
    all_bad_cnt = df[target].sum()
    all_bad_rate = np.round(all_bad_cnt / all_cnt, 3)
    gp['accum_cnt'] = gp['cnt'].cumsum()
    gp['accum_cnt_bad'] = gp['cnt_bad'].cumsum()
    gp['accum_cnt_rate'] = np.round(gp['accum_cnt'] / all_cnt, 2)
    gp['accum_rate_bad'] = np.round(gp['accum_cnt_bad'] / gp['accum_cnt'], 2)
    gp['lift_bad'] = np.round(gp['rate_bad'] / all_bad_rate, 3)
    gp = gp.reset_index()
    return gp


def evaluate_binary_classier(y_true: Union[list, pd.Series, np.array],
                             y_pred: Union[list, pd.Series, np.array],
                             is_show=False):
    """ 二分类模型评估指标
    ...
    混淆矩阵
    -------
                      真实值
                1                 0
    预测值  1    TP               FP
           0    FN               TN
    | 准确率 Accuracy = (TP+TN) / (TP+FP+FN+TN)
    | 精确率(Precision) = TP / (TP+FP) --- 预测为正样本 分母为预测正样本
    | 召回率(Recall) = TP / (TP+FN) -- 实际为正样本 分母为真实正样本
    | F1分数(F1-Score) = 2 *  精确率 * 召回率 / (精确率 + 召回率)
    | P-R曲线 ： y-axis = 精确率；x-axis= 召回率  平衡点 ： 精确率 = 召回率
    | 真正率(TPR) = TP / (TP+FN)-- 以真实样本 分母为真实正样本
    | 假正率(FPR) = FP / (FP+TN)-- 以真实样本 分母为真实负样本
    | Roc 曲线：y-axis=真正率 ; x-axis=假正率； 无视样本不均衡问题
    :return cnt,auc,ks,gini
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        return None, None, None, None
    if len(np.unique(y_true)) == 1:
        # only one class
        logging.info('only one class !!!')
        return None, None, None, None
    # 返回 精确率，召回率，F1
    fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)
    auc = np.round(metrics.roc_auc_score(y_true, y_pred), 3)
    # ks 曲线
    ks_detail = np.round(abs(tpr - fpr), 3)
    # 标注最大值
    ks_max_index = ks_detail.argmax()
    ks_max_x = fpr[ks_max_index]
    ks = ks_detail[ks_max_index]
    # roc 图，ks 图
    # x=fpr;;; y = tpr  ks
    gini = np.round(2 * auc - 1, 3)
    if is_show:
        data = [
            # roc-auc 图
            go.Scatter(x=fpr, y=tpr, mode='lines', name='roc-auc'),
            # ks 图
            go.Scatter(x=fpr, y=ks_detail, mode='lines', name='ks'),
            # ks 最大值
            go.Scatter(x=[ks_max_x], y=[ks], name='ks-max')
        ]
        fig = go.Figure(data)
        fig.add_annotation(dict(font=dict(color='rgba(0,0,200,0.8)', size=14),
                                x=ks_max_x,
                                y=ks + 0.05,
                                showarrow=False,
                                text='ks = ' + str(ks) + '  ',
                                textangle=0,
                                xanchor='auto',
                                xref="x",
                                yref="y"))

        title = 'count:{} rate_bad:{} auc:{} ks:{} gini:{}'.format(
            len(y_true), np.round(np.mean(y_true), 3),
            auc, ks, gini
        )
        # uniformtext_minsize 调整标注文字大小；uniformtext_mode 不合规数字隐藏
        fig.update_layout(title=title, uniformtext_minsize=2, uniformtext_mode='hide')
        fig.show()
    return len(y_true), auc, ks, gini


def evaluate_binary_classier_bygroup(df, y_true: str, y_pred: str, group_cols=[]) -> pd.DataFrame:
    # 分组计算 auc,ks,gini
    if group_cols is not None and len(group_cols) > 0:
        gp_auc = df[df[y_pred].notna()].groupby(group_cols).apply(
            lambda x: evaluate_binary_classier(x[y_true], x[y_pred], is_show=False))
        gp_auc = gp_auc.reset_index().rename(columns={0: 'value'})
        gp_auc.loc[:, ['cnt', 'auc', 'ks', 'gini']] = gp_auc['value'].apply(pd.Series,
                                                                            index=['cnt', 'auc', 'ks', 'gini'])
        gp_auc.drop(['value'], axis=1, inplace=True)
    else:
        cnt, auc, ks, gini = evaluate_binary_classier(df[df[y_pred].notna()][y_true],
                                                      df[df[y_pred].notna()][y_pred])
        gp_auc = pd.DataFrame([[cnt, auc, ks, gini]], columns=['cnt', 'auc', 'ks', 'gini'], index=['all'])
    return gp_auc


def evaluate_multi_classier(y_true, y_pred):
    """
    适用于多分类 y_pred 为预测类别值
    balanced_accuracy_score ： 1/n (sum( 类别i预测对的数量 / 类别i的数量) )
    cohen_kappa_score : 判断评估性一致性  -1~1
         <0 评估随机性   1 perfect   0.8~1 almost perfect
    matthews_corrcoef: -1~1  值越大，一致性越高
    :return cnt,balanced_accuracy,cohen_kappa,mcc
    """
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred, )
    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)
    # show
    cm = metrics.confusion_matrix(y_true, y_pred, normalize='all')
    metrics.ConfusionMatrixDisplay(cm)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    return len(y_true), balanced_accuracy, cohen_kappa, mcc


def evaluate_regression(y_true, y_pred):
    """
    回归评估
    r2
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    return mse, r2


def evaluate_ranking():
    """
    评估排序效果
    :return:
    """
    pass


class Confidence:
    """
    假设估计置信度区间计算
    confidence：置信度
    is_biased_estimate：计算标准差是有偏还是无偏；True:有偏；利用样本估计总体标准差
    """

    def __init__(self, confidence=0.95, is_biased_estimate=False):
        super(Confidence, self).__init__()
        self.confidence = confidence
        self.is_biased_estimate = is_biased_estimate

    def check_norm_confidence(self, df, feature_name):
        '''
        计算正太分布的置信区间
        confidence:置信度
        is_biased_estimate：计算标准差是有偏还是无偏；True:有偏；利用样本估计总体标准差
        '''
        sample_mean = df[feature_name].mean()
        if self.is_biased_estimate:
            # 有偏估计
            sample_std = df[feature_name].std(ddof=0)
        else:
            # 无偏
            sample_std = df[feature_name].std(ddof=1)
        return stats.norm.interval(self.confidence, loc=sample_mean, scale=sample_std)

    def check_t_confidence(self, df, feature_name):
        '''
        计算t分布的置信区间
        '''
        sample_mean = df[feature_name].mean()
        if self.is_biased_estimate:
            # 有偏估计
            sample_std = df[feature_name].std(ddof=0)
        else:
            # 无偏
            sample_std = df[feature_name].std(ddof=1)

        return stats.t.interval(self.confidence, df=(df.shape[0] - 1), loc=sample_mean, scale=sample_std)
