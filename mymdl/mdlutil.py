import pandas as pd
import numpy as np
from typing import Union
from sklearn import metrics
import plotly.graph_objects as go

from model_tools.utils.toolutil import del_none

import logging

logger = logging.getLogger(__file__)


def get_feature_grid(values: Union[list, np.array],
                     cut_type=1, n_bin=10, default_values=[]) -> list:
    """
    计算分桶
    :param values: 要进行分桶的数据
    :param cut_type: 1 : 等频; 2:等宽 3、
    :return
    """
    if (values is None) or (len(values) < 1) or (n_bin <= 0):
        logger.info('param values is %s', values)
        return None
    values = np.array(values)
    # 限制为 1d
    if values.ndim > 1:
        logger.info('param values is %s', values)
        return None
    # 去除空值、空字符串
    values = del_none(values)
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
        f = [-np.Inf] + vs_sort.tolist()
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
        logger.error('data is None ')
        return None
    df[feature_name] = pd.to_numeric(df[feature_name], errors='ignore')
    # 分为空和非空
    t1 = df[(df[feature_name].notna()) & (~df[feature_name].isin(default_values))].copy()
    t2 = df[~df.index.isin(t1.index)].copy()

    if not pd.api.types.is_numeric_dtype(df[feature_name]):
        # 非数字型
        t1['lbl'] = t1[feature_name]
    if not feature_grid:
        # 如果未指定
        feature_grid = get_feature_grid(df[feature_name], cut_type, n_bin
                                        , default_values)
        if len(feature_grid) == 0:
            logger.error('feature_grid is None ')
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


def split_train_val(df: pd.DataFrame, split_type=1, split_ratio=0.8, sort_col=None):
    """
    将数据集进行切分
    :param split_type  1：随机切分  2：按照时间切分
    :param split_ratio  切分比例； 取值范围：(0,1)
    :param sort_col：如果 split_type=2 根据sort_col 排序然后切分
    :return: df_train,df_val
    """
    dftrain = df.reset_index()
    # == dftrain 中划分 训练集，验证集
    if split_type == 1:
        # 随机分配 train / val
        train = dftrain.sample(frac=split_ratio, random_state=7)
        val = dftrain[~dftrain.index.isin(train.index)]
    elif split_type == 2:
        # 按时间序列分配 train /val
        train = dftrain.sort_values(by=sort_col).head(int(len(dftrain) * split_ratio))
        val = dftrain[~dftrain.index.isin(train.index)]
    else:
        raise ValueError('param error or data error')
    return train, val


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


def univar(df: pd.DataFrame, x: str, y: str, feature_grid=[],
           cut_type=1, n_bin=10) -> pd.DataFrame:
    """
    单变量分布：对 x 进行分组，求每组的y的均值
    :param x df 中的字段名称
    :param feature_grid cut_type n_bin
    """
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    # 对应的y mean 计算
    group_cols = ['lbl', 'lbl_index', 'lbl_left']
    gp = pd.pivot_table(df, values=y, index=group_cols,
                        sort='lbl_index', aggfunc=['count', 'sum'],
                        fill_value=0, margins=False, observed=True)
    # 分组计算 y 的数量
    gp.columns = ['cnt', 'sum']
    gp['avg'] = np.round(gp['sum'] / gp['cnt'], 3)
    gp.reset_index(inplace=True)
    return gp


def accumvar(df: pd.DataFrame, x: str, y: str, feature_grid=[],
             cut_type=1, n_bin=10) -> pd.DataFrame:
    """
    变量累计分布 对x 进行分组，然后累计计算每组y的均值和数量
    :param x df 中的字段名称
    :param feature_grid cut_type n_bin
    """
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    group_cols = ['lbl_index', 'lbl', 'lbl_left']
    gp = pd.pivot_table(df, values=y, index=group_cols,
                        sort='lbl_index', aggfunc=['count', 'sum'],
                        fill_value=0, margins=False, observed=True)
    gp.columns = ['cnt', 'sum']
    gp['avg'] = np.round(gp['sum'] / gp['cnt'], 3)
    gp['accum_cnt'] = gp['cnt'].cumsum()
    gp['accum_sum'] = gp['sum'].cumsum()
    gp['accum_avg'] = np.round(gp['accum_sum'] / gp['accum_cnt'], 3)
    return gp


def liftvar(df: pd.DataFrame, x: str, y: str, feature_grid=[],
            cut_type=1, n_bin=10) -> pd.DataFrame:
    """
    变量lift 分布，适用于y值二分类,对 x 变量进行分组
    :param y  定义 坏=1   好=0
    :param feature_grid cut_type[1:等频分布 2:等宽分布] n_bin 分组的参数
    """
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    group_cols = ['lbl', 'lbl_index', 'lbl_left']
    # 分组对y 进行计算
    gp = pd.pivot_table(df, values=y, index=group_cols,
                        sort='lbl_index', aggfunc=['count', 'sum'],
                        fill_value=0, margins=True, observed=True)
    gp.columns = ['cnt', 'cnt_bad']
    gp['cnt_good'] = gp['cnt'] - gp['cnt_bad']
    gp['rate_bad'] = np.round(gp['cnt_bad'] / gp['cnt'], 3)
    # 累计
    gp['accum_cnt_bad'] = gp['cnt_bad'].cumsum()
    gp['accum_cnt_good'] = gp['cnt_good'].cumsum()
    gp.loc['All', 'accum_cnt_bad'] = None
    gp.loc['All', 'accum_cnt_good'] = None
    # 坏样本占整体坏样本比例
    gp['accum_rate_bad_over_allbad'] = np.round(gp['accum_cnt_bad'] / gp.loc['All', 'cnt_bad'].values[0], 3)
    # 好样本占整体好样本比例
    gp['accum_rate_good_over_allgood'] = np.round(gp['accum_cnt_good'] / gp.loc['All', 'cnt_good'].values[0], 3)
    # lift = bad_over_allbad_rate / bad_rate
    gp['accum_lift_bad'] = np.round(gp['accum_rate_bad_over_allbad'] / gp.loc['All', 'rate_bad'].values[0], 3)
    gp.reset_index(inplace=True)
    return gp


def evaluate_binary_classier(y_true: Union[list, pd.Series, np.array], y_pred: Union[list, pd.Series, np.array]):
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
    """
    # 返回 精确率，召回率，F1
    fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)
    auc = np.round(metrics.roc_auc_score(y_true, y_pred),3)
    # ks 曲线
    ks = np.round(abs(tpr - fpr),3)
    # 标注最大值
    ks_max_index = ks.argmax()
    ks_max_y = ks[ks_max_index]
    ks_max_x = ks[ks_max_index]
    # roc 图，ks 图
    # x=fpr;;; y = tpr  ks
    gini = 2 * auc - 1
    data = [
        # roc-auc 图
        go.Scatter(x=fpr, y=tpr, mode='lines', name='roc-auc'),
        # ks 图
        go.Scatter(x=fpr, y=ks, mode='lines', name='ks'),
        # ks 最大值
        go.Scatter(x=[ks_max_x], y=[ks_max_y], name='ks-max')
    ]
    fig = go.Figure(data)
    fig.add_annotation(dict(font=dict(color='rgba(0,0,200,0.8)', size=12),
                            x=ks_max_x,
                            y=ks_max_y + 0.02,
                            showarrow=False,
                            text='ks = ' + str(ks_max_y) + '  ',
                            textangle=0,
                            xanchor='auto',
                            xref="x",
                            yref="y"))

    title = 'count:{} rate_bad:{} auc:{} ks:{} gini:{}'.format(
        len(y_true), np.round(np.mean(y_true), 3),
        auc, ks_max_y, gini
    )
    # uniformtext_minsize 调整标注文字大小；uniformtext_mode 不合规数字隐藏
    fig.update_layout(title=title, uniformtext_minsize=2, uniformtext_mode='hide')
    fig.show()


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
