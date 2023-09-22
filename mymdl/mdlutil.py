import pandas as pd
import numpy as np
from typing import Union

from sklearn import metrics
from scipy import stats
import plotly.graph_objects as go

from model_tools.mymdl import metricutil
import logging

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.precision', 3)
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 2000)



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
           cut_type=1, n_bin=10, group_cols=[]) -> pd.DataFrame:
    """
    单变量分布：对 x 进行分组，求每组的y的均值
    :param x df 中的字段名称
    :param feature_grid cut_type n_bin
    :param group_cols 分组统计
    """
    if df is None or len(df) == 0:
        return None
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = metricutil.get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    # 对应的y mean 计算
    if group_cols is None or len(group_cols) == 0:
        cls_cols = ['lbl', 'lbl_index', 'lbl_left']
    else:
        cls_cols = group_cols + ['lbl', 'lbl_index', 'lbl_left']
    gp = pd.pivot_table(df, values=y, index=cls_cols,
                        sort='lbl_index', aggfunc=['count', 'sum'],
                        fill_value=0, margins=False, observed=True)
    # 分组计算 y 的数量
    gp.columns = ['cnt', 'sum']
    gp['avg'] = np.round(gp['sum'] / gp['cnt'], 3)
    gp.reset_index(inplace=True)
    return gp


def accumvar(df: pd.DataFrame, x: str, y: str, feature_grid=[],
             cut_type=1, n_bin=10, group_cols=[]) -> pd.DataFrame:
    """
    变量累计分布 对x 进行分组，然后累计计算每组y的均值和数量
    :param x df 中的字段名称
    :param feature_grid cut_type n_bin
    :param group_cols 分组统计
    """
    if df is None:
        return None
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = metricutil.get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    cls_cols = group_cols
    if len(group_cols) == 0:
        cls_cols = ['lbl_index', 'lbl', 'lbl_left']
    else:
        # 如果是 cls_cols.extend，则会更新 group_cols
        cls_cols = cls_cols + ['lbl_index', 'lbl', 'lbl_left']
    gp = pd.pivot_table(df, values=y, index=cls_cols,
                        sort='lbl_index', aggfunc=['count', 'sum'],
                        fill_value=0, margins=False, observed=True)
    gp.columns = ['cnt', 'sum']
    gp['avg'] = np.round(gp['sum'] / gp['cnt'], 3)
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


def liftvar(df: pd.DataFrame, x: str, y: str, feature_grid=[],
            cut_type=1, n_bin=10, group_cols=[]) -> pd.DataFrame:
    """
    变量lift 分布，适用于y值二分类,对 x 变量进行分组
    :param y  定义 坏=1   好=0
    :param group_cols 分组统计
    :param feature_grid cut_type[1:等频分布 2:等宽分布] n_bin 分组的参数
    """
    if df is None:
        return None
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = metricutil.get_bin(df, x, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    cls_cols = group_cols
    if group_cols is None or len(group_cols) == 0:
        cls_cols = ['lbl', 'lbl_index', 'lbl_left']
    else:
        # extend : 会 更新 group_cols的数据
        cls_cols = cls_cols + ['lbl', 'lbl_index', 'lbl_left']

    # 分组对y 进行计算
    gp = pd.pivot_table(df, values=y, index=cls_cols,
                        sort='lbl_index', aggfunc=['count', 'sum'],
                        fill_value=0, margins=False, observed=True)
    gp.columns = ['cnt', 'cnt_bad']
    gp['cnt_good'] = gp['cnt'] - gp['cnt_bad']
    gp['rate_bad'] = np.round(gp['cnt_bad'] / gp['cnt'], 3)

    if group_cols is None or len(group_cols) == 0:
        # 累计
        gp['accum_cnt_bad'] = gp['cnt_bad'].cumsum()
        gp['accum_cnt_good'] = gp['cnt_good'].cumsum()

        # 坏样本占整体坏样本比例
        gp['accum_rate_bad_over_allbad'] = np.round(gp['accum_cnt_bad'] / gp['cnt_bad'].sum(), 3)
        # 好样本占整体好样本比例
        gp['accum_rate_good_over_allgood'] = np.round(gp['accum_cnt_good'] / gp['cnt_good'].sum(), 3)
        # lift = bad_over_allbad_rate / bad_rate
        # 整体bad_rate
        all_bad_rate = np.round(gp['cnt_bad'].sum() / gp['cnt'].sum(), 4)
        gp['accum_lift_bad'] = np.round(gp['accum_rate_bad_over_allbad'] / all_bad_rate, 3)
    else:
        gp['accum_cnt_bad'] = gp.groupby(group_cols)['cnt_bad'].cumsum()
        gp['accum_cnt_good'] = gp.groupby(group_cols)['cnt_good'].cumsum()
        tmp = gp.groupby(group_cols).agg(all_cnt_bad=('cnt_bad', 'sum'),
                                         all_cnt_good=('cnt_good', 'sum')).reset_index()
        tmp['all_rate_bad'] = np.round(tmp['all_cnt_bad'] / (tmp['all_cnt_bad'] + tmp['all_cnt_good']), 3)
        # index = group_cols + ['lbl', 'lbl_index', 'lbl_left']
        # tmp = index = group_cols 需要对其，否则丢失字段
        gp = gp.reset_index()
        gp = gp.merge(tmp, on=group_cols, how='left')

        # 坏样本占整体坏样本比例
        gp['accum_rate_bad_over_allbad'] = np.round(gp['accum_cnt_bad'] / gp['all_cnt_bad'], 3)
        # 好样本占整体好样本比例
        gp['accum_rate_good_over_allgood'] = np.round(gp['accum_cnt_good'] / gp['all_cnt_good'], 3)
        # lift = bad_over_allbad_rate / bad_rate
        gp['accum_lift_bad'] = np.round(gp['accum_rate_bad_over_allbad'] / gp['all_rate_bad'], 3)
        # 删除
        gp.drop(['all_cnt_bad', 'all_cnt_good', 'all_rate_bad'], axis=1, inplace=True)
    gp.reset_index(inplace=True)
    return gp


def evaluate_twoscores_lift(df, f1, f2, target, n_bin=10, show_flat=True):
    """
    评估2个模型分的联合的lift 变化
    :param f1 f2 ::: feature_name
    :param target:
    :param show_flat 数据展示模型，True 展开；False 堆叠
    :return:
    """
    cols = [target, f1, f2]
    df = df[cols]
    df = metricutil.get_bin(df, f1, cut_type=1, n_bin=n_bin)
    df.rename(columns={'lbl': '%s_lbl' % f1, 'lbl_index': '%s_lbl_index' % f1}, inplace=True)
    df = metricutil.get_bin(df, f2, cut_type=1, n_bin=n_bin)
    df.rename(columns={'lbl': '%s_lbl' % f2, 'lbl_index': '%s_lbl_index' % f2}, inplace=True)
    ix = '%s_lbl' % f1
    column = '%s_lbl' % f2
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
    t_accum_rate = np.round(t_accum_cnt / all_cnt, 2)
    t_accum_rate = t_accum_rate.stack().reset_index().rename(columns={0: 'accum_cnt_rate'})
    gp_out = gp_out.merge(t_accum_rate, on=[ix, column], how='outer')
    # 累计所有坏的比例
    t_accum_bad = gp['sum'].cumsum(axis=1).cumsum(axis=0)
    t_accum_bad = t_accum_bad / all_bad_cnt
    # lift
    t_accum_bad = t_accum_bad / all_bad_rate
    t_accum_bad = t_accum_bad.stack().reset_index().rename(columns={0: 'accum_lift_bad'})
    gp_out = gp_out.merge(t_accum_bad, on=[ix, column], how='outer')
    # # 设置列名
    # mix = []
    # for c in t_accum_bad.columns.categories:
    #     mix.append(('accum_lift_bad', c))
    # t_accum_bad.columns = pd.MultiIndex.from_tuples(mix, names=[None, column])
    if not show_flat:
        # 堆叠模式
        gp_out = gp_out.set_index([ix, column])
        gp_out = gp_out.stack().reset_index().rename(columns={'level_2': 'key', 0: 'value'})
        gp_out = pd.pivot_table(gp_out, values=['value'], index=ix,
                                columns=[column, 'key'],
                                aggfunc=np.mean, sort=False)
        gp_out = gp_out['value']
    return gp_out


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
            go.Scatter(x=fpr, y=ks, mode='lines', name='ks'),
            # ks 最大值
            go.Scatter(x=[ks_max_x], y=[ks], name='ks-max', size=20)
        ]
        fig = go.Figure(data)
        fig.add_annotation(dict(font=dict(color='rgba(0,0,200,0.8)', size=12),
                                x=ks_max_x,
                                y=ks,
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
