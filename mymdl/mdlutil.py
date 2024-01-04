import pandas as pd
import numpy as np
from typing import Union, Tuple

from sklearn import metrics
from scipy import stats
import plotly.graph_objects as go

import logging

from ..mymdl import statsutil


# pd.set_option('display.float_format', lambda x: '%.3f' % x)
# pd.set_option('display.precision', 3)
# pd.set_option('display.max_rows', 2000)
# pd.set_option('display.max_columns', 2000)

def histvar(df, x, feature_grid=[], n_bin=10, group_cols=[]):
    """
    变量x的等宽分布图
    """
    df = statsutil.get_bin(df, x, feature_grid=feature_grid, cut_type=2, n_bin=n_bin)
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
    :param feature_grid cut_type n_bin  cut_type=[1:等频；2：等宽；3：卡方]
    :param group_cols 分组统计
    :return group_cols,lbl_index,lbl,lbl_left,cnt sum avg total,cnt_over_total,accum_cnt,accum_sum,accum_avg,accum_cnt_over_total
    """
    if df is None:
        return None
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = statsutil.get_bin(df, x, y, feature_grid=feature_grid, cut_type=cut_type, n_bin=n_bin)
    if len(group_cols) == 0:
        cls_cols = ['lbl_index', 'lbl', 'lbl_left']
    else:
        # 如果是 cls_cols.extend，则会更新 group_cols
        cls_cols = group_cols + ['lbl_index', 'lbl', 'lbl_left']
    gp = pd.pivot_table(df, values=y, index=cls_cols,
                        sort='lbl_index', aggfunc=['count', 'sum', 'mean'],
                        fill_value=0, margins=False, observed=True)

    gp.columns = ['cnt', 'sum', 'avg']
    gp = gp.reset_index()
    gp['avg'] = np.round(gp['avg'], 3)
    if group_cols is None or len(group_cols) == 0:
        gp['total'] = gp.cnt.sum()
        gp['cnt_over_total'] = np.round(gp['cnt'] / gp.total, 3)
        gp['accum_cnt'] = gp['cnt'].cumsum()
        gp['accum_sum'] = gp['sum'].cumsum()
        gp['accum_avg'] = np.round(gp['accum_sum'] / gp['accum_cnt'], 3)
        gp['accum_cnt_over_total'] = np.round(gp.accum_cnt / gp.total, 3)
    else:
        t = gp.groupby(group_cols)['cnt'].sum().reset_index().rename(columns={'cnt': 'total'})
        gp = gp.merge(t, on=group_cols, how='left')
        gp['cnt_over_total'] = np.round(gp['cnt'] / gp.total, 3)
        gp['accum_cnt'] = gp.groupby(group_cols)['cnt'].cumsum()
        gp['accum_sum'] = gp.groupby(group_cols)['sum'].cumsum()
        gp['accum_avg'] = np.round(gp['accum_sum'] / gp['accum_cnt'], 3)
        gp['accum_cnt_over_total'] = np.round(gp.accum_cnt / gp.total, 3)
    cols = group_cols + ['lbl', 'lbl_index', 'lbl_left', 'cnt', 'sum', 'avg', 'cnt_over_total', 'accum_cnt',
                         'accum_sum', 'accum_avg',
                         'accum_cnt_over_total', 'total']
    return gp[cols]


def binary_liftvar(df: pd.DataFrame, x: str, y: str, feature_grid=[],
                   cut_type=1, n_bin=10, group_cols=[]) -> pd.DataFrame:
    """
    变量lift 分布，适用于y值二分类,对 x 变量进行分组
    :param y  定义 坏=1   好=0
    :return 'lbl','lbl_index','lbl_left','cnt','cnt_bad','rate_bad','cnt_over_total','accum_cnt','accum_cnt_bad','accum_rate_bad',
                     'accum_cnt_over_total','lift_bad','total','all_bad_rate'
    """
    # group_cols,lbl_index,lbl,cnt sum avg total,cnt_over_total,accum_cnt,accum_sum,accum_avg,accum_cnt_over_total
    gp = univar(df, x, y, feature_grid, cut_type, n_bin, group_cols)
    gp = gp.rename(
        columns={'sum': 'cnt_bad', 'avg': 'rate_bad', 'accum_sum': 'accum_cnt_bad', 'accum_avg': 'accum_rate_bad'})
    # all rate
    if group_cols is None or len(group_cols) == 0:
        # lift = rate_bad / all_bad_rate
        all_bad_rate = np.round(gp['cnt_bad'].sum() / gp['cnt'].sum(), 4)
        gp['all_bad_rate'] = all_bad_rate
        gp['lift_bad'] = np.round(gp['rate_bad'] / all_bad_rate, 1)
    else:
        tmp = gp.groupby(group_cols).agg(all_cnt_bad=('cnt_bad', 'sum'),
                                         all_cnt=('cnt', 'sum')).reset_index()
        tmp['all_rate_bad'] = np.round(tmp['all_cnt_bad'] / tmp['all_cnt'], 3)
        gp = gp.merge(tmp[group_cols + ['all_rate_bad']], on=group_cols, how='left')
        gp['lift_bad'] = np.round(gp['rate_bad'] / gp['all_rate_bad'], 1)
    cols = group_cols + ['lbl', 'lbl_index', 'lbl_left', 'cnt', 'cnt_bad', 'rate_bad', 'cnt_over_total', 'accum_cnt',
                         'accum_cnt_bad', 'accum_rate_bad',
                         'accum_cnt_over_total', 'lift_bad', 'total', 'all_rate_bad']
    return gp[cols]


def twoscores_binary_liftvar(df, f1_score, f2_score, y, f1_grid=[], f2_grid=[], n_bin=10) -> pd.DataFrame:
    """
    评估2个模型分的联合的lift 变化
    :param f1 f2 ::: feature_name
    :param target: 0,1 变量
    :return:'cnt', 'rate_bad', 'accum_cnt_rate', 'accum_rate_bad', 'accum_lift_bad'
    """
    cols = [y, f1_score, f2_score]
    df = df[cols]
    df = statsutil.get_bin(df, f1_score, feature_grid=f1_grid, cut_type=1, n_bin=n_bin)
    if '%s_lbl' % f1_score in df.columns:
        df.drop(['%s_lbl' % f1_score,'%s_lbl_index' % f1_score,'%s_lbl_left' % f1_score],axis=1,inplace=True)
    df.rename(columns={'lbl': '%s_lbl' % f1_score, 'lbl_index': '%s_lbl_index' % f1_score}, inplace=True)
    df = statsutil.get_bin(df, f2_score, feature_grid=f2_grid, cut_type=1, n_bin=n_bin)
    if '%s_lbl' % f2_score in df.columns:
        df.drop(['%s_lbl' % f2_score,'%s_lbl_index' % f2_score,'%s_lbl_left' % f2_score],axis=1,inplace=True)
    df.rename(columns={'lbl': '%s_lbl' % f2_score, 'lbl_index': '%s_lbl_index' % f2_score}, inplace=True)
    ix = '%s_lbl' % f1_score
    column = '%s_lbl' % f2_score

    gp = pd.pivot_table(df, values=[y], index=ix,
                        columns=column, fill_value=0,
                        aggfunc=('count', 'sum', 'mean'), observed=True,)
    gp = gp[y]
    # 所有的数量
    all_cnt = df.shape[0]
    all_bad_cnt = df[y].sum()
    all_bad_rate = np.round(all_bad_cnt / all_cnt, 3)
    # 分组数量计算
    t_cnt = gp['count']
    # ix,column,cnt
    t_cnt = t_cnt.stack().reset_index().rename(columns={0: 'cnt'})
    t_cnt['cnt_over_total']=np.round(t_cnt.cnt/all_cnt,3)
    # 每组坏比例
    t_bad_rate = gp['mean']
    # ix,column,rate_bad
    t_bad_rate = t_bad_rate.stack().reset_index().rename(columns={0: 'rate_bad'})
    t_bad_rate['rate_bad']=np.round(t_bad_rate['rate_bad'],3)
    gp_out = t_cnt.merge(t_bad_rate, on=[ix, column], how='outer')
    # 累计总量
    t_accum_cnt = gp['count'].cumsum(axis=1).cumsum(axis=0)
    # ix,column,accum_cnt
    t_accum_cnt = t_accum_cnt.stack().reset_index().rename(columns={0: 'accum_cnt'})
    # 累计总量占比
    t_accum_cnt['accum_cnt_over_total']=np.round(t_accum_cnt.accum_cnt/all_cnt,3)
    gp_out = gp_out.merge(t_accum_cnt, on=[ix, column], how='outer')
    # 累计所有坏的比例
    t_accum_bad = gp['sum'].cumsum(axis=1).cumsum(axis=0)
    # 累计坏比例
    t_accum_bad = t_accum_bad.stack().reset_index().rename(columns={0: 'accum_cnt_bad'})
    gp_out = gp_out.merge(t_accum_bad, on=[ix, column], how='outer')
    gp_out['accum_rate_bad']=np.round(gp_out.accum_cnt_bad/gp_out.accum_cnt,3)
    gp_out['lift_bad']=np.round(gp_out.rate_bad/all_bad_rate,1)
    # if not show_flat:
    # 堆叠模式
    # gp_out = gp_out.set_index([ix, column])
    # gp_out = gp_out.stack().reset_index().rename(columns={'level_2': 'key', 0: 'value'})
    # gp_out = pd.pivot_table(gp_out, values=['value'], index=ix,
    #                         columns=[column, 'key'],
    #                         aggfunc=np.mean, sort=False)
    # gp_out = gp_out['value']
    gp_out = pd.pivot_table(gp_out,
                            values=['cnt', 'rate_bad','cnt_over_total','accum_cnt','accum_cnt_bad',
                                    'accum_rate_bad','accum_cnt_over_total','lift_bad'],
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
    :return:'cnt', 'rate_bad',
    """
    cols = [target] + model_scores
    df = df[cols]
    # 每个模型分分桶
    for score in model_scores:
        df = statsutil.get_bin(df, score, cut_type=1, n_bin=n_bin)
        df.rename(columns={'lbl': '%s_lbl' % score, 'lbl_index': '%s_lbl_index' % score}, inplace=True)
    ixes = ['%s_lbl' % f for f in model_scores]
    # 空值不计入
    gp = pd.pivot_table(df, values=[target], index=ixes,
                        aggfunc=('count', 'sum', 'mean'), observed=True, )
    gp = gp[target]
    gp.rename(columns={'count': 'cnt', 'sum': 'cnt_bad', 'mean': 'rate_bad'}, inplace=True)
    gp['rate_bad']=np.round(gp['rate_bad'],3)
    # 所有的数量
    all_cnt = df.shape[0]
    all_bad_cnt = df[target].sum()
    all_bad_rate = np.round(all_bad_cnt / all_cnt, 3)
    gp['cnt_over_total'] = np.round(gp.cnt/all_cnt,3)
    gp['lift_bad'] = np.round(gp['rate_bad'] / all_bad_rate, 1)
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
    :return cnt,mean,auc,ks,gini
    """
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        return None, None, None, None, None
    if len(np.unique(y_true)) == 1:
        # only one class
        logging.info('only one class !!!')
        return None, None, None, None, None
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
    return len(y_true), np.round(np.mean(y_true), 3), auc, ks, gini


def evaluate_binary_classier_bygroup(df, y_true: str, y_pred: str, group_cols=[]) -> pd.DataFrame:
    # 分组计算 auc,ks,gini
    if group_cols is not None and len(group_cols) > 0:
        gp_auc = df[df[y_pred].notna()].groupby(group_cols).apply(
            lambda x: evaluate_binary_classier(x[y_true], x[y_pred], is_show=False))
        gp_auc = gp_auc.reset_index().rename(columns={0: 'value'})
        gp_auc.loc[:, ['cnt', 'rate_bad', 'auc', 'ks', 'gini']] = gp_auc['value'].apply(pd.Series,
                                                                                        index=['cnt', 'rate_bad', 'auc',
                                                                                               'ks', 'gini'])
        gp_auc.drop(['value'], axis=1, inplace=True)
    else:
        cnt, rate_bad, auc, ks, gini = evaluate_binary_classier(df[df[y_pred].notna()][y_true],
                                                                df[df[y_pred].notna()][y_pred])
        gp_auc = pd.DataFrame([[cnt, rate_bad, auc, ks, gini]], columns=['cnt', 'rate_bad', 'auc', 'ks', 'gini'],
                              index=['all'])
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


def evaluate_roi():
    """
    模型AUC 指标，如何表现为样本的影响上
    ROI:投资回报率 = 收益/成本
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
