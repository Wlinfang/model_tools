import pandas as pd
import numpy as np
import pickle
import logging
from collections import Counter
from utils.toolutil import del_none

import category_encoders as ce 

logger = logging.getLogger(__file__)


def get_feature_grid(values, qcut_type=1, n_bin=10, default_values=[]) -> list:
    """
    返回分组 默认值和空值划分为一组
    values:list , np.array
    qcut_type: 1 : 等频; 2:等宽 3、
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
    values = values[~np.isin(default_values)]
    # 非字符型数据，直接返回
    if values.dtype.kind not in ['i', 'u', 'f', 'c']:
        return np.unique(values)
    n = np.nunique(values).size
    if n == 0:
        return None
    # 对values 进行排序
    vs_sort = np.sort(values)
    if n <= n_bin:
        f = [-np.Inf] + vs_sort.tolist()
    else:
        if qcut_type == 1:
            # 等频
            bin_index = [i / n_bin for i in range(0, n_bin + 1)]
            f = np.sort(np.unique(np.quantile(vs_sort, bin_index)))
        elif qcut_type == 2:
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


def get_bin(df, feature_name, qcut_type=1, n_bin=10, feature_grid=[], default_values=[]):
    """
    分组；默认值+缺失值为分为1组
    feature_grid:如果未指定，则根据 qcut_type + n_bin 分组
    返回：字段 lbl  lbl_index lbl_left
    """
    if (df is None) or (df.shape[0] == 0):
        raise ValueError('data is None ')
    df[feature_name] = pd.to_numeric(df[feature_name], errors='ignore')
    # 分为空和非空
    t1 = df[(df[feature_name].notna()) & (~df[feature_name].isin(default_values))].copy()
    t2 = df[~df.index.isin(t1.index)].copy()

    if not pd.api.types.is_numeric_dtype(df[feature_name]):
        # 非数字型
        t1['lbl'] = t1[feature_name]
    if not feature_grid:
        # 如果未指定
        feature_grid = get_feature_grid(df[feature_name], qcut_type, n_bin
                                        , default_values)
    if pd.api.types.is_numeric_dtype(df[feature_name]):
        # 数字型
        t1['lbl'] = pd.cut(t1[feature_name], feature_grid, include_lowest=True, right=False, precision=4)

    t1['lbl'] = t1['lbl'].astype('category')
    # 则为缺失值
    t1 = pd.concat([t1, t2], axis=0)
    # 填充空值
    t1['lbl'] = t1['lbl'].cat.add_categories('miss data')
    cates = t1['lbl'].cat.categories.tolist()
    cates = [i.left if isinstance(i, pd.Interval) else i for i in cates]
    t1['lbl_index'] = t1['lbl'].cat.codes
    t1['lbl_left'] = t1.lbl_index.apply(lambda x: cates[x])
    return t1


def split_train_val(df, split_type=1, split_ratio=0.8, sort_col=None):
    """
    将数据集进行切分，默认为随机切分
    split_type：1：随机切分；2：按照时间切分
    split_ratio: 切分比例； 取值范围：(0,1)
    sort_col：按照时间切分，切分比例为 split_ratio
    :return: train,val
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
    feature_names: list : 特征名称
    """
    res = []
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


def univar(df: pd.DataFrame, x, y, classes=[], feature_grid=[], qcut_type=1, n_bin=10) -> pd.DataFrame:
    """
    对 x 进行分组，对y求每组对应的均值
    x: column of x
    classes:维度
    feature_grid:如果未指定，则 根据 qcut_type + n_bin 分组
    """
    # 对x 进行分组； 'lbl', 'lbl_index', 'lbl_left'
    df = get_bin(df, x, feature_grid=feature_grid, qcut_type=qcut_type, n_bin=n_bin)
    # 对应的y mean 计算
    if classes:
        classes.extend(['lbl', 'lbl_index', 'lbl_left'])
    else:
        classes = ['lbl', 'lbl_index', 'lbl_left']
    gp = df.groupby(classes).agg({y: ['count', 'mean']})
    gp.columns = ['count', 'mean']
    gp.reset_index(inplace=True)
    tmp = df.groupby(classes).size().reset_index().rename(columns={0: "total"})
    gp = tmp.merge(gp, on=classes, how='left')
    gp['miss_rate'] = np.round(gp['count'] / gp['total'], 3)
    return gp


def sample_label(df, label, classes=[]) -> pd.DataFrame:
    """
    分组计算  正、负样本的 样本量，mean
    label:[0,1] 二分类
    classes:维度集合
    lift : 分组后的 正样本率 / 整体的正样本率
    """
    # 总计
    gp = pd.DataFrame([[df.shape[0], df[label].sum(), df[label].mean()]],
                      columns=['样本量', '正样本量', '正样本率'], index=['总计'])
    if classes:
        tmp = df.groupby(classes).agg(样本量=(label, 'count'), 正样本量=(label, 'sum'), 正样本率=(label, 'mean')).reset_index()
        gp = pd.concat([tmp, gp], axis=0)
        gp['lift'] = np.round(gp['正样本率'] / gp.loc['总计', '正样本率'], 2)
    gp['正样本率'] = np.round(gp['正样本率'], 2)
    return gp


def model_evaluate_classier(y_real, y_pred):
    """
    二分类模型评估指标
    @param y_real : list，真实值
    @param y_pred : list, 预测值
    混淆矩阵：
            真实值
            1	0
    预测值 1 TP  FP
          0 FN  TN
    准确率 Accuracy = (TP+TN) / (TP+FP+FN+TN)
    精确率(Precision) = TP / (TP+FP) --- 预测为正样本 分母为预测正样本
    召回率(Recall) = TP / (TP+FN) -- 实际为正样本 分母为真实正样本
    F-Meauter = (a^2 + 1) * 精确率 * 召回率 / [a^2 * (精确率 + 召回率)]
    a^2 如何定义
    F1分数(F1-Score) = 2 *  精确率 * 召回率 / (精确率 + 召回率)
    P-R曲线 ： 平衡点即为 F1分数；y-axis = 精确率；x-axis= 召回率
    平衡点 ： 精确率 = 召回率
    真正率(TPR) = TP / (TP+FN)-- 以真实样本 分母为真实正样本
    假正率(FPR) = FP / (FP+TN)-- 以真实样本 分母为真实负样本
    Roc 曲线：y-axis=真正率 ; x-axis=假正率； 无视样本不均衡问题
    AUC = Roc 曲线面积
    """
    accuracy = accuracy_score(y_real, y_pred)
    # p=precision_score(y_real, y_pred)
    # f1=f1_score(y_real, y_pred)
    # 返回confusion matrix
    cm = confusion_matrix(y_real, y_pred)
    # 返回 精确率，召回率，F1
    cr = classification_report(y_real, y_pred)
    auc = roc_auc_score(y_real, y_pred)


def psi(data_base, data_test, feature_grid=[], n_bin=10):
    """
    支持数字型和非数字型；如果是非数字型，则按每一个取值分组
    计算稳定性--
    data_base: list, 1d - np.array, pd.Series；默认剔除空值 空字符串
    data_test:以df_base 为基准，进行分段
    feature_grid:如果未指定，则按照等频 n_bin 分组
    psi:
    > 0.25  分布差异明显
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
            feature_grid = get_feature_grid(data_base, qcut_type=1, n_bin=n_bin)
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


def eval_miss(df, feature_name, label):
    """
    不同 label 的情况下，feature_name 的缺失率
    如果随着时间的偏移，不一致，缺失的采取了其他的策略-> 其他的字段分布明显的差异
    """
    # 分组统计
    gp = df.groupby(label)[feature_name].count().reset_index()
    tmp = df.groupby(label).size().reset_index().rename(columns={0: 'total'})
    gp = gp.merge(tmp, on='label', how='right')
    gp['miss_rate'] = np.round(gp[feature_name] / gp['total'], 3)
    return gp


def woe(df,x,y,n_bins=10):
    """
    woe 计算
    x:list or str
    """
    if isinstance(x,str):
        x=[x]
    ce.WOEEncoder(df[x])


