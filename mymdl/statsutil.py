"""
主要用于统计分析
"""
import pandas as pd
import numpy as np
from typing import Union, Tuple, List
from ..utils import toolutil


#################### 分桶 ##########################################
def get_featuregrid_by_equal_width(values: List, n_bin=10):
    """
    等宽，传入的值必须为数字型
    """
    if (values is None) or (len(values) < 1) or (n_bin <= 0):
        raise ValueError('values is None')
    values = np.array(values)
    if values.dtype.kind not in ['i', 'u', 'f', 'c']:
        raise ValueError('values must be number')
    # 去除空值、空字符串
    values = toolutil.del_none(values)
    # 限制为 1d
    if values.ndim > 1:
        raise ValueError('ndim of values must be 1')
    n = np.unique(values).size
    if n <= n_bin:
        f = np.sort(np.unique(values))
    else:
        mi, mx = values.min(), values.max()
        bin_index = [mi + (mx - mi) * i / n_bin for i in range(0, n_bin + 1)]
        f = 1.0*np.sort(np.unique(bin_index))
    # 包括无穷大，样本集中数据可能有些最小值，最大值不全
    f[0] = -np.Inf
    f[-1] = np.inf
    return np.round(f, 3)


def get_featuregrid_by_equal_freq(values: List, n_bin=10) -> list:
    """
    等频
    """
    if (values is None) or (len(values) < 1) or (n_bin <= 0):
        raise ValueError('values is None')
    values = np.array(values)
    if values.dtype.kind not in ['i', 'u', 'f', 'c']:
        raise ValueError('values must be number')
    # 去除空值、空字符串
    values = toolutil.del_none(values)
    # 限制为 1d
    if values.ndim > 1:
        raise ValueError('ndim of values must be 1')
    n = np.unique(values).size
    if n <= n_bin:
        f = np.sort(np.unique(values))
    else:
        vs_sort = np.sort(values)
        bin_index = [i / n_bin for i in range(0, n_bin + 1)]
        f = 1.0*np.sort(np.unique(np.quantile(vs_sort, bin_index, method='lower')))
    # 包括无穷大，样本集中数据可能有些最小值，最大值不全
    f[0] = -np.Inf
    f[-1] = np.inf
    return np.round(f, 3)


def get_featuregrid_by_chi(df, x, y, n_bin=10, chi2_threold=1.7) -> list:
    """
    卡方分箱：具有最小卡方值的相邻区间合并在一起，适用于分类
    初始化n个区间，分别计算相邻区间的卡方值，找到最小卡方值的，合并区间，重复迭代
    :param: y 目标变量，类别变量
    """
    # 剔除缺失值的情况
    df[x] = pd.to_numeric(df[x], errors='coerce')
    df = df[df[x].notna()]
    # 计算每个类别的变量
    gp_y = df[y].value_counts().to_frame().reset_index().rename(columns={'count': 'cnt'})
    gp_y['y_rate'] = np.round(gp_y[y] / gp_y.cnt, 3)
    # 初始化分桶,利用等频分桶
    f = get_featuregrid_by_equal_freq(df[x], n_bin)
    # 最少2个箱体
    while len(f) > 3:
        # 对X进行分桶
        if 'lbl' in df.columns:
            df.drop(['lbl', 'lbl_left','lbl_index'], axis=1, inplace=True)
        df = get_bin(df, x, feature_grid=f)
        # 对分桶进行统计,区间数量统计
        gp_x = df.groupby('lbl_left').size().reset_index().rename(columns={0: 'lbl_cnt'})
        # 区间内，每个类别的数量统计
        gp = df.groupby(['lbl_left', y]).size().reset_index().rename(columns={0: 'lbl_y_cnt'})
        gp = gp.merge(gp_x, on='lbl_left', how='left')
        gp = gp.merge(gp_y, on=y, how='lbl_left')
        # 每个区间期望数量= y_rate * lbl_cnt
        gp['lbl_y_expt_cnt'] = np.round(gp.y_rate * gp.lbl_cnt, 1)
        # 每个区间卡方值= (期实际数量 - 期望数量)^2 / 期望数量
        gp['chi2'] = np.round((gp.lbl_y_cnt - gp['lbl_y_expt_cnt']) ^ 2 / gp['lbl_y_expt_cnt'], 3)
        gp = gp.groupby('lbl_left')['chi2'].sum().reset_index()
        # 计算相邻两区间的卡方=各个区间的卡方的和
        gp['last_chi2'] = gp['chi2'].shift(periods=1)
        gp['interval_chi2'] = gp['last_chi2'] + gp['chi2']
        # 查找最小的 chi2
        idx_min = gp['interval_chi2'].idxmin()
        value_min = gp['interval_chi2'].min()
        if value_min < chi2_threold:
            f = gp[gp.index != (idx_min - 1)].lbl_left.unique().tolist()
            continue
        else:
            break
    # 包括无穷大，样本集中数据可能有些最小值，最大值不全
    f[0] = -np.Inf
    f[-1] = np.inf
    return np.round(f, 3)


def get_feature_grid(df: pd.DataFrame, x: str, y=None, cut_type=1, n_bin=10) -> list:
    """
    计算分桶
    :param values: 要进行分桶的数据
    :param cut_type: 1 : 等频; 2:等宽 3、卡方
    """
    if (df is None) or (len(df) < 1) or (n_bin <= 0):
        raise ValueError('param is error')
        return None
    # 非数字型
    if not pd.api.types.is_numeric_dtype(df[x]):
        return df[df[x].notna()][x].unique().tolist()
    n = df[x].nunique(dropna=True)
    if n == 0:
        return None
    if cut_type == 1:
        return get_featuregrid_by_equal_freq(df[x], n_bin)
    elif cut_type == 2:
        return get_featuregrid_by_equal_width(df[x], n_bin)
    elif cut_type == 3:
        return get_featuregrid_by_chi(df, x, y, n_bin)
    else:
        return None


def get_bin(df: pd.DataFrame, feature_name: str, y=None, cut_type=1,
            n_bin=10, feature_grid=[], default_values=[]) -> pd.DataFrame:
    """
    分组；默认值+缺失值为分为1组
    :param cut_type 分组类型： 1 ：等频分组；2：等宽分组 3:卡方分组
    :param n_bin 分组个数，默认10个组；如果加入缺失值，则分为11组
    :param feature_grid 优先按照 feature_grid 分组；如果未指定，按照cut_type
    :param y:目标变量，如果是卡方分组，则必须存在
    :return df 含有字段 lbl  lbl_index lbl_left
    """
    if (df is None) or (df.shape[0] == 0):
        raise ValueError('data is None ')
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
        feature_grid = get_feature_grid(t1,feature_name,y=y, cut_type=cut_type, n_bin=n_bin)
        if len(feature_grid) == 0:
            raise ValueError('feature_grid is None ')
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


def woe(df: pd.DataFrame, x: str, y: str, feature_grid=[], n_bin=10):
    """
    woe = ln (坏人比例) / (好人比例)
    坏人比例 =  组内的坏人数/ 总坏人数
    好人比例 = 组内的好人数/ 总好人数
    缺失值的woe 可不满足单调性，因为缺失值尤其逻辑含义
    如果相邻分箱的woe值相同，则合并为1个分箱
    当一个分箱内只有bad 或者 good时，修正公式公式计算中，加入 eps
    如果训练集woe满足单调性；but 验证集或测试集上不满足，则分箱不合理或者这个特征不稳定，时效性差
    :param 等频分箱
    :param y 二值变量  定义 坏=1  好=0
    """
    # 好人比例 = rate_good_over_allgood
    # 坏人比例 = rate_bad_over_allbad
    df = get_bin(df, x, feature_grid=feature_grid, cut_type=1, n_bin=n_bin)
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


def iv(df: pd.DataFrame, x: str, y: str, feature_grid=[], n_bin=10):
    """
    对 x 进行分组，对于每个组内的iv_i 求和
    iv_i =  (坏人比例-好人比例) * woe_i
    iv = sum(iv_i)
    坏人比例 =  组内的坏人数/ 总坏人数
    好人比例 = 组内的好人数/ 总好人数
    :return   <0.02 (无用特征)  0.02~0.1（弱特征） 0.1~0.3（中价值）0.3~0.5（高价值）>0.5(数据不真实)
    """
    gp = woe(df, x, y, feature_grid, n_bin)
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



def ctribtion_rate_one(df_base, df_test, x, sub_group_cols: list, group_cols=[]) -> pd.DataFrame:
    """
    贡献率公式1，加法模式 case: uv,pv
    贡献度 = 维度值绝对DIFF / 大盘绝对DIFF。
    Y = ∑X_i【子项】   △Y= (Y_test - Y_base ) / Y_base
    某子项的贡献率 = △Y_i/△Y = (X_test_i - X_base_i) / Y_base / △Y = (X_test_i - X_base_i) / (Y_test - Y_base )
    :param group_cols 整体 可以为空；sub_group_cols 子项
    """
    df_base[x] = pd.to_numeric(df_base[x], errors='coerce')
    df_test[x] = pd.to_numeric(df_test[x], errors='coerce')

    if sub_group_cols is None or len(sub_group_cols) == 0:
        raise ValueError('sub_group_cols must not be empty')
    gp_base = df_base.groupby(sub_group_cols).agg(sub_sum=(x, 'sum')).reset_index()
    gp_test = df_test.groupby(sub_group_cols).agg(sub_sum=(x, 'sum')).reset_index()
    # 总体，
    if group_cols is None or len(group_cols) == 0:
        gp_base['all_sum'] = df_base[x].sum()
        gp_test['all_sum'] = df_test[x].sum()
    else:
        if len(set(group_cols) - set(sub_group_cols)) != 0:
            raise ValueError('group_cols is sub set of sub_group_cols')
        t = df_base.groupby(group_cols).agg(all_sum=(x, 'sum')).reset_index()
        gp_base = gp_base.merge(t, on=group_cols, how='left')
        # test
        t = df_test.groupby(group_cols).agg(all_sum=(x, 'sum')).reset_index()
        gp_test = gp_test.merge(t, on=group_cols, how='left')
    gp = gp_base.merge(gp_test, on=sub_group_cols, how='outer', suffiexs=('_base', '_test'))
    gp['贡献率'] = np.round((gp.sub_sum_test - gp.sub_sum_base) / (gp.all_sum_test - gp.all_sum_base), 3)
    return gp


def ctribtion_rate_two(df_base, df_test, x, y, x_method='sum', y_method='count', group_cols=[]) -> pd.DataFrame:
    """
    适用于比例类指标：考虑指标变化 + 子项样本占比变化
    case: R = 放款率 = 放款数/ 授信数 ; 放款数=安卓放款数+IOS 放款数  授信数=安卓授信数+IOS 授信数
    样本占比  P = 安卓授信数 / 授信数   S = 安卓放款数/放款数
    指标变化测算：A_i=(R_test_{ios} - R_base_{ios}) * P_base_{ios} : 假设 样本占比不变的情况下，指标变化率的贡献
    占比变化测算：B_i=(P_test_{ios}-P_base_{ios}) * (R_test_{ios} - R_base)
    总体贡献度 C_i = A_i + B_i
    diff R = R_test - R_base =  ∑C_i
    证明过程：P_test_{ios} * R_test_{ios} +P_test_{and} * R_test_{and}  = R_test
    """
    df_base[x] = pd.to_numeric(df_base[x], errors='coerce')
    df_base[y] = pd.to_numeric(df_base[y], errors='coerce')

    df_test[x] = pd.to_numeric(df_test[x], errors='coerce')
    df_test[y] = pd.to_numeric(df_test[y], errors='coerce')

    # 总的变化
    if x_method == 'sum':
        x_base_total = df_base[x].sum()
        x_test_total = df_test[x].sum()
    elif x_method == 'count':
        x_base_total = df_base[x].shape[0]
        x_test_total = df_test[x].shape[0]
    else:
        raise ValueError('x_method not support,only sum or count')

    if y_method == 'sum':
        y_base_total = df_base[y].sum()
        y_test_total = df_test[y].sum()
    elif y_method == 'count':
        y_base_total = df_base[y].shape[0]
        y_test_total = df_test[y].shape[0]
    else:
        raise ValueError('y_method not support,only sum or count')

    r = np.round(x_test_total / y_test_total - x_base_total / y_base_total, 3)

    # 维度变化
    gp_base = df_base.groupby(group_cols).agg(x_sub=(x, x_method), y_sub=(y, y_method)).reset_index()
    gp_base['r'] = np.round(gp_base.x_sub / gp_base.y_sub, 3)
    # 样本占比
    gp_base['y_over_ally'] = np.round(gp_base.y_sub / y_base_total, 3)
    gp_test = df_test.groupby(group_cols).agg(x_sub=(x, x_method), y_sub=(y, y_method)).reset_index()
    gp_test['r'] = np.round(gp_test.x_sub / gp_test.y_sub, 3)
    gp_test['y_over_ally'] = np.round(gp_test.y_sub / y_test_total, 3)

    # 合并
    gp = gp_base.merge(gp_test, on=group_cols, suffixes=('_base', '_test'), how='outer')
    gp = gp.fillna(0)
    gp['贡献率_指标变化'] = np.round((gp.r_test - gp.r_base) * gp.y_over_ally_base / r, 3)
    gp['贡献率_占比变化'] = np.round(
        (gp.y_over_ally_test - gp.y_over_ally_base) * (gp.r_test - x_base_total / y_base_total) / r, 3)
    gp['贡献率'] = gp['贡献率_指标变化'] + gp['贡献率_占比变化']
    cols = group_cols + ['x_sub_base', 'y_sub_base', 'r_base', 'y_over_ally_base', 'x_sub_test', 'y_sub_test', 'r_test',
                         'y_over_ally_test', '贡献率_指标变化', '贡献率_占比变化', '贡献率']
    gp = gp[cols]
    gp = gp.rename(
        columns={'x_sub_base': 'x_base', 'y_sub_base': 'y_base', 'y_over_ally_base': '占比_base', 'r_base': '指标_base',
                 'x_sub_test': 'x_test', 'y_sub_test': 'y_test', 'y_over_ally_test': '占比_test',
                 'r_test': '指标_test'})
    return gp


def ctribtion_rate_lmdi():
    """
    适用于乘法链路：case: Y【新增成交用户数】=N[新增用户量] × A[用户激活率] × S[激活用户3日留存率]  × P[留存用户购买率]
    ln y_test / y_base = ln N_test/N_base + ln A_test/A_base + ln S_test/S_base + ln P_test/P_base
    1 = (ln N_test/N_base + ln A_test/A_base + ln S_test/S_base + ln P_test/P_base) / ln y_test / y_base
    """
    pass
