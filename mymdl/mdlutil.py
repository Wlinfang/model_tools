import pandas as pd
import numpy as np
import seaborn as sn

from mylog import logutil
logger=logutil.MyLogging().get_logger(__file__)


class MyBin:
    def __init__(self,n_bin=10,is_same_width=False):
        '''
        计算分组，剔除空值或默认值后剩余的数据进行分组；空值和默认值为单独一个组
        is_same_width:True: 等宽 否则 等频
        如果是非数字型，则所有的取值作为分组
        '''
        self._n_bin=n_bin
        self._is_same_width=is_same_width

    def _get_feature_grid(self,df,feature_name,default_value=None):
        '''
        返回分组
        :param df:
        :return:  list
        '''
        if (df is None) or (df.shape[0] == 0) or (self._n_bin <= 0):
            logger.info(' df is empty ')
            return None
        df.loc[df[feature_name] == '', feature_name] = None
        df[feature_name] = pd.to_numeric(df[feature_name], errors='ignore')
        if pd.api.types.is_numeric_dtype(df[feature_name]) == False:
            # 字符型的，直接返回
            return df[(df[feature_name].notna()) & (df[feature_name] != default_value)][feature_name].unique()
        tmp = df[(df[feature_name].notna()) & (df[feature_name] != default_value)]
        n = tmp[feature_name].nunique()
        if n == 0:
            return None
        if n <= self._n_bin:
            f = [-np.inf] + sorted(tmp[feature_name].unique().tolist())
        else:
            if self._is_same_width:
                mi, mx = tmp[feature_name].min(), tmp[feature_name].max()
                bin_index = [mi + (mx - mi) * i / self._n_bin for i in range(0, self._n_bin + 1)]
                f = sorted(set(bin_index))
            else:
                bin_index = [i / self._n_bin for i in range(0, self._n_bin + 1)]
                f = sorted(set(tmp[feature_name].quantile(bin_index)))
            # 包括无穷大，样本集中数据可能有些最小值，最大值不全
            f[0] = -np.Inf
            f[-1] = np.inf
        return np.round(f, 3)

    def _get_bin(self,df, feature_name,feature_grid=[],default_value=None):
        '''
        对 df中的每笔明细划分区间
        默认值+缺失值为分为1组
        :param df:
        :param feature_name:
        :param feature_grid:如果未指定，则 _get_feature_grid 获取
        :return: df
        '''
        if (df is None) or (df.shape[0] == 0):
            return None
        df.loc[df[feature_name] == '', feature_name] = None
        df[feature_name] = pd.to_numeric(df[feature_name], errors='ignore')
        if pd.api.types.is_numeric_dtype(df[feature_name]) == False:
            # 非数字型
            df['qujian'] = df[feature_name]
            df.loc[df[feature_name].isna(), 'qujian'] = 'miss data'
            df.loc[df[feature_name] == default_value, 'qujian'] = 'miss data'
            return df
        if len(feature_grid) == 0:
            feature_grid = sorted(self._get_feature_grid(df,feature_name,default_value))
            if feature_grid is None:
                logger.info(' no valid data ')
                return None
        # 分为空和非空
        t1 = df[(df[feature_name].notna()) & (df[feature_name] != default_value)]
        # t2 是一个copy 数据设置有warnning，进行copy可以消除，断开同源数据的联系
        t2 = df[(df[feature_name].isna()) | (df[feature_name] == default_value)]
        t1['qujian'] = pd.cut(t1[feature_name], feature_grid, include_lowest=False, precision=4)
        t1['qujian'] = t1['qujian'].astype('category')
        # t1['bucket']=t1['qujian'].cat.codes
        # 则为缺失值
        if t2.shape[0] > 0:
            t2['qujian'] = 'miss data'
            return pd.concat([t1, t2])
        return t1


def cal_describe(df,feature_name_list):
    '''
    feature_name_list: list : 特征名称
    特征的描述性分析，支持数字型+字符型
    top:众数
    freq:众数次数
    freq_rate:众数占比
    count:有值的样本数
    all_count:所有样本数
    skew 偏度：当偏度<0时，概率分布图左偏。
         当偏度=0时，表示数据相对均匀的分布在平均值两侧，不一定是绝对的对称分布。
         当偏度>0时，概率分布图右偏
    kurt 峰度:对比正太，正太的峰度=3，计算的时候-3，正太峰度=0
    return dataframe
    '''
    res = []
    # 提取数字型数据
    num_list = df[feature_name_list].select_dtypes(include=np.number).columns.tolist()
    str_list = df[feature_name_list].select_dtypes(include='object').columns.tolist()
    if len(num_list) > 0:
        gp_num = df[num_list].describe().T
        gp_num['all_count'] = df.shape[0]
        gp_num['miss_rate'] = np.round(1 - gp_num['count'] / df.shape[0], 3)
        gp_num = gp_num[['all_count', 'count', 'miss_rate', 'mean', 'std', 'min', '50%', 'max']]
        gp_num[['miss_rate', 'mean', 'std', 'min', '50%', 'max']] = np.round(
            gp_num[['miss_rate', 'mean', 'std', 'min', '50%', 'max']], 3)
        gp_num['count'] = gp_num['count'].astype(int)
        # 取数
        gp_num = gp_num.join(df[num_list].nunique(axis=0).to_frame().rename(columns={0: 'unique'}))
        gp_num = gp_num.reset_index().rename(columns={'index': 'feature_name'})
        a = df[num_list].mode(axis=0)
        if a.shape[0] == 1:
            # # a 中过滤有多个众数的特征,多个众数，则不计算
            # if a.shape[0] > 1:
            # 	d=a[1:2].dropna(axis=1,how='all').columns.tolist()
            # 	a=a[set(feature_name_list)-set(d)]
            # 	a=a.T.dropna(axis=1,how='any').reset_index()
            a = a.T.reset_index()
            a.columns = ['feature_name', 'top']
            gp_num = gp_num.merge(a, how='left')
            # 众数占有值的比例
            gp_num['freq'] = gp_num.apply(lambda x: df[df[x.feature_name] == x['top']].shape[0], axis=1)
            gp_num['freq_rate'] = np.round(gp_num['freq'] / gp_num['count'], 3)
            # 计算偏度，<0 左偏 =0 正太分布；>0 右偏
            a = df[num_list].skew(axis=0, skipna=True).reset_index().rename(
                columns={'index': 'feature_name', 0: 'skew'})
            gp_num = gp_num.merge(a, on='feature_name', how='left')
            # 计算峰度
            a = df[num_list].kurt(axis=0, skipna=True).reset_index().rename(
                columns={'index': 'feature_name', 0: 'kurt'})
            gp_num = gp_num.merge(a, on='feature_name', how='left')
            res.append(gp_num)
    if len(str_list) > 0:
        gp_str = df[str_list].describe().T
        gp_str['all_count'] = df.shape[0]
        gp_str['count'] = gp_str['count'].astype(int)
        gp_str['freq'] = gp_str['freq'].astype(int)
        gp_str['miss_rate'] = np.round(1 - gp_str['count'] / df.shape[0], 3)
        gp_str = gp_str[['all_count', 'count', 'miss_rate', 'unique', 'top', 'freq']]
        gp_str['freq_rate'] = np.round(gp_str['freq'] / gp_str['count'], 3)
        gp_str = gp_str.reset_index().rename(columns={'index': 'feature_name'})
        res.append(gp_str)
    c_dict={'all_count':'样本量','count':'有值样本量','miss_rate':'缺失率','50%':'中位数','top':'众数','freq':'众数样本量','freq_rate':'众数占比'}
    gp= pd.concat(res)
    gp.rename(columns=c_dict,inplace=True)
    return gp


class EvalFeature:

    def __init__(self,feature_name,label=None):
        self.feature_name=feature_name
        self.label=label


    def _simple(self,df,sub_classes,classes=[]):
        '''
        特征单个分布；样本比例
        :param df:
        :param classes:
        :return:
        '''



def eval_miss(df, feature_name, label):
    '''
    评估缺失率的影响
    '''
    cnt=df.shape[0]
    cnt_miss=df[df[feature_name].isna()].shape[0]
    rate_all=np.round(df[label].mean(),3)
    cnt_good=cnt-cnt_miss
    rate_good=df[df[feature_name].notna()][label].mean()
    gp=pd.DataFrame([[cnt,cnt_miss,cnt_miss/cnt,rate_all,cnt_good,rate_good]],columns=['样本数','缺失样本数','缺失率','坏样本率','未缺失样本数','未缺失坏样本率'])
    gp['缺失率']=np.round(gp['缺失率'],3)
    gp['坏样本率'] = np.round(gp['坏样本率'], 3)
    gp['未缺失坏样本率'] = np.round(gp['未缺失坏样本率'], 3)
    return gp


def cal_sample_rate(df,sub_classes,classes=[]):
    '''
    计算样本比例;如果classes 为空，则按照全量样本量计算
    '''
    gp_sub = df.groupby(sub_classes).size().reset_index().rename(columns={0: 'cnt_sub'})
    if len(classes) > 0:
        if len(set(classes)-set(sub_classes)) > 0:
            # sub_classes 是 classes的子维度
            logger.error(" sub_classes or classes error ")
            return None
        gp = df.groupby(classes).size().reset_index().rename(columns={0: 'cnt'})
        gp_sub=gp_sub.merge(gp,on=classes,how='left')
    else:
        gp_sub['cnt']=df.shape[0]
    gp_sub['样本比例']=np.round(gp_sub.cnt_sub/gp_sub.cnt,2)
    return gp_sub

def cal_y_sample_rate(df,label,sub_classes,classes=[]):
    '''
    带有y值的样本比例计算
    :param df:
    :param label:
    :param sub_classes:
    :param classes:
    :return:
    '''
    df[label]=pd.to_numeric(df[label],errors='coerce')
    gp_sub=df.groupby(sub_classes).agg(样本量=(label,'count'),坏样本量=(label,'sum'),坏样本率=(label,'mean')).reset_index()
    if len(classes) > 0:
        if len(set(classes)-set(sub_classes)) > 0:
            # sub_classes 是 classes的子维度
            logger.error(" sub_classes or classes error ")
            return None
        gp = df.groupby(classes).agg(总样本量=(label,'count'),总坏样本量=(label,'sum'),总坏样本率=(label,'mean')).reset_index()
        gp_sub=gp_sub.merge(gp,on=classes,how='left')
    else:
        gp_sub['总样本量']=df.shape[0]
        gp_sub['总坏样本量']=df[label].sum()
        gp_sub['总坏样本率']=df[label].mean()

    gp_sub['样本占比']=np.round(gp_sub['样本量']/gp_sub['总样本量'],2)
    gp_sub['坏样本占比'] = np.round(gp_sub['坏样本量'] / gp_sub['总坏样本量'], 2)
    gp_sub['lift']=np.round(gp_sub['坏样本占比']/gp_sub['样本占比'],3)

    return gp_sub

def plot_y_x_bin(df_base,df_test,feature_name,label,feature_grid=[], n_bin=10,is_same_width=False,default_value=None,is_show=True):
    '''
    用于评估 base 和 test的特征和标签的表现分布图
    Parameters
    ----------
    df_base:dataframe
    df_test:dataframe
    feature_name:特征或模型
    label：标签
    feature_grid：list：用于对 feature_name 进行分组的；如果不为空，则使用 feature_grid；如果为空，则根据后续定义的 n_bin 、is_same_width 进行确定
    n_bin：分桶的个数
    is_same_width:True:等宽划分；False : 等频划分； 默认等频
    default_value: feature_name 默认值；用于分组； 默认值和缺失值统一为1组
    is_show:plot 是否显示
    Returns
    -------
    '''
    if (df_base is None) or (df_base.shape[0] == 0) or (df_test is None) or (df_test.shape[0]==0):
        logger.error(' df_base is empty or df_test is empty ',ValueError)
        return None
    if len(feature_grid) == 0:
        # base 数据计算分桶
        feature_grid = cal_feature_grid(df_base,feature_name,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
    df_base['train_or_test']='base'
    df_test['train_or_test']='test'
    df=pd.concat([df_base,df_test])
    df=cal_bin(df,feature_name,feature_grid=feature_grid)
    # 计算分桶的y的均值
    psi=cal_psi(df_base,df_test,feature_name,feature_grid=feature_grid)
    gp=df.groupby(['qujian','train_or_test']).agg(cnt=(label,'count'),cnt_bad=(label,'sum'),rate_bad=(label,'mean')).reset_index()
    gp_all = df.groupby(['train_or_test']).agg(cnt_all=(label, 'count'), cnt_bad_all=(label, 'sum'),rate_bad_all=(label, 'mean')).reset_index()
    gp=gp.merge(gp_all,on='train_or_test',how='right')
    gp['样本比例']=np.round(gp['cnt']/gp.cnt_all,3)
    if is_show:
        fig = plt.figure(figsize=[12, 6])  # 设定图片大小
        ax1 = fig.add_subplot(111)  # 添加第一副图
        gp['qujian']=gp['qujian'].astype(str)
        sn.lineplot(data=gp, x='qujian', y='rate_bad', hue='train_or_test', ax=ax1, sort=False,markers='o')
        ax1.legend(loc='upper left', labels=['坏样本率'])
        ax1.set_ylabel('坏样本率')
        plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')

        ax2 = ax1.twinx()
        sn.barplot(x='qujian', y='样本比例', data=gp, ax=ax2, hue='train_or_test', alpha=0.6)
        ax2.set_ylabel('样本比例')

        # 计算二者的卡方，pvalue;t检验的pvalue
        chi, pvalue, _, _ = stats.chi2_contingency(
            pd.pivot_table(index='train_or_test', columns='qujian', values=['cnt'], aggfunc='sum', data=gp).fillna(0))
        # 均值检验
        if pd.api.types.is_numeric_dtype(df[feature_name]):
            a = np.array(df_base[df_base[feature_name].notna()][feature_name])
            b = np.array(df_test[df_test[feature_name].notna()][feature_name])
            statistic, ttest_pvalue = stats.ttest_ind(a, b, equal_var=False)
            plt.title('{}_psi:{}_chi:{}_{}__ttest:{}'.format(feature_name, psi, np.round(chi, 2), np.round(pvalue, 2),
                                                             np.round(ttest_pvalue, 2)))
        else:
            plt.title('{}_psi:{}_chi:{}_{}'.format(feature_name, psi, np.round(chi, 2), np.round(pvalue, 2)))
    return gp


def plot_x_bin(df_base,df_test,feature_name,feature_grid=[], n_bin=10,is_same_width=False,default_value=None,is_show=True):
    '''
    用于评估 base 和 test的特征和标签的表现分布图
    Parameters
    ----------
    df_base:dataframe
    df_test:dataframe
    feature_name:特征或模型
    feature_grid：list：用于对 feature_name 进行分组的；如果不为空，则使用 feature_grid；如果为空，则根据后续定义的 n_bin 、is_same_width 进行确定
    n_bin：分桶的个数
    is_same_width:True:等宽划分；False : 等频划分； 默认等频
    default_value: feature_name 默认值；用于分组； 默认值和缺失值统一为1组
    is_show:plot 是否显示
    Returns
    -------
    '''
    if (df_base is None) or (df_base.shape[0] == 0) or (df_test is None) or (df_test.shape[0]==0):
        logger.error(' df_base is empty or df_test is empty ',ValueError)
        return None
    if len(feature_grid) == 0:
        # base 数据计算分桶
        feature_grid = cal_feature_grid(df_base,feature_name,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
    df_base['train_or_test']='base'
    df_test['train_or_test']='test'
    df=pd.concat([df_base,df_test])
    df=cal_bin(df,feature_name,feature_grid=feature_grid)
    # 计算分桶的y的均值
    psi=cal_psi(df_base,df_test,feature_name,feature_grid=feature_grid)
    gp=df.groupby(['qujian','train_or_test']).size().reset_index().rename(columns={0:'cnt'})
    gp_all = df.groupby(['train_or_test']).size().reset_index().rename(columns={0:'cnt_all'})
    gp=gp.merge(gp_all,on='train_or_test',how='right')
    gp['样本比例']=np.round(gp['cnt']/gp.cnt_all,3)
    if is_show:
        fig = plt.figure(figsize=[12, 6])  # 设定图片大小
        ax1 = fig.add_subplot(111)  # 添加第一副图
        gp['qujian']=gp['qujian'].astype(str)
        sn.lineplot(data=gp, x='qujian', y='样本比例', hue='train_or_test', ax=ax1, sort=False,markers='o')
        ax1.legend(loc='upper left', labels=['样本比例'])
        ax1.set_ylabel('样本比例')
        plt.setp(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
        # 计算二者的卡方，pvalue;t检验的pvalue
        chi, pvalue, _, _ = stats.chi2_contingency(
            pd.pivot_table(index='train_or_test', columns='qujian', values=['cnt'], aggfunc='mean', data=gp).fillna(0))
        # 均值检验
        if pd.api.types.is_numeric_dtype(df[feature_name]):
            a = np.array(df_base[df_base[feature_name].notna()][feature_name])
            b = np.array(df_test[df_test[feature_name].notna()][feature_name])
            statistic, ttest_pvalue = stats.ttest_ind(a, b, equal_var=False)
            plt.title('{}_psi:{}_chi:{}_{}__ttest:{}'.format(feature_name, psi, np.round(chi, 2), np.round(pvalue, 2),
                                                             np.round(ttest_pvalue, 2)))
        else:
            plt.title('{}_psi:{}_chi:{}_{}'.format(feature_name,psi,np.round(chi,2),np.round(pvalue,2)))
    return gp

def plot_y_x_bin_bytime(df_base,df_test,feature_name,label,by_time='apply_time',feature_grid=[], n_bin=10,is_same_width=False,default_value=None,is_show=True):
    '''
    按照日期排序，查看每个区间的分布
    Parameters
    ----------
    df_base
    df_test
    feature_name
    label
    by_time
    feature_grid
    n_bin
    is_same_width
    default_value
    is_show

    Returns
    -------
    '''
    if (df_base is None) or (df_base.shape[0] == 0) or (df_test is None) or (df_test.shape[0]==0):
        logger.error(' df_base is empty or df_test is empty ',ValueError)
        return None
    if len(feature_grid) == 0:
        # base 数据计算分桶
        feature_grid = cal_feature_grid(df_base,feature_name,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
    df_base['train_or_test']='base'
    df_test['train_or_test']='test'
    df=pd.concat([df_base,df_test])
    df=cal_bin(df,feature_name,feature_grid=feature_grid)
    # 计算分桶的y的均值
    psi=cal_psi(df_base,df_test,feature_name,feature_grid=feature_grid)
    gp=df.groupby(['qujian','train_or_test',by_time]).agg(cnt=(label,'count'),cnt_bad=(label,'sum'),rate_bad=(label,'mean')).reset_index()
    gp_all = df.groupby(['train_or_test',by_time]).agg(cnt_all=(label, 'count'), cnt_bad_all=(label, 'sum'),rate_bad_all=(label, 'mean')).reset_index()
    gp=gp.merge(gp_all,on=['train_or_test',by_time],how='right')
    gp['样本比例']=np.round(gp['cnt']/gp.cnt_all,3)
    if is_show:
        # gp['qujian']=gp['qujian'].astype(str)
        for qj in gp['qujian'].unique():
            fig = plt.figure(figsize=[12, 6])  # 设定图片大小
            ax1 = fig.add_subplot(111)  # 添加第一副图
            sn.pointplot(data=gp[gp.qujian==qj], x=by_time, y='rate_bad', hue='train_or_test', ax=ax1, sort=False,markers='o')
            ax1.legend(loc='upper left', labels=['坏样本率'])
            ax1.set_ylabel('坏样本率')
            plt.setp(ax1.get_xticklabels(), rotation=90, horizontalalignment='right')


            ax2 = ax1.twinx()
            sn.barplot(x=by_time, y='样本比例', data=gp[gp.qujian==qj], ax=ax2, hue='train_or_test', alpha=0.6)
            ax2.set_ylabel('样本比例')
            plt.title('{}_{}_psi:{}'.format(feature_name,qj, psi, ))

    return gp



'''
风险因子的分析；
单个的描述性统计、分布情况、按组划分、
因子同结果变量的关系分析
'''


def cal_psi(df_base, df_test, feature_name, feature_grid=[], n_bin=10, is_same_width=False, default_value=None):
    '''
    支持数字型和非数字型；如果是非数字型，则按每一个取值分组
    计算稳定性-- df_base 剔除空值；df_test 剔除空值
    df_base:以base_bin 进行分位
    df_test:以df_base 为基准，进行分段
    '''
    if (df_base.shape[0] == 0) | (df_test.shape[0] == 0):
        logger.warning('shape of df_base is {} and df_curr is {} '.format(df_base.shape[0], df_test.shape[0]))
        return None

    df_base.loc[df_base[feature_name] == '', feature_name] = None
    df_test.loc[df_test[feature_name] == '', feature_name] == None

    df_base = df_base[(df_base[feature_name].notna()) & (df_base[feature_name] != default_value)]
    if df_base.shape[0] == 0:
        logger.warning('shape of df_base is {}'.format(df_base.shape[0]))
        return None

    df_test = df_test[(df_test[feature_name].notna()) & (df_test[feature_name] != default_value)]
    if df_test.shape[0] == 0:
        logger.warning('shape of df_curr is {}'.format(df_test.shape[0]))
        return None
    # 是否数字进行转换
    df_base[feature_name] = pd.to_numeric(df_base[feature_name], errors='ignore')
    df_test[feature_name] = pd.to_numeric(df_test[feature_name], errors='ignore')
    if pd.api.types.is_numeric_dtype(df_base[feature_name]) == False:
        df_base['qujian'] = df_base[feature_name]
        df_test['qujian'] = df_test[feature_name]
    else:
        if len(feature_grid) == 0:
            feature_grid = cal_feature_grid(df=df_base, feature_name=feature_name, n_bin=n_bin,
                                            is_same_width=is_same_width, default_value=default_value)

        df_base = cal_bin(df_base, feature_name, feature_grid=feature_grid)
        df_test = cal_bin(df_test, feature_name, feature_grid=feature_grid)

    base_gp = df_base.groupby(['qujian'])[feature_name].count().reset_index().rename(columns={feature_name: 'base_cnt'})
    base_gp['base_rate'] = np.round(base_gp['base_cnt'] / df_base.shape[0], 4)

    test_gp = df_test.groupby(['qujian'])[feature_name].count().reset_index().rename(columns={feature_name: 'test_cnt'})
    test_gp['test_rate'] = np.round(test_gp['test_cnt'] / df_test.shape[0], 4)

    # psi 分组计算求和，分组公式=(base_rate-pre_rate) * ln(base_rate/pre_rate)
    gp = pd.merge(base_gp, test_gp, on=['qujian'], how='outer')
    gp['base_rate'].fillna(0, inplace=True)
    gp['test_rate'].fillna(0, inplace=True)
    eps = np.finfo(np.float32).eps
    gp['psi'] = (gp.base_rate - gp.test_rate) * np.log((gp.base_rate + eps) / (gp.test_rate + eps))
    return np.round(gp['psi'].sum(),4)





if __name__ == "__main__":
    MyBin()