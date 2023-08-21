import pandas as pd
import numpy as np

from pyspark.sql.types import *
import pyspark.sql.functions as F
from sklearn.metrics import *
import xgboost as xgb
from sklearn import tree
import pydotplus
import graphviz
import shap


































def cal_lift(df,feature_name,label,feature_grid=[],n_bin=10,is_same_width=False,default_value=None):
    '''
    feature_name: 模型分或特征名称
    feature_grid: 模型分组； 如果不为空，则按照feature_grid分bin；否则按照原规则进行
    label: 标签值；默认1为坏用户；0为好用户
    is_same_width：分组方式；True：等宽；False：等频
    default_value 默认值和缺失值统一为缺失值

    ks =累计坏样本比例-累计好样本比例最大值，可以表示为 误杀了累计好样本比例的用户的情况下，可以识别出多少的坏样本
    坏样本比例=坏样本个数/坏样本总数
    坏样本率=坏样本数/样本数
    累计坏样本率=累计坏样本数/累计样本数
    '''
    df=cal_bin(df=df,feature_name=feature_name,feature_grid=feature_grid,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
    # 包括了缺失值
    gp=df.groupby(['qujian']).agg(坏样本数=(label,'sum'),样本数=(label,'count'),坏样本率=(label,'mean')).reset_index()
    gp['qujian']=gp['qujian'].astype('category')
    cnt_bad_all=df[label].sum()
    cnt_good_all=df.shape[0]-cnt_bad_all

    gp.rename(columns={'cnt':'样本数','cnt_bad':'坏样本数','rate_bad':'坏样本率'},inplace=True)
    gp=gp[['qujian','样本数','坏样本数','坏样本率']]
    gp['坏样本比例']=np.round(gp['坏样本数']/cnt_bad_all,4)
    gp['好样本比例']=np.round((gp['样本数']-gp['坏样本数'])/cnt_good_all,4)
    gp['样本比例']=np.round(gp['样本数']/df.shape[0],4)
    gp['lift']=np.round(gp['坏样本比例']/gp['样本比例'],2)

    eps = np.finfo(np.float32).eps
    # woe，eps 是为了解决分箱中只有good 或者只有bad 时
    gp['woe']=np.round(np.log((gp['坏样本比例'] + eps) / (gp['好样本比例'] + eps)),3)
    # iv
    gp['iv']=np.round((gp['坏样本比例'] - gp['好样本比例']) * gp['woe'],3)

    # 累计
    gp.sort_values('bucket',ascending=True,inplace=True)
    gp['累计坏样本数']=gp['坏样本数'].cumsum()
    gp['累计样本数']=gp['样本数'].cumsum()
    gp['累计样本比例']=np.round(gp['累计样本数']/df.shape[0],2)
    gp['累计坏样本比例']=np.round(gp['累计坏样本数']/cnt_bad_all,4)
    gp['累计lift']=np.round(gp['累计坏样本比例']/gp['累计样本比例'],2)

    # 这个根据分的递增或下降，可理解为拒绝或通过件的坏样本率
    # 判断递增或递减方向
    if gp[gp.bucket==2]['坏样本率'].values[0] > gp[gp.bucket==gp.bucket.max()]['坏样本率'].values[0]:
        # 递减趋势,分越小坏样本比例越高；拒绝分小的
        gp['通过率']=1-gp['累计样本比例']
        gp['拒绝件坏样本率']=np.round(gp['累计坏样本数']/gp['累计样本数'],4)
        gp['通过件坏样本率']=np.round((cnt_bad_all-gp['累计坏样本数'])/(df.shape[0]-gp['累计样本数']),4)
    else:
        # 递增趋势，分越大坏样本比例越高；拒绝分大的
        gp['通过率']=gp['累计样本比例']
        gp['拒绝件坏样本率']=np.round((cnt_bad_all-gp['累计坏样本数'])/(df.shape[0]-gp['累计样本数']),4)
        gp['通过件坏样本率']=np.round(gp['累计坏样本数']/gp['累计样本数'],4)

    # 可以表示，误杀
    gp['累计好样本比例']=np.round((gp['累计样本数']-gp['累计坏样本数'])/cnt_good_all,4)
    gp['KS']=gp['累计坏样本比例']-gp['累计好样本比例']
    return gp
