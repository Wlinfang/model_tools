import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, roc_curve
from enum import Enum
import logging
logger = logging.getLogger(__name__)

def cal_week(df,date_name,date_name_new):
	'''
	:param df: dateframe
	:param date_name: date
	:return: %y-%m-%d 每周第一天
	'''
	columns = df.columns.tolist()
	if date_name not in columns:
		raise ('not found %' % date_name)
	df[date_name] = pd.to_datetime(df[date_name])
	df[date_name_new] = df[date_name].dt.strftime('%w')
	df[date_name_new] = df[date_name_new].astype(int)
	df[date_name_new] = df.apply(lambda x: x[date_name] + datetime.timedelta(days=-x[date_name_new]), axis=1)
	df[date_name_new] = pd.to_datetime(df[date_name_new]).dt.date
	return df


def get_stat(cls, df_data,feature_name,label_name,n_bin=10,qcut_method=1):
	'''
	如果是离散值，则根据离散值进行划分
	如果是连续值，则先进行bin 划分，然后进行计算
	'''
	df_data = pd.DataFrame({'val': df_feature, 'label': df_label})

	# statistics of total count, total ratio, bad count, bad rate

	df_stat = df_data.groupby('val').agg(total=('label', 'count'),
										 bad=('label', 'sum'),
										 bad_rate=('label', 'mean'))
	df_stat['var'] = var
	df_stat['good'] = df_stat['total'] - df_stat['bad']
	df_stat['total_ratio'] = df_stat['total'] / df_stat['total'].sum()
	df_stat['good_density'] = df_stat['good'] / df_stat['good'].sum()
	df_stat['bad_density'] = df_stat['bad'] / df_stat['bad'].sum()

	eps = np.finfo(np.float32).eps
	df_stat.loc[:, 'iv'] = (df_stat['bad_density'] - df_stat['good_density']) * \
						   np.log((df_stat['bad_density'] + eps) / (df_stat['good_density'] + eps))

	cols = ['var', 'total', 'total_ratio', 'bad', 'bad_rate', 'iv', 'val']
	df_stat = df_stat.reset_index()[cols].set_index('var')
	return df_stat



def get_iv(cls, df_label, df_feature):
	df_stat = cls.get_stat(df_label, df_feature)
	return df_stat['iv'].sum()




def cal_eval_binary(ytrue,yprob):
	'''
	二分类的一些指标计算 iv,ks,auc
	ytrue : list or pd.Series 
	yprob : list or pd.Series 
	'''
	fpr, tpr, thr = roc_curve(ytrue, yprob)
	ks = max(abs(tpr - fpr))

	auc = roc_auc_score(ytrue, yprob)
	gini = 2 * auc - 1

	recall = metrics.recall()

	# woe ,iv 


def cal_feature_grid(df,feature_name,n_bin=10,is_same_width=True,default_value=None):
	'''
	剔除空值或默认值
	is_same_width:True: 等宽 否则 等频

	'''
	if df is None:
		return None 
	if df.shape[0] ==0:
		return None 
	tmp=df[(df[feature_name].notna()) & (df[feature_name] != default_value)]
	n=tmp[feature_name].nunique()
	if n ==0 :
		return None 
	if n <= n_bin:
		f=sorted(tmp[feature_name].unique().tolist())
	else:
		if is_same_width:
			mi,mx=tmp[feature_name].min(),v[feature_name].max()
			bin_index = [mi+(mx-mi)*i / bin for i in range(0, n_bin + 1)]
			f=sorted(set(bin_index))
		else:
			bin_index = [i / n_bin for i in range(0, n_bin + 1)]
			f=sorted(set(tmp[feature_name].quantile(bin_index)))
	#TODO 包括无穷大，样本集中数据可能有些最小值，最大值不全
	f[0]=-np.Inf
	f[-1]=np.inf
	return f

def cal_bin(df,feature_name,n_bin=10,is_same_width=False,default_value=-1):
	
	f = cal_feature_grid(t,feature_name,n_bin,is_same_width,default_value)
	if f is None :
		return None 
	# 分为空和非空
	t1 = df[(df[feature_name].notna()) & (df[feature_name] != default_value)].copy()
	# t2 是一个copy 数据设置有warnning，进行copy可以消除，断开同源数据的联系
	t2 = df[(df[feature_name].isna()) | (df[feature_name] == default_value)].copy()
	del df 
	t1['qujian']=pd.cut(t1[feature_name], f, include_lowest=True,precision=4)

	# if n < n_bin:
	#	 b=sorted(list(df[feature_name].unique()))
	#	 t['qujian']=pd.cut(t[feature_name],n_bin)
	# else:
	#	 if is_same_width == False:
	#		 # 等频
	#		 a=np.linspace(0,0.9999,n_bin)
	#		 t['qujian']=pd.qcut(t[feature_name],a,precision=4)
	#	 else :
	#		 # 等宽
	#		 # b=sorted(np.linspace(t[feature_name].min()-0.0001,t[feature_name].max()+0.0001,n_bin))
	#		 t['qujian']=pd.cut(t[feature_name],n_bin,precision=4)
	t1['qujian_bin']=t1['qujian'].cat.codes
	t1['qujian_bin']=t1['qujian_bin'].astype(int)
	t1['qujian_left']=t1['qujian'].apply(lambda x:x.left)
	t1['qujian_left']=t1['qujian_left'].astype(float)
	t1['qujian_right']=t1['qujian'].apply(lambda x:x.right)
	t1['qujian_right']=t1['qujian_right'].astype(float)

	# 如果 df['qujian'] 为空，则为缺失值
	if t2.shape[0] > 0:
		print('miss data ')
		t2['qujian']=None
		t2['qujian_bin']=-1
		t2['qujian_right']=None 
		t2['qujian_right']=None 
		return pd.concat([t1,t2])
	return t1 


def cal_psi(df_base,df_curr,feature_name,n_bin=10,is_same_width=False,default_value=None):
	'''
	计算稳定性-- df_base 剔除空值；df_curr 剔除空值
	df_base:以base_bin 进行分位
	df_curr:以df_base 为基准，进行分段
	'''

	df_base=df_base[(df_base[feature_name].notna()) & (df_base[feature_name] != default_value)]
	if df_base.shape[0] == 0:
		print('shape of df_base is {}'.format(df_base.shape[0]))
		return None 

	df_curr=df_curr[(df_curr[feature_name].notna()) & (df_curr[feature_name] != default_value)]
	if df_curr.shape[0] == 0:
		print('shape of df_curr is {}'.format(df_curr.shape[0]))
		return None 

	f=cal_feature_grid(df_base,feature_name,n_bin,is_same_width,default_value)
	df_base['lbl'] = pd.cut(df_base[feature_name], f, include_lowest=True)
	df_base['lbl_index'] = df_base['lbl'].cat.codes
	base_gp=df_base.groupby(['lbl_index'])[feature_name].count().reset_index().rename(columns={feature_name:'base_cnt'})
	base_gp['base_rate']=np.round(base_gp['base_cnt']/df_base.shape[0],4)

	df_curr['lbl'] = pd.cut(df_curr[feature_name], f, include_lowest=True)
	df_curr['lbl_index'] = df_curr['lbl'].cat.codes
	curr_gp=df_curr.groupby(['lbl_index'])[feature_name].count().reset_index().rename(columns={feature_name:'curr_cnt'})
	curr_gp['curr_rate']=np.round(curr_gp['curr_cnt']/df_curr.shape[0],4)

	# psi 分组计算求和，分组公式=(base_rate-pre_rate) * ln(base_rate/pre_rate)
	gp=pd.merge(base_gp,curr_gp,on=['lbl_index'],how='outer')
	gp['psi']=(gp.base_rate-gp.curr_rate) * np.log(gp.base_rate/gp.curr_rate)
	gp['psi']=np.round(gp['psi'],4)
	gp.loc[gp.curr_rate==0,'psi']=0
	del df_base,df_curr,curr_gp,base_gp
	return gp['psi'].sum()



def cal_univar(df,feature_name,label,n_bin=10,is_same_width=False,default_value=-1):
	# 默认值--缺失值处理
	df=cal_bin(df,feature_name=feature_name,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
	if df is None :
		return None 
	gp=df[df.qujian.notna()].groupby(['qujian']).agg(cnt=(label,'count'),cnt_bad=(label,'sum'),rate_bad=(label,'mean')).reset_index()
	gp['qujian']=gp['qujian'].astype('category')
	gp['qujian_bin']=gp['qujian'].cat.codes
	gp['qujian_bin']=gp['qujian_bin'].astype(int)
	gp['qujian_left']=gp['qujian'].apply(lambda x:x.left)
	gp['qujian_left']=gp['qujian_left'].astype(float)
	na = df[df.qujian.isna()]
	if na.shape[0] > 0:
		gp_na=pd.DataFrame([[None,-1,None,na.shape[0],na[label].sum(),na[label].mean()]],columns=['qujian','qujian_bin','qujian_left','cnt','cnt_bad','rate_bad'])
		gp = pd.concat([gp,gp_na])
	gp['rate_bad']=np.round(gp['rate_bad'],3)
	return gp


def cal_univar_by_classes(df,feature_name,label,hue='',n_bin=10,is_same_width=False,default_value=-1):
	# 计算分组的等频分区
	if pd.isnull(hue) ==False:
		res=[]
		ids=df[hue].unique().tolist()
		for i in ids:
			tmp=df[df[hue]==i]
			gp=cal_univar(tmp,feature_name,label,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
			if gp is None :
				continue 
			gp['hue']=i
			res.append(gp)
		if len(res) == 0:
			return None 
		return pd.concat(res)
	else:
		return cal_univar(df,feature_name,label,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)




def cal_lift(df,feature_name,label,n_bin=10,is_same_width=False):
	# feature_name: 模型分名称
	# label: 标签值；默认1为坏用户；0为好用户
	# is_same_width：分组方式；True：等宽；False：等频

	if df.shape[0] == 0:
		raise ValueError("df is empty ")
		return np.nan 

	# fst:数据分组
	gp=cal_bin(df,feature_name,n_bin,is_same_width=is_same_width)

	# 所有的bad的用户
	cnt_bad=df[df[label]==1].shape[0]
	if cnt_bad==0:
		raise ValueError("no bad user ")
		return np.nan 

	#对分组数据进行计算

	gp=gp.groupby(['qujian','qujian_right','qujian_bin']).agg(cnt_bad=(label,'sum'),cnt=(label,'count'),bad_rate=(label,'mean')).reset_index()
	# 排序，然后进行cum
	gp.sort_values('qujian_bin',ascending=True,inplace=True)
	gp['cum_bad']=gp['cnt_bad'].cumsum()
	gp['cum_cnt']=gp['cnt'].cumsum()
	# 计算 占比； bad 占所有的bad 比例 与 累计总样本
	gp['cum_bad_of_total_bad']=np.round(gp['cum_bad']/cnt_bad,3)
	gp['cum_cnt_of_total_cnt']=np.round(gp['cum_cnt']/df.shape[0],3)
	# lift ,如果 >1 则有识别；if < 1；则无识别
	gp['lift']=np.round(gp['cum_bad_of_total_bad']/gp['cum_cnt_of_total_cnt'],2)
	gp['feature']=feature_name

	out_cols=[['feature','qujian','qujian_bin','qujian_right','bad_rate','cnt_bad','cum_bad','cum_bad_of_total_bad','cnt','cum_cnt','cum_cnt_of_total_cnt','lift']]

	return gp[out_cols] 



def cal_pdp(df,feature_name,model_name,n_bin=10,is_same_width=True):
	'''
	主要计算特征同模型分之间的关系图
	'''
	# fst:数据分组
	gp=cal_bin(df,feature_name,n_bin=n_bin,is_same_width=is_same_width)
	# 按组分计算模型分的均值
	gp=gp.groupby(['qujian','qujian_right','qujian_bin']).agg(cnt=(model_name,'count'),model_avg=(model_name,'mean')).reset_index()
	# 排序，然后进行cum
	gp.sort_values('qujian_bin',ascending=True,inplace=True)
	gp.fillna(-1,inplace=True)
	gp['feature']=feature_name
	gp['model']=model_name

	out_cols=['feature','model','']
	return gp 
