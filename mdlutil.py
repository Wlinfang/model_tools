import pandas as pd
import numpy as np
import datetime
import dateutil
from scipy import stats 
# import warnings
# warnings.filterwarnings('ignore')




pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.precision',3)
pd.set_option('display.max_rows',2000)
pd.set_option('display.max_columns',2000)

class  Confidence:
	"""
	假设估计置信度区间计算
	confidence：置信度
	is_biased_estimate：计算标准差是有偏还是无偏；True:有偏；利用样本估计总体标准差
	"""
	def __init__(self, confidence=0.95,is_biased_estimate=False):
		super( Confidence, self).__init__()
		self.confidence = confidence
		self.is_biased_estimate=is_biased_estimate
		
	def cal_norm_confidence(df,feature_name):
		'''
		计算正太分布的置信区间
		confidence:置信度
		is_biased_estimate：计算标准差是有偏还是无偏；True:有偏；利用样本估计总体标准差
		'''
		sample_mean = df[feature_name].mean()
		if self.is_biased_estimate == True:
			# 有偏估计
			sample_std = df[feature_name].std(ddof=0)
		else:
			# 无偏 
			sample_std = df[feature_name].std(ddof=1)
		return stats.norm.interval(self.confidence, loc=sample_mean, scale=sample_std)

	def cal_t_confidence(df,feature_name):
		'''
		计算t分布的置信区间
		'''
		sample_mean = df[feature_name].mean()
		if self.is_biased_estimate == True:
			# 有偏估计
			sample_std = df[feature_name].std(ddof=0)
		else:
			# 无偏 
			sample_std = df[feature_name].std(ddof=1)

		return stats.t.interval(self.confidence, df=(df.shape[0]-1),loc=sample_mean, scale=sample_std)




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

def cal_month(df,date_name,date_name_new):
	'''
	:param df: dateframe
	:param date_name: date
	:param date_name_new：月份名称
	:return: %y-%m
	'''
	columns=df.columns.tolist()
	if date_name not in columns:
		raise('not found %' % date_name)
	df[date_name]=pd.to_datetime(df[date_name])
	df[date_name_new]=df[date_name].dt.strftime('%y-%m')
	return df

def parse_timestamp(df,date_name,date_name_new):
	'''
	日期转时间戳
	date_name:原时间名称
	date_name_new:时间戳的name
	'''
	df[date_name]=pd.to_datetime(df[date_name])
	df[date_name]=df[date_name].astype(str)
	df[date_name_new]=df[date_time].apply(lambda x:time.mktime(time.strptime(x,'%Y-%m-%d %H:%M:%S')))
	return df


def cal_describe(df,feature_name_list):
	'''
	计算特征描述性分析 包括cnt,缺失率，方差，分位数，众数，众数占比
	众数限制为1个的情况，如果多个众数，则不进行计算
	偏度：当偏度<0时，概率分布图左偏。
		 当偏度=0时，表示数据相对均匀的分布在平均值两侧，不一定是绝对的对称分布。
		 当偏度>0时，概率分布图右偏
		 计算公式=1/n Σ((X-μ)/σ)^3  同方差的区别是 2次方变为3次方，其实核心是求矩的
	峰度:对比正太，正太的峰度=3，计算的时候-3，正太峰度=0
		 峰度值>0，则表示该数据分布与正态分布相比较为高尖，
		 当峰度值<0，则表示该数据分布与正态分布相比较为矮胖。
	return df:
	'''
	# 提取数字型数据
	feature_name_list=df[feature_name_list].select_dtypes(include=np.number).columns.tolist()
	
	gp=df[feature_name_list].describe().T
	# 取近似值了
	gp['all_count']=df.shape[0]
	gp['miss_rate']=np.round(1-gp['count']/df.shape[0],3)	
	gp=gp[['all_count','count', 'miss_rate', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
	
	gp[['miss_rate', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]=np.round(gp[['miss_rate', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']],3)

	gp=gp.join(df[feature_name_list].nunique(axis=0).to_frame().rename(columns={0:'nunique'}))
	gp=gp.reset_index().rename(columns={'index':'feature_name'})
	a=df[feature_name_list].mode(axis=0)
	if a.shape[0] ==1 :
		# # a 中过滤有多个众数的特征,多个众数，则不计算
		# if a.shape[0] > 1:
		# 	d=a[1:2].dropna(axis=1,how='all').columns.tolist()
		# 	a=a[set(feature_name_list)-set(d)]
		# 	a=a.T.dropna(axis=1,how='any').reset_index()
		a=a.T.reset_index()
		a.columns=['feature_name','mode']
		gp=gp.merge(a,how='left')
		# 众数占有值的比例
		gp['mode_count']=gp.apply(lambda x:df[df[x.feature_name]==x['mode']].shape[0],axis=1)
		gp['mode_rate']=np.round(gp['mode_count']/gp['count'],3)
	else:
		gp['mode']=None
		gp['mode_count']=None
		gp['mode_rate']=None
	# 计算偏度，<0 左偏 =0 正太分布；>0 右偏
	a=df[feature_name_list].skew(axis=0,skipna=True).reset_index().rename(columns={'index':'feature_name',0:'skew'})
	gp=gp.merge(a,on='feature_name',how='left')

	#计算峰度
	a=df[feature_name_list].kurt(axis=0,skipna=True).reset_index().rename(columns={'index':'feature_name',0:'kurt'})
	gp=gp.merge(a,on='feature_name',how='left')
	return gp


def cal_feature_grid(df,feature_name,n_bin=10,is_same_width=True,default_value=None):
	'''
	计算分组，剔除空值或默认值后剩余的数据进行分组；空值和默认值为单独一个组
	is_same_width:True: 等宽 否则 等频
	'''
	if (df is None)  or (df.shape[0] ==0) or (n_bin <= 0):
		return None 
	tmp=df[(df[feature_name].notna()) & (df[feature_name] != default_value)]
	n=tmp[feature_name].nunique()
	if n ==0 :
		return None 
	if n <= n_bin:
		f=sorted(tmp[feature_name].unique().tolist())
	else:
		if is_same_width:
			mi,mx=tmp[feature_name].min(),tmp[feature_name].max()
			bin_index = [mi+(mx-mi)*i / n_bin for i in range(0, n_bin + 1)]
			f=sorted(set(bin_index))
		else:
			bin_index = [i / n_bin for i in range(0, n_bin + 1)]
			f=sorted(set(tmp[feature_name].quantile(bin_index)))
		# 包括无穷大，样本集中数据可能有些最小值，最大值不全
		f[0]=-np.Inf
		f[-1]=np.inf
	return np.round(f,3)


def cal_bin(df,feature_name,feature_grid=[],n_bin=10,is_same_width=False,default_value=-1):
	
	'''
	对 df中的每笔明细划分区间
	feature_grid：分段区间；如果为空，则根据 n_bin=10,•is_same_width=False 计算得到
	'''

	if (df is None) or (df.shape[0]==0):
		return None 

	if len(feature_grid) == 0:
		feature_grid = cal_feature_grid(df,feature_name,n_bin,is_same_width,default_value)
		if feature_grid is None :
			return None 
	feature_grid = sorted(feature_grid)
	# 分为空和非空
	t1 = df[(df[feature_name].notna()) & (df[feature_name] != default_value)].copy()
	# t2 是一个copy 数据设置有warnning，进行copy可以消除，断开同源数据的联系
	t2 = df[(df[feature_name].isna()) | (df[feature_name] == default_value)].copy()
	del df 
	t1['qujian']=pd.cut(t1[feature_name], feature_grid, include_lowest=True,precision=4)
	t1['qujian']=t1['qujian'].astype('category')
	t1['bucket']=t1['qujian'].cat.codes
	t1['bucket']=t1['bucket'].astype(int)+1

	# 如果 df['qujian'] 为空，则为缺失值
	if t2.shape[0] > 0:
		print('miss data ')
		t2['qujian']='miss data'
		t2['bucket']=-1
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

	f=cal_feature_grid(df=df_base,feature_name=feature_name,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
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
	gp=df.groupby(['qujian']).agg(cnt_bad=(label,'sum'),cnt=(label,'count'),rate_bad=(label,'mean')).reset_index()
	
	gp['qujian']=gp['qujian'].astype('category')
	gp['bucket']=gp['qujian'].cat.codes 
	gp.loc[gp['qujian']=='miss data','bucket']=-1
	gp['bucket']=gp['bucket'].astype(int)+1

	cnt_bad_all=df[label].sum()
	cnt_good_all=df.shape[0]-cnt_bad_all

	gp.rename(columns={'cnt':'样本数','cnt_bad':'坏样本数','rate_bad':'坏样本率'},inplace=True)
	gp=gp[['bucket','qujian','样本数','坏样本数','坏样本率']]
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
	if gp[gp.bucket.min()==2]['坏样本率'].values[0] > gp[gp.bucket==gp.bucket.max()]['坏样本率'].values[0]:
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


def cal_y_by_classes(df,label,sub_classes=[],classes=[]):
	'''
	label : 取值[0，1]；  1：bad; 0: good
	sub_classes:在 classes 的基础上进一步细分;比如 classes=['loan_month'],sub_classes=['loan_month','loan_amount'...]
	如果 sub_classes 为空，则总体计算； 
	'''
	if len(sub_classes) > 0:
		gp=df.groupby(sub_classes).agg(cnt=(label,'count'),cnt_bad=(label,'sum'),rate_bad=(label,'mean')).reset_index()
		if len(classes) > 0 and len(set(sub_classes) & set(classes)) > 0:
			share_classes=list(set(sub_classes) & set(classes))
			gp=gp.merge(df.groupby(classes).agg(cnt_all=(label,'count'),cnt_bad_all=(label,'sum')).reset_index(),on=share_classes,how='left')
			gp['bad_of_total_bad']=np.round(gp['cnt_bad']/gp['cnt_bad_all'],3)
			gp['cnt_of_total_cnt']=np.round(gp['cnt']/gp['cnt_all'],3)
			gp['lift']=np.round(gp['bad_of_total_bad']/gp['cnt_of_total_cnt'],3)

		gp['rate_bad']=np.round(gp['rate_bad'],3)
	else:
		gp = pd.DataFrame([[df.shape[0],df[label].sum(),np.round(df[label].mean(),3)]],columns=['cnt','cnt_bad','rate_bad'])

	return gp 



def cal_pdp(df,feature_name,model_name,feature_grid=[],n_bin=10,is_same_width=False,default_value=None):
	'''
	主要计算特征同模型分之间的关系
	对特征值进行分组,每组计算模型分的均值;模型分不能为空
	feature_grid 优先指定分组区间；
	'''

	df=df[df[model_name].notna()]
	# fst:数据分组
	df=cal_bin(df=df,feature_name=feature_name,feature_grid=feature_grid,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
	if df is None:
		return None
	# 按组分计算模型分的均值
	gp=df.groupby(['qujian']).agg(cnt=(model_name,'count'),model_avg=(model_name,'mean')).reset_index()
	
	gp['qujian']=gp['qujian'].astype('category')
	gp['bucket']=gp['qujian'].cat.codes 
	gp.loc[gp['qujian']=='miss data','bucket']=-1
	gp['bucket']=gp['bucket'].astype(int)+1
	
	gp.sort_values('bucket',ascending=True,inplace=True)
	gp.rename(columns={'cnt':'样本数','model_avg':'模型分均值'},inplace=True)
	out_cols=['bucket','qujian','样本数','模型分均值']
	return gp[out_cols]


def eval_miss(df,feature_name,label,classes=['train_or_test']):
	'''
	评估缺失率的影响
	'''
	gp=df.groupby(classes).apply(lambda x:cal_describe(x,[feature_name])).reset_index()
	gp=gp[classes+['feature_name','all_count','count','miss_rate']]
	gp=gp.merge(df.groupby(classes)[label].mean().reset_index().rename(columns={label:'坏样本率'}),on=classes,how='left')
	df['is_miss']=0
	df.loc[df[feature_name].isna(),'is_miss']=1
	gp_rate=cal_y_by_classes(df,label,sub_classes=classes+['is_miss'],classes=classes)
	gp_no=gp_rate[gp_rate.is_miss==0][classes+['样本数','坏样本率','样本比例']]
	gp_yes=gp_rate[gp_rate.is_miss==1][classes+['样本数','坏样本率','样本比例']]
	
	gp=gp.merge(gp_no,on=classes,how='left')
	gp=gp.merge(gp_yes,on=classes,how='left')
	if len(classes)==1:
		gp.index=gp[classes[0]]
	else:
		gp.index=gp[classes]
	gp.drop(classes,inplace=True,axis=1)
	gp.columns=pd.Index([('整体','特征名'),('整体','样本量'),('整体','有值量'),('整体','缺失率'),('整体','坏样本率'),
		  ('未缺失','样本数'),('未缺失','坏样本率'),('未缺失','样本比例'),
		 ('缺失','样本数'),('缺失','坏样本率'),('缺失','样本比例')])
	return gp
	

def cal_woe(df,feature_name,label,feature_grid=[],n_bin=10,is_same_width=False,default_value=None):
	'''
	Weight of Evidence
	对数据进行分组，
	返回每组的woe,iv值，总的iv值为sum()
	label:[0,1] 1表示bad
	woe: ln( 分组中bad占所有bad的比例 / 分组中good占所有good的比例)
	iv : (分组中bad占所有bad的比例-分组中good占所有good的比例) * woe

	woe:

	1、缺失值的woe 可不满足单调性，因为缺失值尤其逻辑含义
	2、如果相邻分箱的woe值相同，则合并为1个分箱
	3、当一个分箱内只有bad 或者 good时，修正公式公式计算中，加入 eps 
	4、如果训练集woe满足单调性；but 验证集或测试集上不满足，则分箱不合理或者这个特征不稳定，时效性差
	'''
	# fst.数据分组
	df=cal_bin(df=df,feature_name=feature_name,feature_grid=feature_grid,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
	if df is None:
		return None 

	# 分组计算
	gp=df.groupby('qujian').agg(cnt=(label,'count'),cnt_bad=(label,'sum'),rate_bad=(label,'mean')).reset_index()
	gp['qujian']=gp['qujian'].astype('category')
	gp['bucket']=gp['qujian'].cat.codes 
	gp.loc[gp['qujian']=='miss data','bucket']=-1
	gp['bucket']=gp['bucket'].astype(int) +1 
	gp.sort_values('bucket',ascending=True,inplace=True)

	gp['cnt_good']=gp['cnt']-gp['cnt_bad']
	gp['cnt_of_total_cnt']=np.round(gp['cnt']/df.shape[0],3)
	gp['good_of_total_good']=np.round(gp['cnt_good']/gp['cnt_good'].sum(),3)
	gp['bad_of_total_bad']=np.round(gp['cnt_bad']/gp['cnt_bad'].sum(),3)
	eps = np.finfo(np.float32).eps
	# woe，eps 是为了解决分箱中只有good 或者只有bad 时
	gp['woe']=np.log((gp['bad_of_total_bad'] + eps) / (gp['good_of_total_good'] + eps))
	# iv 
	gp['iv']=(gp['bad_of_total_bad'] - gp['good_of_total_good']) * gp['woe']

	# 3 位小数
	gp['woe']=np.round(gp['woe'],3)
	gp['iv']=np.round(gp['iv'],3)

	gp.rename(columns={'cnt':'样本数','cnt_bad':'坏样本数','cnt_good':'好样本数','cnt_of_total_cnt':'样本比例',
		'bad_of_total_bad':'坏样本比例','good_of_total_good':'好样本比例'},inplace=True)

	out_cols=['bucket','qujian','样本数','坏样本数','好样本数','样本比例','坏样本比例','好样本比例','woe','iv']

	return gp[out_cols]


def cal_iv(df,feature_name,label,feature_grid=[],n_bin=10,is_same_width=False,default_value=None):
	'''
	同 cal_woe 的方法的区别：
	cal_woe 中的iv是针对每组的计算；
	cal_iv 则返回一个float ，cal_woe()['iv'].sum()=各个组的iv求和

	woe: 描述了预测和目标的关系，范围是实数范围
	iv:衡量了这个关系的大小

	iv 效果范围：

	[0,0.02) 几乎没有效果
	[0.02,0.1) 弱
	[0.1,0.3) 中等
	[0.3,0.5) 强
	[0.5,∞）效果难以置信，需要确认下
	'''
	t = cal_woe(df,feature_name,label,feature_grid=feature_grid,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
	if t is None:
		return None 
	return np.round(t['iv'].sum(),3)

def cal_corr(df,feature_name_list):
	'''
	特征之家的相关性
	线性相关 ： 皮尔逊系数计算
	非线性相关 ：
	'''
	plt.figure(figsize=(12,6))
	sn.heatmap(df[feature_name_list].corr(),annot=True,cmap="YlGnBu")
	plt.title('person corr')
	plt.show()

	# 非线性相关计算


def get_model_describe(df,feature_name,label,feature_grid=[],n_bin=10,is_same_width=False,default_value=None):
	'''
	对模型分或单特征进行简单的描述评估
	返回 auc,ks,逾期率,样本数，样本量，iv
	'''
	cnt=df.shape[0]
	cnt_bad=df[label].sum()
	rate_bad=np.round(df[label].mean(),3)
	# 缺失值样本量
	cnt_miss=df[df[feature_name].isna()].shape[0]
	rate_miss=np.round(cnt_miss/cnt,3)
	# 缺失值的逾期率
	rate_miss_bad=np.round(df[df[feature_name].isna()][label].mean(),3)
	# 未缺失
	t=df[df[feature_name].notna()]
	cnt_nmiss=t.shape[0]
	rate_nmiss_rate=np.round(t[label].mean(),3)
	auc=roc_auc_score(t[label],t[feature_name])
	fpr, tpr, thr = roc_curve(t[label],t[feature_name])
	ks = max(abs(tpr - fpr))

	iv=cal_iv(df=df,feature_name=feature_name,label=label,feature_grid=feature_grid,n_bin=n_bin,is_same_width=is_same_width,default_value=default_value)
	return pd.DataFrame([[cnt,cnt_bad,rate_bad,cnt_miss,rate_miss,rate_miss_bad,cnt_nmiss,rate_nmiss_rate,auc,ks,iv]],
		columns=['样本数','坏样本数','坏样本率','缺失数','缺失比例','特征缺失坏样本率','特征未缺失数','未缺失坏样本率','auc','ks','iv'])
	

def cal_evaluate_classier(df,y_real,y_pred,label=''):
	
	'''
	label:表示 roc 图的 label 

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
	'''
	# accuracy = metrics.accuracy_score(y_real, y_pred)
	# p=precision_score(y_real, y_pred)
	# f1=f1_score(y_real, y_pred)
	# 返回confusion matrix
	cnt=df.shape[0]
	cnt_bad=df[y_real].sum()
	rate_bad=np.round(df[y_real].mean(),3)

	# 这个y_real y_pred 取值范围一致
	# cm = confusion_matrix(y_real,y_pred)
	# # 返回 精确率，召回率，F1；y_real 和 y_pred 取值范围一致
	# cr = classification_report(y_real,y_pred)
	auc = roc_auc_score(df[y_real],df[y_pred]) 

	fpr, tpr, thr = roc_curve(df[y_real],df[y_pred])
	ks = max(abs(tpr - fpr))
	# roc 图
	plt.plot(fpr,tpr,label=label)
	plt.title('roc_curve')
	plt.xlabel('fpr')
	plt.ylabel('tpr')
	plt.legend(loc='upper right')

	return pd.DataFrame([[cnt,cnt_bad,rate_bad,auc,ks]],columns=['样本数','坏样本数','坏样本率','auc','ks'])









		