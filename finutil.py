'''
关于业务的一些分析计算
账龄分析，滚动率，迁徙率，回款计算等
'''

'''
首先，数据准备，必须有以下字段
总期数(stage_sum)、期数(term_no)、应还金额(plan_amount)、
逾期天数(overdue_day)、贷款金额(loan_amount)、
应还时间(deadline)、放款月份(loan_time)、还款状态(repay_status)
'''

rename_columns={'':'stage_sum','step_no':'term_no',}



class  Vintage:
	"""
	vintage：账龄分析
	df ： loan_id loan_amount loan_time loan_term
		  plan_id term_no due_time repay_time plan_prin_amt【应还本金】 act_prin_amt【实还本金】
		  repay_time 如果未还款，则为null 

	MOB0:未逾期
	MOB1:1-30
	MOB2:31-60
	MOB3:61-90
	MOB4:90-120
	MOB5:121-150
	MOB6:151-180
	MOB7:181+
			MOB0  MOB1 MOB2 MOB3 MOB4 MOB5 MOB6 MOB7
	放款月份
	分为2种维度，金额和笔数维度
	无需考虑放款期数；
	分母：如果使用应还金额，则更接近实际；如果使用借款金额，则具有延后性才能体现出来
	"""
	def __init__(self, flag_m=30,mod_method='month',overdue_method='over'):
		'''
		mod_method 观察日口径，分月末时点和期末时点
		月末时点 month：
		6-2号放款，mob0 = 6-2号到6-30号；mob1 = 6-2号到7-31号
		期末时点 term：
		6-2号放款：mob0 = 6-2号到7-2号；mob1=6-2号到8-2号
		逾期计算方式 overdue_method 
		over：曾经逾期，也为逾期； current：截止到观察日，如果曾经逾期但结清，则为未逾期

		flag_m:逾期标志；>30,贼m1;
		'''
		self.mod_method = mod_method
		self.overdue_method=overdue_method
		self.flag_m=flag_m

	def cal_mod_date(self,loan_time,mob_num=0):
		'''
		mob_num: 账龄；0，1，2，对应mob0,mob1...
		'''
		loan_time=pd.to_datetime(loan_time)
		if self.mod_method=='month':
			# 下个月月份
			tmp=loan_time+dateutil.relativedelta.relativedelta(months=mob_num+1)
			# 账龄月末
			tmp=datetime.date(tmp.year,tmp.month,1)+datetime.timedelta(days=-1)
			return tmp.date()
		elif self.mod_method=='term':
			# 期末时点，这块假设每期32天，规定是到期第二天
			tmp=loan_time+datetime.timedelta(days=(mob_num+1) * 32)
			return tmp.date()
		else:
			raise ValueError('mod_method must be month or term ')

		raise ValueError('loan_time error')

	def get_mod_date(self,df):

		# 最大账龄
		mob_max=df.loan_term.max()
		res=[]
		for i in range(0,mob_max+1):
			df['mob']=i
			df['mob_date']=df.loan_time.apply(lambda x:self.cal_mod_date(x,i))
			res.append(df)
		return pd.concat(res)

	def get_passdue_day(self,df):
		'''
		计算逾期天数 2种口径计算方法
		曾经逾期口径、当前口径逾期天数
		截止到观察日的逾期天数
		'''
		df.mod_date=pd.to_datetime(df.mod_date).dt.date
		df.due_time=pd.to_datetime(df.due_time).dt.date 
		df.repay_time=pd.to_datetime(df.repay_time).dt.date 

		df['overdue_day']=np.nan 
		# 如果还款日未到观察日，则未逾期
		con=df.due_time>=df.mod_date
		df.loc[con,'overdue_day']=0

		# 当前日
		curr_date=datetime.datetime.now().date

		# 应还日，当前日 观察日 且未还款
		con=(df.repay_time.isna()) & (df.due_time < df.mod_date) & (curr_date < df.mod_date)
		df.loc[con,'overdue_day']=(curr_date-df[con].due_time).dt.days

		# 应还日，观察日 当前日 且未还款
		con=(df.repay_time.isna()) & (df.due_time < df.mod_date) & (curr_date >= df.mod_date)
		df.loc[con,'overdue_day']=(df[con].mod_date-df[con].due_time).dt.days

		# 应还日，观察日，还款日 且已还款
		con=(df.repay_time >= df.mod_date) & (df.due_time<df.mod_date)
		df.loc[con,'overdue_day']=(df[con].mod_date-df[con].due_time).dt.days 

		# 应还日，还款日，当前日，观察日 且已还款
		con=(df.repay_time < df.mod_date) & (curr_date > df.repay_time)
		df.loc[con,'overdue_day']=(df[con].repay_time-df[con].due_time).dt.days

		#  应还日，当前日，还款日，观察日 且已还款
		con=(curr_date < df.repay_time) & (df.repay_time < df.mod_date)
		df.loc[con,'overdue_day']=(curr_date-df[con].due_time).dt.days 

		if self.overdue_method=='current':
			# 观察日之前结清
			con=(df.repay_time < df.mod_date)
			df.loc[con,'overdue_day']=0

		# 有些提前还款的
		con=df.overdue_day < 0 
		df.loc[con,'overdue_day']=0
		return df 

	def get_vintage_detail(self,df):
		'''
		vintage 计算的明细，每笔订单的 放款详情、剩余未还本金、label
		'''
		mob_max=df.loan_term.max()
		res=[]
		for i in range(0,mob_max+1):
			# 观察日计算
			df['mob']=i
			t=df[['loan_id','loan_time']].drop_duplicates(['loan_id'])
			t['mob_date']=t.loan_time.apply(lambda x:self.cal_mod_date(x,i))
			if 'mod_date' in df.columns:
				df.drop(['mod_date'],axis=1,inplace=True)
			df=df.merge(t,on='loan_id',how='left')
			# 逾期天数
			df=self.get_passdue_day(df)
			# 逾期标志
			df['label']=0
			df.loc[df.overdue_day > self.flag_m,'label']=1
			# 剩余本金
			cols=['loan_id','loan_amount','loan_term','loan_time','mod','mod_date']
			gp=df.groupby(cols).label.max().reset_index()

			tmp=df[(df.loan_id.isin(df[df.label==1].loan_id)) & (df.repay_time < df.mod_date)].groupby(cols).act_prin_amt.sum().reset_index()
			gp=gp.merge(tmp,on=cols,how='left')
			gp.act_prin_amt.fillna(0,inplace=True)
			gp['逾期未还本金']=gp.loan_amount-gp.act_prin_amt
			res.append(gp)

		df_v=pd.concat(res)
		df_v['loan_month']=pd.to_datetime(df_v.loan_time).dt.strftime('%Y-%m')
		return df_v 
		

	def cal_vintage(self,df):
		'''
		vintage 统计计算，这个只是 loan_month,mob 进行了统计
		如果需要更详细的 基于期数，渠道等的统计，则get_vintage_detail，在进行计算
		'''
		df_v=self.get_vintage_detail(df)
		gp=df_v.groupby(['loan_month','mob']).agg(放款本金=('loan_amount','sum'),逾期剩余未还本金=('逾期未还本金','sum'),
			loan_cnt=('loan_id','count'),逾期样本量=('label','sum')).reset_index()
		gp['vintage']=np.round(gp['逾期剩余未还本金']/gp['放款本金'],3)
		# 画图

		return gp 

class  FlowRate:
	"""
	迁徙率 计算，月末时点观察
	迁移率 = 前一期逾期金额到下一期逾期金额的转化率
	M0-M1 = 当月进入M1的贷款余额 / 上月末M0的贷款余额
	df ： loan_id loan_amount loan_time loan_term
		  plan_id term_no due_time repay_time plan_prin_amt【应还本金】 act_prin_amt【实还本金】
		  repay_time 如果未还款，则为null 
	"""
	def __init__(self,flag_m=30):
		'''
		flag_m:逾期标志；>30,则为m1;
		'''
		self.mod_method = mod_method
		self.overdue_method=overdue_method
		self.flag_m=flag_m
	def cal_mod_date(self):
		'''
		观察日计算
		'''
	def get_passdue_day(self,df,mod_date_name):
		'''
		计算逾期天数
		mod_date_name:截止到观察日的逾期天数
		'''
		df['overdue_day']=np.nan 
		df.repay_time=pd.to_datetime(df.repay_time).dt.date
		df[mod_date_name]=pd.to_datetime(df[mod_date_name]).dt.date
		curr_date=datetime.datetime.now().date()
		# 还款日一定小于当前日
		# 如果due_time 还款日 观察日
		con = (df.repay_time < df[mod_date_name])
		df.loc[con,'overdue_day']=(df[con].repay_time-df[con].due_time).dt.days
		# 如果due_time   观察日 还款日
		con=(df.repay_time > df[mod_date_name]) 
		df.loc[con,'overdue_day']=(df[con][mod_date_name]-df[con].due_time).dt.days

		# 如果未还款 due_time  观察日 当前日
		con=(df.repay_time.isna()) & (df[mod_date_name] < curr_date)
		df.loc[con,'overdue_day']=(df[con][mod_date_name]-df[con].due_time).dt.days
		# 如果未还款 due_time  当前日 观察日
		con=(df.repay_time.isna()) & (df[mod_date_name] >= curr_date)
		df.loc[con,'overdue_day']=(curr_date-df[con].due_time).dt.days

		df['overdue_m']=np.nan
		df.loc[df.overdue_day < 1,'overdue_m']='m0'
		con=(df.overdue_day >=1) & (df.overdue_day <31)
		df.loc[con,'overdue_m']='m1'
		con=(df.overdue_day >=31) & (df.overdue_day <61)
		df.loc[con,'overdue_m']='m2'
		con=(df.overdue_day >=61) & (df.overdue_day <91)
		df.loc[con,'overdue_m']='m3'
		con=(df.overdue_day >=91) & (df.overdue_day <121)
		df.loc[con,'overdue_m']='m4'
		con=(df.overdue_day >=121) & (df.overdue_day <151)
		df.loc[con,'overdue_m']='m5'
		con=(df.overdue_day >=151) & (df.overdue_day <181)
		df.loc[con,'overdue_m']='m6'
		con=(df.overdue_day >=181) 
		df.loc[con,'overdue_m']='m7+'

		return df 

	def get_flow_rate_detail(self,df):
		'''
		每笔订单的流动-- mod_date 当前日期，loan_id,每一笔订单的剩余本金
		'''
		df=self.get_passdue_day(df,'mod_date')
		cur






