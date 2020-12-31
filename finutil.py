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


		
def cal_vintage(df,overdue_day=90):
	'''
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
	这个是动态变化的，不同的时期，计算的MOB2都不一样
	分为2种维度，金额和笔数维度
	无需考虑放款期数；
	分母：如果使用应还金额，则更接近实际；如果使用借款金额，则具有延后性才能体现出来
	'''
	# 订单数+借款总额
	df['loan_time']=pd.to_datetime(df['loan_time'])
	
	gp=




def cal_roll_rate(df,overdue_day=90):
	'''
	计算滚动率，同放款期数相关
	'''

def cal_rate(df,overdue_day=90):
	'''
	计算迁徙率，同放款期数相关
	'''





