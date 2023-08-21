'''
获取业务信息
TODO 内置spark 的查询功能
TODO 单元测试add
'''

import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from wejoy_analysis.utils.log import get_simple_logger
logger=get_simple_logger('sample')


def get_register(spark,start_date,end_date,product_type=None,reg_brand_name=None,finance_caliber=None):
    '''
    用户用户生成到注册的信息
    注册：绑定手机号成功即为注册成功
    注册口径： 当日用户生成，当日注册
    Parameters
    ----------
    spark
    start_date
    end_date
    finance_caliber：str or list 注册二级渠道类型；分为粉丝通、营销广告、自然流量等
    product_type:str or list
    reg_brand_name:str or list 注册渠道名称
    Returns  label_is_register：表示当日用户生成当日注册
    -------
    '''

    sql='''
    select ub.user_id,ub.brand_name,
    ub.product_type,ub.first_channel,
    ub.second_channel,ub.first_bind_mobile_succ_time,
    ub.user_create_time,ub.channel_reg_time,ub.channel_sex,
    ub.first_visit_sku,ub.is_invite,
    dcd.channel_name,dcd.caliber_category,
    dcd.marketing_caliber,dcd.channel_pay_host,
    dcd.force_register,dcd.status
    from edw_tinyv.e_user_base_credit_info_d ub
    left join dim_tinyv.dim_channel_d dcd ON dcd.suffix_code=ub.second_channel and dcd.dt =date_sub(current_date(), 1)
    where ub.dt=date_sub(current_date(),1)
    and to_date(ub.user_create_time) >= '{start_date}'
    and to_date(ub.user_create_time) < '{end_date}'
    '''
    # finance_caliber
    if isinstance(finance_caliber,list):
        if len(finance_caliber) > 1:
            sql = sql + " and dcd.finance_caliber in {finance_caliber} ".format(finance_caliber=str(tuple(finance_caliber)))
        else:
            finance_caliber=finance_caliber[0]
    if isinstance(finance_caliber,str) and pd.isnull(finance_caliber)==False:
        sql = sql + " and dcd.finance_caliber = '{finance_caliber}' ".format(finance_caliber=finance_caliber)

    # product_type
    if isinstance(product_type,list):
        if len(product_type) > 1:
            sql = sql + " and ub.product_type in {product_type} ".format(product_type=str(tuple(product_type)))
        else:
            product_type=product_type[0]
    if isinstance(product_type,str) and pd.isnull(product_type)==False:
        sql = sql + " and ub.product_type = '{product_type}' ".format(product_type=product_type)
    #brand_name-注册渠道
    if isinstance(reg_brand_name,list):
        if len(reg_brand_name) > 1:
            sql = sql + " and ub.brand_name in {brand_name} ".format(brand_name=str(tuple(reg_brand_name)))
        else:
            reg_brand_name=reg_brand_name[0]
    if isinstance(reg_brand_name,str) and pd.isnull(reg_brand_name)==False:
        sql = sql + " and ub.brand_name = '{brand_name}' ".format(brand_name=reg_brand_name)

    ds = pd.period_range(start_date, end_date, freq='5D')
    ds = ds.to_native_types().tolist()
    if len(ds) == 1:
        ds = [start_date, end_date]
    else:
        ds[0] = start_date
        ds[-1] = end_date
    res = []
    for i in range(1, len(ds), 1):
        print(sql.format(start_date=ds[i - 1], end_date=ds[i]))
        res.append(spark.sql(sql.format(start_date=ds[i - 1], end_date=ds[i])).toPandas())
    df = pd.concat(res)
    df['user_create_month'] = pd.to_datetime(df.user_create_time).dt.strftime('%Y-%m')
    # 用户生成距离绑定手机号天数
    df['diff_create_bind_mobile_days'] = (pd.to_datetime(df.first_bind_mobile_succ_time).dt.date - pd.to_datetime(df.user_create_time).dt.date).dt.days
    # 当日生成；当日注册
    df['label_is_register'] = 0
    df.loc[df.diff_create_bind_mobile_days == 0, 'label_is_register'] = 1
    return df



def get_r0(spark,start_date,end_date,finance_caliber=None,product_type=None,reg_brand_name=None):
    '''
    提取注册到首次授信发起的数据提取
    TODO 发现 同一天多个sku同时进行授信；这个是怎样的操作？？？？
    spark
    start_date:注册开始时间
    end_date：注册截止时间
    finance_caliber:str or list 注册二级渠道类型；分为粉丝通、营销广告、自然流量等
    product_type：str or list
    reg_brand_name str or list 注册渠道名
    return 返回标签：label_r0:表示当日注册当日授信申请
    '''
    sql='''
    select ub.user_id,ub.brand_name,ub.product_type,
    ub.first_channel,ub.second_channel,
    ub.is_written_off,ub.written_off_time,ub.written_off_memo,
    ub.recent_7_login_day_cnt,ub.recent_30_login_day_cnt,
    ub.first_bind_mobile_succ_time,
    ub.channel_reg_time,ub.channel_bind_mobile_time,ub.bind_ocr_succ_time,
    ub.user_sex,ub.user_age,ub.user_birthday,id_card_info['card_city'] ip_city,
    credit.apply_time,credit.sku,credit.group_code_judge,
    dcd.finance_caliber,dcd.channel_name,dcd.caliber_category,
    dcd.marketing_caliber,dcd.channel_pay_host
    from edw_tinyv.e_user_base_credit_info_d ub
    left join dim_tinyv.dim_channel_d dcd ON dcd.suffix_code=ub.second_channel  and dcd.dt =date_sub(current_date(), 1)
    left join edw_tinyv.e_credit_apply_sku_core_d credit on ub.user_id=credit.user_id and credit.dt=date_sub(current_date(),1)
    and credit.credit_sku_apply_no_asc=1
    where ub.dt=date_sub(current_date(),1)
    and ub.first_bind_mobile_succ_time >= '{register_start_date}'
    and ub.first_bind_mobile_succ_time < '{register_end_date}'
    '''

    # finance_caliber
    if isinstance(finance_caliber,list):
        if len(finance_caliber) > 1:
            sql = sql + " and dcd.finance_caliber in {finance_caliber} ".format(finance_caliber=str(tuple(finance_caliber)))
        else:
            finance_caliber=finance_caliber[0]
    if isinstance(finance_caliber,str) and pd.isnull(finance_caliber)==False:
        sql = sql + " and dcd.finance_caliber = '{finance_caliber}' ".format(finance_caliber=finance_caliber)

    # product_type
    if isinstance(product_type,list):
        if len(product_type) > 1:
            sql = sql + " and ub.product_type in {product_type} ".format(product_type=str(tuple(product_type)))
        else:
            product_type=product_type[0]
    if isinstance(product_type,str) and pd.isnull(product_type)==False:
        sql = sql + " and ub.product_type = '{product_type}' ".format(product_type=product_type)
    #brand_name-注册渠道
    if isinstance(reg_brand_name,list):
        if len(reg_brand_name) > 1:
            sql = sql + " and ub.brand_name in {brand_name} ".format(brand_name=str(tuple(reg_brand_name)))
        else:
            reg_brand_name=reg_brand_name[0]
    if isinstance(reg_brand_name,str) and pd.isnull(reg_brand_name)==False:
        sql = sql + " and ub.brand_name = '{brand_name}' ".format(brand_name=reg_brand_name)

    ds = pd.period_range(start_date, end_date, freq='5D')
    ds = ds.to_native_types().tolist()
    if len(ds) == 1:
        ds = [start_date, end_date]
    else:
        ds[0] = start_date
        ds[-1] = end_date
    res = []
    for i in range(1, len(ds), 1):
        print(sql.format(register_start_date=ds[i - 1], register_end_date=ds[i]))
        res.append(spark.sql(sql.format(register_start_date=ds[i - 1], register_end_date=ds[i])).toPandas())
    df = pd.concat(res)
    df['register_month'] = pd.to_datetime(df.first_bind_mobile_succ_time).dt.strftime('%Y-%m')
    # 注册到首次授信申请的天数
    df['diff_register_credit_days'] = (pd.to_datetime(df.apply_time).dt.date - pd.to_datetime(df.first_bind_mobile_succ_time).dt.date).dt.days
    # 当日注册当日授信申请
    df['label_r0'] = 0
    df.loc[df.diff_register_credit_days == 0, 'label_r0'] = 1
    return df


def get_r1(spark,start_date,end_date,sku=None,product_type=None,reg_brand_name=None,finance_caliber=None):
    '''
    授信发起到授信通过的业务数据
    Parameters
    spark
    start_date:授信申请开始时间
    end_date：授信申请结束时间
    sku：str or list
    product_type:str or list
    reg_brand_name:str or list 注册渠道
    finance_caliber:str or list 注册二级渠道类型；分为粉丝通、营销广告、自然流量等
    Returns  label_r1 当日申请当日审核通过；
    '''
    sql='''
    SELECT credit.user_id,credit.credit_id,
    credit.apply_status,credit.apply_time,credit.audit_time,
    credit.credit_amount,credit.credit_sku_apply_no_asc,
    credit.group_code_judge,credit.sku,credit.product_type,
    credit.is_precredit,credit.risk_id,credit.reject_day,
    credit.reg_brand_name,credit.reg_second_channel,credit.bi_second_channel,
    ub.user_sex,ub.user_age,ub.user_birthday,id_card_info['card_city'] ip_city,
    ub.first_bind_mobile_succ_time,
    base.first_all_loan_apply_succ_time,base.first_all_loan_succ_time,
    dcd.finance_caliber,dcd.channel_name,dcd.caliber_category,dcd.marketing_caliber,
    dcd.channel_pay_host
    FROM edw_tinyv.e_credit_apply_sku_core_d credit
    LEFT JOIN dim_tinyv.dim_channel_d dcd ON dcd.suffix_code=credit.reg_second_channel
    AND dcd.dt =date_sub(current_date(), 1)
    LEFT JOIN edw_tinyv.e_user_base_credit_info_d ub ON ub.user_id=credit.user_id
    AND ub.dt=date_sub(current_date(),1)
    LEFT JOIN edw_tinyv.e_user_base_loan_info_d base ON credit.user_id=base.user_id
    AND base.dt=date_add(current_date(),-1)
    WHERE credit.dt=date_sub(current_date(),1)
    and credit.apply_time >= '{start_date}'
    and credit.apply_time < '{end_date}'
    '''

    # sku
    if isinstance(sku,list):
        if len(sku) > 1:
            sql = sql + " and credit.sku in {sku} ".format(sku=str(tuple(sku)))
        else:
            sku=sku[0]
    if isinstance(sku,str) and pd.isnull(sku)==False:
        sql = sql + " and credit.sku = '{sku}' ".format(sku=sku)

    # product_type
    if isinstance(product_type, list):
        if len(product_type) > 1:
            sql = sql + " and credit.product_type in {product_type} ".format(product_type=str(tuple(product_type)))
        else:
            product_type = product_type[0]
    if isinstance(product_type, str) and pd.isnull(product_type) == False:
        sql = sql + " and credit.product_type = '{product_type}' ".format(product_type=product_type)

    # brand_name-注册渠道
    if isinstance(reg_brand_name, list):
        if len(reg_brand_name) > 1:
            sql = sql + " and credit.reg_brand_name in {reg_brand_name} ".format(brand_name=str(tuple(reg_brand_name)))
        else:
            reg_brand_name = reg_brand_name[0]
    if isinstance(reg_brand_name, str) and pd.isnull(reg_brand_name) == False:
        sql = sql + " and credit.reg_brand_name = '{reg_brand_name}' ".format(brand_name=reg_brand_name)

    # finance_caliber
    if isinstance(finance_caliber,list):
        if len(finance_caliber) > 1:
            sql = sql + " and dcd.finance_caliber in {finance_caliber} ".format(finance_caliber=str(tuple(finance_caliber)))
        else:
            finance_caliber=finance_caliber[0]
    if isinstance(finance_caliber,str) and pd.isnull(finance_caliber)==False:
        sql = sql + " and dcd.finance_caliber = '{finance_caliber}' ".format(finance_caliber=finance_caliber)

    ds = pd.period_range(start_date, end_date, freq='5D')
    ds = ds.to_native_types().tolist()
    if len(ds) == 1:
        ds = [start_date, end_date]
    else:
        ds[0] = start_date
        ds[-1] = end_date
    res = []
    for i in range(1, len(ds), 1):
        print(sql.format(start_date=ds[i - 1], end_date=ds[i]))
        res.append(spark.sql(sql.format(start_date=ds[i - 1], end_date=ds[i])).toPandas())
    df = pd.concat(res)

    df['apply_month'] = pd.to_datetime(df.apply_time).dt.strftime('%Y-%m')
    # 是否审核通过-当日申请当日审核通过口径
    df['diff_apply_audit_days'] = (pd.to_datetime(df.audit_time).dt.date - pd.to_datetime(df.apply_time).dt.date).dt.days
    # 贷前审核-当日申请当日审核通过
    df['label_r1'] = 0
    df.loc[(df.diff_apply_audit_days == 0) & (df.apply_status=='001002004'), 'label_r1'] = 1
    #TODO 授信申请时的年龄计算
    return df

def get_r2(spark,start_date,end_date,sku=None,product_type=None,reg_brand_name=None,finance_caliber=None):
    '''
    授信审核通过到首次借款申请时的业务数据
    Parameters
    ----------
    spark
    start_date
    end_date
    sku：str or list
    product_type:str or list
    reg_brand_name:str or list 注册渠道
    finance_caliber:str or list 注册二级渠道类型；分为粉丝通、营销广告、自然流量等
    Returns label_r2:当日审核通过当日发起借款申请
    -------
    '''
    #TODO 授信通过后的利率；可用额度，在贷额度
    sql = '''
        SELECT credit.user_id,credit.credit_id,
        credit.apply_status,credit.apply_time,credit.audit_time,
        credit.credit_amount,credit.credit_sku_apply_no_asc,
        credit.group_code_judge,credit.sku,credit.product_type,
        credit.is_precredit,credit.risk_id,credit.reject_day,
        credit.reg_brand_name,credit.reg_second_channel,credit.bi_second_channel,
        ub.user_sex,ub.user_age,ub.user_birthday,id_card_info['card_city'] ip_city,
        ub.first_bind_mobile_succ_time,
        loan.apply_time loan_apply_time,loan.loan_id,
        base.first_all_loan_apply_succ_time,base.first_all_loan_succ_time,
        dcd.finance_caliber,dcd.channel_name,dcd.caliber_category,dcd.marketing_caliber,dcd.channel_pay_host
        FROM edw_tinyv.e_credit_apply_sku_core_d credit
        LEFT JOIN dim_tinyv.dim_channel_d dcd ON dcd.suffix_code=credit.reg_second_channel
        AND dcd.dt =date_sub(current_date(), 1)
        LEFT JOIN edw_tinyv.e_user_base_credit_info_d ub ON ub.user_id=credit.user_id
        AND ub.dt=date_sub(current_date(),1)
        LEFT JOIN edw_tinyv.e_loan_info_core_d loan ON credit.user_id=loan.user_id AND credit.credit_id=loan.credit_id AND credit.sku=loan.sku
        and loan.is_test=0  and loan.dt=date_add(current_date(),-1) and loan.order_no_asc=1
        LEFT JOIN edw_tinyv.e_user_base_loan_info_d base ON credit.user_id=base.user_id
        AND base.dt=date_add(current_date(),-1)
        WHERE credit.dt=date_sub(current_date(),1)
        and credit.apply_status='001002004'
        and credit.audit_time >='{start_date}'
        and credit.audit_time < '{end_date}'
    '''
    # sku
    if isinstance(sku, list):
        if len(sku) > 1:
            sql = sql + " and credit.sku in {sku} ".format(sku=str(tuple(sku)))
        else:
            sku = sku[0]
    if isinstance(sku, str) and pd.isnull(sku) == False:
        sql = sql + " and credit.sku = '{sku}' ".format(sku=sku)

    # product_type
    if isinstance(product_type, list):
        if len(product_type) > 1:
            sql = sql + " and credit.product_type in {product_type} ".format(product_type=str(tuple(product_type)))
        else:
            product_type = product_type[0]
    if isinstance(product_type, str) and pd.isnull(product_type) == False:
        sql = sql + " and credit.product_type = '{product_type}' ".format(product_type=product_type)

    # brand_name-注册渠道
    if isinstance(reg_brand_name, list):
        if len(reg_brand_name) > 1:
            sql = sql + " and credit.reg_brand_name in {reg_brand_name} ".format(brand_name=str(tuple(reg_brand_name)))
        else:
            reg_brand_name = reg_brand_name[0]
    if isinstance(reg_brand_name, str) and pd.isnull(reg_brand_name) == False:
        sql = sql + " and credit.reg_brand_name = '{reg_brand_name}' ".format(brand_name=reg_brand_name)

    # finance_caliber
    if isinstance(finance_caliber, list):
        if len(finance_caliber) > 1:
            sql = sql + " and dcd.finance_caliber in {finance_caliber} ".format(
                finance_caliber=str(tuple(finance_caliber)))
        else:
            finance_caliber = finance_caliber[0]
    if isinstance(finance_caliber, str) and pd.isnull(finance_caliber) == False:
        sql = sql + " and dcd.finance_caliber = '{finance_caliber}' ".format(finance_caliber=finance_caliber)

    ds = pd.period_range(start_date, end_date, freq='5D')
    ds = ds.to_native_types().tolist()
    if len(ds) == 1:
        ds = [start_date, end_date]
    else:
        ds[0] = start_date
        ds[-1] = end_date
    res = []
    for i in range(1, len(ds), 1):
        print(sql.format(start_date=ds[i - 1], end_date=ds[i]))
        res.append(spark.sql(sql.format(start_date=ds[i - 1], end_date=ds[i])).toPandas())
    df = pd.concat(res)
    df['audit_month'] = pd.to_datetime(df.audit_time).dt.strftime('%Y-%m')
    df['diff_audit_loanapply_days'] = (pd.to_datetime(df.loan_apply_time).dt.date - pd.to_datetime(df.audit_time).dt.date).dt.days
    # 口径为 当日审核通过；当日发起申请
    df['label_r2'] = 0
    df.loc[df.diff_audit_loanapply_days == 0, 'label_r2'] = 1
    return df

def get_r3(spark, start_date,end_date,sku=None,product_type=None,reg_brand_name=None,group_code=None,
           finance_caliber=None,is_first_income_loan=False):
    '''
    获取借款申请到借款审核通过的业务数据
    新老客的逻辑：user_type -- 同风控口径 2021-05-24
      注意：首单和新老客的逻辑不一致；首单一定是新客
      新客：用户首笔放款成功 或 用户从未有过放款成功的借据
      次新客： 该笔支用申请时间距离用户首笔放款成功对应的支用申请时间 <=100 天
      老客：该笔支用申请时间距离用户首笔放款成功对应的支用申请时间 >100天
    贷中审核通过标签：label_r3:同日申请同日审核通过; label_is_loan 是否放款；当日申请当日放款
    ----------
    spark
    sku：str or list
    product_type:str or list
    reg_brand_name:str or list 注册渠道
    finance_caliber:str or list 注册二级渠道类型；分为粉丝通、营销广告、自然流量等
    group_code:str or list
    is_first_income_loan:是否首单；True ： 表示首单；False：不限制
    Returns 贷中明细数据 dataframe
    -------
    '''
    sql = '''
    SELECT loan.user_id,loan.credit_id,loan.loan_id,loan.risk_loan_id,
    loan.apply_time,loan.audit_time,loan.loan_time,loan.audit_status,loan.loan_status,loan.rejection_days,
    loan.loan_amount,loan.apply_amount,loan.desired_total_rate,
    loan.loan_day,loan.loan_unit,loan.stage_sum,
    loan.begin_date,loan.end_date,loan.settle_time,
    loan.group_code,loan.product_type,
    loan.reg_first_channel,loan.reg_second_channel,loan.bi_second_channel,loan.reg_brand_name,
    loan.order_no_asc,loan.order_suc_no_asc,loan.is_first_income_loan,
    loan.curr_inloan_amt,loan.max_overdue_day,
    base.first_all_loan_apply_succ_time,base.first_all_loan_succ_time,
    dcd.finance_caliber,dcd.caliber_category,dcd.marketing_caliber,dcd.channel_name,dcd.channel_pay_host
    FROM edw_tinyv.e_loan_info_core_d loan
    LEFT JOIN edw_tinyv.e_user_base_loan_info_d base ON loan.user_id=base.user_id
    AND base.dt=date_add(current_date(),-1)
    LEFT JOIN dim_tinyv.dim_channel_d dcd ON dcd.suffix_code=loan.reg_second_channel
    AND dcd.dt =date_sub(current_date(), 1)
    WHERE loan.dt=date_add(current_date(),-1)
    AND loan.is_test=0
    and loan.apply_time >= '{start_date}'
    and loan.apply_time < '{end_date}'
    '''

    if is_first_income_loan:
        sql = sql + ' and loan.is_first_income_loan=1'

    # sku
    if isinstance(sku, list):
        if len(sku) > 1:
            sql = sql + " and loan.sku in {sku} ".format(sku=str(tuple(sku)))
        else:
            sku = sku[0]
    if isinstance(sku, str) and pd.isnull(sku) == False:
        sql = sql + " and loan.sku = '{sku}' ".format(sku=sku)

    # product_type
    if isinstance(product_type, list):
        if len(product_type) > 1:
            sql = sql + " and loan.product_type in {product_type} ".format(product_type=str(tuple(product_type)))
        else:
            product_type = product_type[0]
    if isinstance(product_type, str) and pd.isnull(product_type) == False:
        sql = sql + " and loan.product_type = '{product_type}' ".format(product_type=product_type)

    # group_code
    if isinstance(group_code, list):
        if len(group_code) > 1:
            sql = sql + " and loan.group_code in {group_code} ".format(group_code=str(tuple(group_code)))
        else:
            group_code = group_code[0]
    if isinstance(group_code, str) and pd.isnull(group_code) == False:
        sql = sql + " and loan.group_code = '{group_code}' ".format(group_code=group_code)

    # brand_name-注册渠道
    if isinstance(reg_brand_name, list):
        if len(reg_brand_name) > 1:
            sql = sql + " and loan.reg_brand_name in {reg_brand_name} ".format(
                brand_name=str(tuple(reg_brand_name)))
        else:
            reg_brand_name = reg_brand_name[0]
    if isinstance(reg_brand_name, str) and pd.isnull(reg_brand_name) == False:
        sql = sql + " and loan.reg_brand_name = '{reg_brand_name}' ".format(brand_name=reg_brand_name)

    # finance_caliber
    if isinstance(finance_caliber, list):
        if len(finance_caliber) > 1:
            sql = sql + " and dcd.finance_caliber in {finance_caliber} ".format(
                finance_caliber=str(tuple(finance_caliber)))
        else:
            finance_caliber = finance_caliber[0]
    if isinstance(finance_caliber, str) and pd.isnull(finance_caliber) == False:
        sql = sql + " and dcd.finance_caliber = '{finance_caliber}' ".format(finance_caliber=finance_caliber)

    ds = pd.period_range(start_date, end_date, freq='5D')
    ds = ds.to_native_types().tolist()
    if len(ds) == 1:
        ds = [start_date, end_date]
    else:
        ds[0] = start_date
        ds[-1] = end_date
    res = []
    for i in range(1, len(ds), 1):
        print(sql.format(start_date=ds[i - 1], end_date=ds[i]))
        res.append(spark.sql(sql.format(start_date=ds[i - 1], end_date=ds[i])).toPandas())
    df = pd.concat(res)

    df['apply_month']=pd.to_datetime(df.apply_time).dt.strftime('%Y-%m')
    # 是否审核通过-当日申请当日审核通过口径
    df['diff_apply_audit_days']=(pd.to_datetime(df.audit_time).dt.date - pd.to_datetime(df.apply_time).dt.date).dt.days
    # 贷中审核通过
    df['label_r3']=0
    df.loc[(df.diff_apply_audit_days==0) & (df.audit_status.isin(['002002005', '002002003'])),'label_r3']=1
    # 是否放款
    df['diff_apply_loan_days'] = (pd.to_datetime(df.loan_time).dt.date - pd.to_datetime(df.apply_time).dt.date).dt.days
    df['label_is_loan'] = 0
    df.loc[df.diff_apply_loan_days == 0, 'label_is_loan'] = 1

    df.reg_first_channel = df.reg_first_channel.astype(str)
    df.reg_second_channel = df.reg_second_channel.astype(str)
    df.audit_status = df.audit_status.astype(str)
    df.loan_status = df.loan_status.astype(str)
    # 新老客类型
    df.order_suc_no_asc.fillna(-1, inplace=True)
    df.order_suc_no_asc = df.order_suc_no_asc.astype(int)
    df['diff_apply_lstloansuccapply_days']=(pd.to_datetime(df.apply_time).dt.date - pd.to_datetime(df.first_all_loan_apply_succ_time).dt.date).dt.days
    df['user_type']='新客'
    # 新客：用户首笔放款成功 或 用户从未有过放款成功的借据
    # 次新客： 该笔支用申请时间距离用户首笔放款成功对应的支用申请时间 <=100 天
    # 老客：该笔支用申请时间距离用户首笔放款成功对应的支用申请时间 >100天
    df.loc[(df.order_suc_no_asc>1) & (df.diff_apply_lstloansuccapply_days<=100),'user_type']='次新客'
    df.loc[(df.order_suc_no_asc > 1) & (df.diff_apply_lstloansuccapply_days > 100), 'user_type'] = '老客'
    return df

def get_loan_repay(spark,start_date,end_date):
    '''
    获取贷后数据
    Parameters
    ----------
    spark
    start_date
    end_date
    Returns
    -------
    '''



def get_feature(spark,feature_name,start_date,end_date,biz_type=None,sku=None,product_type=None,reg_brand_name=None,group_code=None,
                finance_caliber=None):
    '''
    获取单个feature的明细
    Parameters
    ----------
    spark
    biz_type: type:str  value=[credit or loan or None ]; 如果 biz_type='credit' & sku=None；默认为 循环贷；如果想取备用金；则需要指定sku=petty_cash
    feature_name：str
    sku:str or list
    reg_brand_name:str or list
    group_code:str or list
    product_type:str or list
    finance_caliber:str or list ; 指注册渠道
    Returns
    -------
    '''
    sku_biz_type_dict={'cycle_loan':'credit','flexible_loan':'credit_flexible','installment':'credit_huge',
     'petty_cash':'credit_petty','low_rate':'credit_low_rate','virtual_card':'credit_virtual_card'}

    if biz_type =='credit':
        if isinstance(sku, list):
            biz_type=[ sku_biz_type_dict.get(i,None) for i in sku if pd.isnull(i)==False ]
        if isinstance(sku, str) and pd.isnull(sku)==False:
            biz_type=sku_biz_type_dict.get(sku,None)

    # 这个sql的取法；限制了必须是发起调用该特征的
    sql = '''
    select rf.user_id,rf.biz_id,rf.biz_type,rf.apply_month,rf.reg_brand_name,
    rf.reg_second_channel,rf.biz_code,rf.apply_time,rf.apply_status,rf.apply_asc_no,rf.group_code,rf.{query_key},
    info.sku,info.product_type,info.audit_time,
    dcd.finance_caliber,dcd.caliber_category,dcd.marketing_caliber,dcd.channel_name,dcd.channel_pay_host,
    base.first_all_loan_apply_succ_time,base.first_all_loan_succ_time
    from (
        SELECT user_id,biz_id,biz_type,apply_month,reg_brand_name,reg_second_channel,biz_code,apply_time,apply_status,
              apply_asc_no,group_code,value AS {query_key}
        FROM edw_tinyv.e_risk_feature_d LATERAL VIEW explode(resultdata) res AS KEY,value
        WHERE dt=date_add(current_date(),-1)
          AND scene='weibo_cashloan'
          AND apply_month='{apply_month}'
          AND res.key='{feature_name}'
    ) rf
    left join (
        select user_id,credit_id biz_id,
        case when sku='cycle_loan' then 'credit'
        when sku='flexible_loan' then 'credit_flexible'
        when sku='installment' then 'credit_huge'
        when sku='petty_cash' then 'credit_petty'
        when sku='low_rate' then 'credit_low_rate'
        when sku='virtual_card' then 'credit_virtual_card'
        end as biz_type,sku,product_type,audit_time from edw_tinyv.e_credit_apply_sku_core_d
        where dt=date_add(current_date(),-1)
        union all
        select user_id,loan_id biz_id,
        'loan' as biz_type,sku,product_type,audit_time from edw_tinyv.e_loan_info_core_d
        where dt=date_add(current_date(),-1)
    ) info on rf.user_id=info.user_id and rf.biz_type=info.biz_type and rf.biz_id=info.biz_id

    LEFT JOIN dim_tinyv.dim_channel_d dcd ON dcd.suffix_code=rf.reg_second_channel
    AND dcd.dt =date_sub(current_date(), 1)

    LEFT JOIN edw_tinyv.e_user_base_loan_info_d base ON rf.user_id=base.user_id
    AND base.dt=date_add(current_date(),-1)

    where rf.apply_time >= '{start_date}'
    and rf.apply_time < '{end_date}'
    '''

    if isinstance(biz_type, list):
        if len(biz_type) > 1:
            sql = sql + " and info.biz_type in {biz_type} ".format(biz_type=str(tuple(biz_type)))
        else:
            biz_type = biz_type[0]
    if isinstance(biz_type, str) and pd.isnull(biz_type) == False:
        sql = sql + " and info.biz_type = '{biz_type}' ".format(biz_type=biz_type)

    if isinstance(sku, list):
        if len(sku) > 1:
            sql = sql + " and info.sku in {sku} ".format(sku=str(tuple(sku)))
        else:
            sku = sku[0]
    if isinstance(sku, str) and pd.isnull(sku) == False:
        sql = sql + " and info.sku = '{sku}' ".format(sku=sku)

    # product_type
    if isinstance(product_type, list):
        if len(product_type) > 1:
            sql = sql + " and info.product_type in {product_type} ".format(product_type=str(tuple(product_type)))
        else:
            product_type = product_type[0]
    if isinstance(product_type, str) and pd.isnull(product_type) == False:
        sql = sql + " and info.product_type = '{product_type}' ".format(product_type=product_type)

    # group_code
    if isinstance(group_code, list):
        if len(group_code) > 1:
            sql = sql + " and info.group_code in {group_code} ".format(group_code=str(tuple(group_code)))
        else:
            group_code = group_code[0]
    if isinstance(group_code, str) and pd.isnull(group_code) == False:
        sql = sql + " and info.group_code = '{group_code}' ".format(group_code=group_code)

    # brand_name-注册渠道
    if isinstance(reg_brand_name, list):
        if len(reg_brand_name) > 1:
            sql = sql + " and rf.reg_brand_name in {reg_brand_name} ".format(
                brand_name=str(tuple(reg_brand_name)))
        else:
            reg_brand_name = reg_brand_name[0]
    if isinstance(reg_brand_name, str) and pd.isnull(reg_brand_name) == False:
        sql = sql + " and rf.reg_brand_name = '{reg_brand_name}' ".format(reg_brand_name=reg_brand_name)

    # finance_caliber
    if isinstance(finance_caliber, list):
        if len(finance_caliber) > 1:
            sql = sql + " and dcd.finance_caliber in {finance_caliber} ".format(
                finance_caliber=str(tuple(finance_caliber)))
        else:
            finance_caliber = finance_caliber[0]
    if isinstance(finance_caliber, str) and pd.isnull(finance_caliber) == False:
        sql = sql + " and dcd.finance_caliber = '{finance_caliber}' ".format(finance_caliber=finance_caliber)

    query_key = str(feature_name).strip().replace('-', '_').replace('.', '')

    res = []
    for i in pd.period_range(start_date, end_date, freq='M'):
        logger.info(sql.format(apply_month=str(i), feature_name=feature_name, start_date=start_date,end_date=end_date, query_key=query_key))
        res.append(spark.sql(sql.format(apply_month=str(i), feature_name=feature_name, start_date=start_date,end_date=end_date, query_key=query_key)).toPandas())

    df = pd.concat(res)
    if df.shape[0] ==0 :
        return df
    df['diff_apply_audit_days'] = (pd.to_datetime(df.audit_time).dt.date-pd.to_datetime(df.apply_time).dt.date).dt.days
    # credit--授信通过率
    df['label_r1']=None
    df.loc[df.biz_type.str.find('credit')>=0,'label_r1']=0
    df.loc[(df.biz_type.str.find('credit')>=0) & (df.apply_status.isin(['001002004'])) & (df.diff_apply_audit_days==0),'label_r1']=1

    # loan -- 贷中通过率
    df['label_r3'] = None
    df.loc[df.biz_type=='loan', 'label_r3'] = 0
    df.loc[(df.biz_type=='loan') & (df.apply_status.isin(['002002003', '002002005'])) & (df.diff_apply_audit_days == 0), 'label_r3'] = 1

    df.loc[df[query_key] == '', query_key] = None
    df[query_key]=pd.to_numeric(df[query_key],errors='ignore')
    return df


def get_feature_by_userids(spark,feature_name,user_ids,start_date,end_date):

    '''
    获取指定用户的feature_name
    Parameters
    ----------
    spark
    feature_name:str
    user_ids:list
    Returns
    -------
    '''
    # 临时视图
    spark.sql('drop table if exists tmp_table')
    df_user=pd.DataFrame(user_ids,columns=['user_id'])
    tmp = spark.createDataFrame(df_user[['user_id']])
    tmp.createOrReplaceTempView("tmp_table")

    sql = '''
    SELECT rf.user_id,rf.biz_id,rf.biz_type,rf.apply_month,rf.reg_brand_name,rf.reg_second_channel,rf.biz_code,rf.apply_time,rf.apply_status,
    rf.apply_asc_no,rf.group_code,rf.resultdata['{feature_name}'] {query_key},
    info.sku,info.product_type,info.audit_time,
    dcd.finance_caliber,dcd.caliber_category,dcd.marketing_caliber,dcd.channel_name,dcd.channel_pay_host,
    base.first_all_loan_apply_succ_time,base.first_all_loan_succ_time
    from edw_tinyv.e_risk_feature_d rf
    join tmp_table usr on rf.user_id=usr.user_id
    left join (
            select user_id,credit_id biz_id,
            case when sku='cycle_loan' then 'credit'
            when sku='flexible_loan' then 'credit_flexible'
            when sku='installment' then 'credit_huge'
            when sku='petty_cash' then 'credit_petty'
            when sku='low_rate' then 'credit_low_rate'
            when sku='virtual_card' then 'credit_virtual_card'
            end as biz_type,sku,product_type,audit_time from edw_tinyv.e_credit_apply_sku_core_d
            where dt=date_add(current_date(),-1)
            union all
            select user_id,loan_id biz_id,
            'loan' as biz_type,sku,product_type,audit_time from edw_tinyv.e_loan_info_core_d
            where dt=date_add(current_date(),-1)
    ) info on rf.user_id=info.user_id and rf.biz_type=info.biz_type and rf.biz_id=info.biz_id
    LEFT JOIN dim_tinyv.dim_channel_d dcd ON dcd.suffix_code=rf.reg_second_channel
    AND dcd.dt =date_sub(current_date(), 1)
    LEFT JOIN edw_tinyv.e_user_base_loan_info_d base ON rf.user_id=base.user_id
    AND base.dt=date_add(current_date(),-1)
    where rf.dt = date_add(current_date(),-1)
    and  rf.apply_time >= '{start_date}'
    and rf.apply_time < '{end_date}'
    and rf.apply_month='{apply_month}'
    '''

    query_key = str(feature_name).strip().replace('-', '_').replace('.', '')

    res = []
    for i in pd.period_range(start_date, end_date, freq='M'):
        logger.info(sql.format(apply_month=str(i), feature_name=feature_name, start_date=start_date, end_date=end_date,
                               query_key=query_key))
        res.append(spark.sql(
            sql.format(apply_month=str(i), feature_name=feature_name, start_date=start_date, end_date=end_date,
                       query_key=query_key)).toPandas())

    df = pd.concat(res)
    df['diff_apply_audit_days'] = (
            pd.to_datetime(df.audit_time).dt.date - pd.to_datetime(df.apply_time).dt.date).dt.days
    # credit--授信通过率
    df['label_r1'] = None
    df.loc[df.biz_type.str.find('credit') >= 0, 'label_r1'] = 0
    df.loc[(df.biz_type.str.find('credit') >= 0) & (df.apply_status.isin(['001002004'])) & (
            df.diff_apply_audit_days == 0), 'label_r1'] = 1

    # loan -- 贷中通过率
    df['label_r3'] = None
    df.loc[df.biz_type == 'loan', 'label_r3'] = 0
    df.loc[(df.biz_type == 'loan') & (df.apply_status.isin(['002002003', '002002005'])) & (
            df.diff_apply_audit_days == 0), 'label_r3'] = 1

    df.loc[df[query_key] == '', query_key] = None
    df[query_key] = pd.to_numeric(df[query_key], errors='ignore')
    return df


def get_features(spark,feature_names):
    '''

    Parameters
    ----------
    spark
    feature_names:list
    Returns
    -------

    '''

def get_decision_by_ids(spark,biz_ids,start_date,end_date,biz_type='credit',sku='cycle_loan'):
    '''
    Parameters
    ----------
    spark
    biz_ids:
    start_date
    end_date
    biz_type:[credit, loan]
    sku:str or list
    Returns
    -------
    '''

    sku_biz_type_dict = {'cycle_loan': 'credit', 'flexible_loan': 'credit_flexible', 'installment': 'credit_huge',
                         'petty_cash': 'credit_petty', 'low_rate': 'credit_low_rate',
                         'virtual_card': 'credit_virtual_card'}

    df_id=pd.DataFrame(biz_ids,columns=['biz_id'])
    spark.sql('drop table if exists tmp_table')
    tmp = spark.createDataFrame(df_id[['biz_id']])
    tmp.createOrReplaceTempView("tmp_table")

    if biz_type == 'loan':
        sql='''
        select mdl.user_id,mdl.biz_id,mdl.strategy_id,mdl.strategy_version,mdl.result_code,mdl.result_message,
        mdl.apply_time,mdl.audit_time,mdl.strategy_stage_type,mdl.strategy_type,mdl.result_result,loan.sku
        from bdw_tinyv.b_fuse_decision_logs_d mdl
        join edw_tinyv.e_loan_info_core_d loan on mdl.user_id=loan.user_id and mdl.biz_id=loan.loan_id
        and loan.dt=date_add(current_date(),-1)
        join tmp_table tt on loan.loan_id=tt.loan_id
        where mdl.dt=date_add(current_date(),-1)
        and mdl.biz_type='loan'
        and mdl.strategy_version_type='N'
        and mdl.apply_time >= '{start_date}'
        and mdl.apply_time < '{end_date}'
        '''
        # sku
        if isinstance(sku, list):
            if len(sku) > 1:
                sql = sql + " and loan.sku in {sku} ".format(sku=str(tuple(sku)))
            else:
                sku = sku[0]
        if isinstance(sku, str) and pd.isnull(sku) == False:
            sql = sql + " and loan.sku = '{sku}' ".format(sku=sku)
    elif biz_type=='credit':
        if isinstance(sku, list):
            biz_type = [sku_biz_type_dict.get(i, None) for i in sku if pd.isnull(i) == False]
        if isinstance(sku, str) and pd.isnull(sku) == False:
            biz_type = sku_biz_type_dict.get(sku, None)
        sql='''
        select mdl.user_id,mdl.biz_id,mdl.biz_type,mdl.strategy_id,mdl.strategy_version,mdl.result_code,mdl.result_message,
        mdl.apply_time,mdl.audit_time,mdl.strategy_stage_type,mdl.strategy_type,mdl.result_result,
        case when mdl.biz_type='credit' then 'cycle_loan'
        when mdl.biz_type='credit_flexible' then 'flexible_loan'
        when mdl.biz_type='credit_huge' then 'installment'
        when mdl.biz_type='credit_petty' then 'petty_cash'
        when mdl.biz_type='credit_low_rate' then 'low_rate'
        when mdl.biz_type='credit_virtual_card' then 'virtual_card'
        end as sku
        from bdw_tinyv.b_fuse_decision_logs_d mdl
        join tmp_table tt on loan.loan_id=tt.loan_id
        where mdl.dt=date_add(current_date(),-1)
        and mdl.strategy_version_type='N'
        and mdl.apply_time >= '{start_date}'
        and mdl.apply_time < '{end_date}'
        '''
        # biz_type
        if isinstance(biz_type, list):
            if len(biz_type) > 1:
                sql = sql + " and mdl.biz_type in {biz_type} ".format(biz_type=str(tuple(biz_type)))
            else:
                biz_type = biz_type[0]
        if isinstance(biz_type, str) and pd.isnull(biz_type) == False:
            sql = sql + " and mdl.biz_type = '{biz_type}' ".format(biz_type=biz_type)
    else:
        logger.error('---biz_type error !!! -----',ValueError)

    ds = pd.period_range(start_date, end_date, freq='5D')
    ds = ds.to_native_types().tolist()
    if len(ds) == 1:
        ds = [start_date, end_date]
    else:
        ds[0] = start_date
        ds[-1] = end_date
    res = []
    for i in range(1, len(ds), 1):
        print(sql.format(start_date=ds[i - 1], end_date=ds[i]))
        res.append(spark.sql(sql.format(start_date=ds[i - 1], end_date=ds[i])).toPandas())
    df = pd.concat(res)
    return df
