import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tqdm
import dask.dataframe as dd
from wejoy_analysis.utils.log import get_simple_logger
from .parq import read

logger = get_simple_logger('data.biz')


def get_r0(
    dt_st,
    dt_ed,
    product_type=None,
    reg_brand_name=None,
    finance_caliber=None,
    T=0
):
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
    credit_filters = {}
    dcd_filters = {}
    if product_type:
        credit_filters['product_type'] = product_type
    if reg_brand_name:
        credit_filters['reg_brand_name'] = reg_brand_name
    if finance_caliber:
        dcd_filters['finance_caliber'] = finance_caliber
    df1 = read('e_user_base_credit_info_d_register', dt_st, dt_ed,
        filters=credit_filters,
        columns=[
            'user_id',
            'brand_name',
            'product_type',
            'first_channel',
            'second_channel',
            'is_written_off',
            'written_off_time',
            'written_off_memo',
            'first_bind_mobile_succ_time',
            'channel_reg_time',
            'channel_bind_mobile_time',
            'bind_ocr_succ_time',
            'user_sex',
            'user_age',
            'user_birthday',
            'card_city'
        ],
        renames={'card_city': 'ip_city'},
        transforms={
            'first_bind_mobile_succ_time': lambda ddf: dd.to_datetime(ddf['first_bind_mobile_succ_time']),
        }
    )

    df2 = read(
        'e_credit_apply_sku_core_d',
        dt_st,
        dt_ed,
        filters={'user_id': df1['user_id']},
        columns=[
            'user_id',
            'sku',
            'apply_time',
            'group_code_judge'
        ],
        transforms={
            'apply_time': lambda ddf: dd.to_datetime(ddf['apply_time']),
        }
    )

    df3 = read('dim_channel_d', dt_st, dt_ed,
        filters={},
        columns=[
            'finance_caliber',
            'caliber_category',
            'marketing_caliber',
            'channel_name',
            'suffix_code',
            'channel_pay_host'
        ],
        renames={'suffix_code': 'second_channel'}
    )

    df = df1.merge(df2, on=['user_id'], how='left')
    df = df.merge(df3, on=['second_channel'], how='left')

    if isinstance(finance_caliber, str):
        df = df[df['finance_caliber'] == finance_caliber]
    elif isinstance(finance_caliber, list):
        df = df[df['finance_caliber'].isin(finance_caliber)]
    df['register_month'] = df['first_bind_mobile_succ_time'].dt.strftime('%Y-%m')
    # 注册到首次授信申请的天数
    df['diff_register_credit_days'] = (df['apply_time'].dt.date - df['first_bind_mobile_succ_time'].dt.date).dt.days
    # 当日注册当日授信申请
    df['label_r0'] = 0
    df.loc[df['diff_register_credit_days'] <= T, 'label_r0'] = 1
    return df


def get_r1(
    dt_st,
    dt_ed,
    sku=None,
    product_type=None,
    reg_brand_name=None,
    finance_caliber=None,
    T=0
):
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
    credit_filters = {}
    if sku:
        credit_filters['sku'] = sku
    if product_type:
        credit_filters['product_type'] = product_type
    if reg_brand_name:
        credit_filters['reg_brand_name'] = reg_brand_name
    df1 = read('e_credit_apply_sku_core_d', dt_st, dt_ed,
        filters=credit_filters,
        columns=[
            'user_id',
            'credit_id',
            'apply_status',
            'apply_time',
            'audit_time',
            'credit_amount',
            'credit_sku_apply_no_asc',
            'group_code_judge',
            'sku',
            'product_type',
            'is_precredit',
            'risk_id',
            'reject_day',
            'brand_name',
            'reg_brand_name',
            'reg_second_channel',
            'bi_second_channel',
        ],
        transforms={
            'apply_time': lambda ddf: dd.to_datetime(ddf['apply_time']),
            'audit_time': lambda ddf: dd.to_datetime(ddf['audit_time']),
        }
    )

    df2 = read('e_user_base_credit_info_d', dt_st, dt_ed,
        filters={'credit_id': df1['credit_id'].values},
        columns=[
            'user_id',
            'credit_id',
            'user_sex',
            'user_age',
            'user_birthday',
            'card_city',
            'first_bind_mobile_succ_time',
        ],
        renames={'card_city': 'ip_city'},
        transforms={
            'first_bind_mobile_succ_time': lambda ddf: dd.to_datetime(ddf['first_bind_mobile_succ_time'])
        }
    )

    df3 = read('e_user_base_loan_info_d_credit', dt_st, dt_ed,
        filters={'credit_id': df1['credit_id'].values},
        columns=[
            'user_id',
            'credit_id',
            'first_all_loan_apply_succ_time',
            'first_all_loan_succ_time',
        ],
        transforms={
            'first_all_loan_apply_succ_time': lambda ddf: dd.to_datetime(ddf['first_all_loan_apply_succ_time']),
            'first_all_loan_succ_time': lambda ddf: dd.to_datetime(ddf['first_all_loan_succ_time'])
        }
    )

    df4 = read('dim_channel_d', dt_st, dt_ed,
        filters={},
        columns=[
            'finance_caliber',
            'caliber_category',
            'marketing_caliber',
            'channel_name',
            'suffix_code',
            'channel_pay_host'
        ],
        renames={'suffix_code': 'reg_second_channel'}
    )

    df = df1.merge(df2, on=['credit_id', 'user_id'], how='left')
    df = df.merge(df3, on=['credit_id', 'user_id'], how='left')
    df = df.merge(df4, on=['reg_second_channel'], how='left')

    if isinstance(finance_caliber, str):
        df = df[df['finance_caliber'] == finance_caliber]
    elif isinstance(finance_caliber, list):
        df = df[df['finance_caliber'].isin(finance_caliber)]

    # 新老客逻辑参考 get_r3, user_type 的生成去掉 order_suc_no_asc 限制 (因为这个字段只在 e_loan_info_core_d)
    df['diff_apply_lstloansuccapply_days'] = ((df['apply_time']).dt.date - (df['first_all_loan_apply_succ_time']).dt.date).dt.days
    df['user_type'] = '新客'
    df.loc[(df.diff_apply_lstloansuccapply_days <= 100), 'user_type'] = '次新客'
    df.loc[(df.diff_apply_lstloansuccapply_days > 100), 'user_type'] = '老客'
    # 是否首单
    df['是否首单'] = 1 - (df['first_all_loan_succ_time'] < df['audit_time']).astype(int)

    df['apply_month'] = df['apply_time'].dt.strftime('%Y-%m')
    # 是否审核通过-当日申请当日审核通过口径
    df['diff_apply_audit_days'] = (df['audit_time'].dt.date - df['apply_time'].dt.date).dt.days
    # 贷前审核-当日申请当日审核通过 -> T+x
    df['label_r1'] = 0
    df.loc[(df['diff_apply_audit_days'] <= T) & (df['apply_status']=='001002004'), 'label_r1'] = 1
    #TODO 授信申请时的年龄计算
    return df


def get_r2(dt_st,
           dt_ed,
           sku=None,
           product_type=None,
           reg_brand_name=None,
           finance_caliber=None,
           T=0
):
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
    credit_filters = {'apply_status': '001002004'}
    dcd_filters = {}
    if sku:
        credit_filters['sku'] = sku
    if product_type:
        credit_filters['product_type'] = product_type
    if reg_brand_name:
        credit_filters['reg_brand_name'] = reg_brand_name
    if finance_caliber:
        dcd_filters['finance_caliber'] = finance_caliber
    df1 = read('e_credit_apply_sku_core_d', dt_st, dt_ed,
        filters=credit_filters,
        columns=[
            'user_id',
            'credit_id',
            'apply_status',
            'apply_time',
            'audit_time',
            'credit_amount',
            'credit_sku_apply_no_asc',
            'group_code_judge',
            'sku',
            'product_type',
            'is_precredit',
            'risk_id',
            'reject_day',
            'brand_name',
            'reg_brand_name',
            'reg_second_channel',
            'bi_second_channel',
        ],
        transforms={
            'apply_time': lambda ddf: dd.to_datetime(ddf['apply_time']),
            'audit_time': lambda ddf: dd.to_datetime(ddf['audit_time']),
        }
    )

    df2 = read('e_user_base_credit_info_d', dt_st, dt_ed,
        filters={'credit_id': df1['credit_id'].values},
        columns=[
            'user_id',
            'credit_id',
            'user_sex',
            'user_age',
            'user_birthday',
            'card_city',
            'first_bind_mobile_succ_time',
        ],
        renames={'card_city': 'ip_city'},
        transforms={
            'first_bind_mobile_succ_time': lambda ddf: dd.to_datetime(ddf['first_bind_mobile_succ_time'])
        }
    )

    df3 = read('e_loan_info_core_d', dt_st, dt_ed,
        filters={'credit_id': df1['credit_id'].values, 'is_test': 0, 'order_no_asc': 1},
        columns=[
            'apply_time',
            'user_id',
            'credit_id',
            'loan_id',
            'sku',
            'order_suc_no_asc',
            'is_first_income_loan',
        ],
        renames={'apply_time': 'loan_apply_time'},
        transforms={
            'loan_apply_time': lambda ddf: dd.to_datetime(ddf['loan_apply_time'])
        }
    )

    df4 = read('e_user_base_loan_info_d_credit', dt_st, dt_ed,
        filters={'credit_id': df1['credit_id'].values},
        columns=[
            'credit_id',
            'user_id',
            'first_all_loan_apply_succ_time',
            'first_all_loan_succ_time'
        ],
        transforms={
            'first_all_loan_apply_succ_time': lambda ddf: dd.to_datetime(ddf['first_all_loan_apply_succ_time']),
            'first_all_loan_succ_time': lambda ddf: dd.to_datetime(ddf['first_all_loan_succ_time'])
        }
    )

    df5 = read('dim_channel_d', dt_st, dt_ed,
        filters=dcd_filters,
        columns=[
            'finance_caliber',
            'caliber_category',
            'marketing_caliber',
            'channel_name',
            'suffix_code',
        ],
        renames={'suffix_code': 'reg_second_channel'}
    )

    df = df1.merge(df2, on=['credit_id', 'user_id'], how='left')
    df = df.merge(df3, on=['credit_id', 'user_id', 'sku'], how='left')
    df = df.merge(df4, on=['credit_id', 'user_id'], how='left')
    df = df.merge(df5, on=['reg_second_channel'], how='left')

    # 新老客逻辑参考 get_r3, user_type 的生成去掉 order_suc_no_asc 限制 (因为这个字段只在 e_loan_info_core_d)
    df['diff_apply_lstloansuccapply_days'] = ((df.audit_time).dt.date - (df.first_all_loan_apply_succ_time).dt.date).dt.days
    df['user_type'] = '新客'
    df.loc[(df.diff_apply_lstloansuccapply_days <= 100), 'user_type'] = '次新客'
    df.loc[(df.diff_apply_lstloansuccapply_days > 100), 'user_type'] = '老客'
    # 是否首单
    df['是否首单'] = 1 - (df['first_all_loan_succ_time'] < df['audit_time']).astype(int)

    df['diff_audit_loanapply_days'] = (pd.to_datetime(df.loan_apply_time).dt.date - pd.to_datetime(df.audit_time).dt.date).dt.days
    # 口径为 当日审核通过；当日发起申请 -> T+x
    df['label_r2'] = 0
    df.loc[df.diff_audit_loanapply_days <= T, 'label_r2'] = 1
    return df


def get_r3(
    dt_st,
    dt_ed,
    sku=None,
    product_type=None,
    reg_brand_name=None,
    group_code=None,
    finance_caliber=None,
    is_first_income_loan=False,
    T=0
):
    filters = {'is_test': 0}
    dcd_filters = {}
    if is_first_income_loan:
        filters['is_first_income_loan'] = 1
    if sku:
        filters['sku'] = sku
    if product_type:
        filters['product_type'] = product_type
    if group_code:
        filters['group_code'] = group_code
    if reg_brand_name:
        filters['reg_brand_name'] = reg_brand_name
    if finance_caliber:
        dcd_filters['finance_caliber'] = finance_caliber
    df1 = read('e_loan_info_core_d', dt_st, dt_ed,
        filters=filters,
        columns=[
            'user_id',
            'credit_id',
            'loan_id',
            'risk_loan_id',
            'apply_time',
            'audit_time',
            'loan_time',
            'audit_status',
            'loan_status',
            'rejection_days',
            'loan_amount',
            'apply_amount',
            'desired_total_rate',
            'loan_day',
            'loan_unit',
            'stage_sum',
            'begin_date',
            'end_date',
            'settle_time',
            'group_code',
            'product_type',
            'reg_first_channel',
            'reg_second_channel',
            'bi_second_channel',
            'brand_name',
            'reg_brand_name',
            'order_no_asc',
            'order_suc_no_asc',
            'is_first_income_loan',
            'curr_inloan_amt',
            'max_overdue_day',
        ],
        transforms={
            'apply_time': lambda ddf: dd.to_datetime(ddf['apply_time']),
            'audit_time': lambda ddf: dd.to_datetime(ddf['audit_time']),
            'loan_time': lambda ddf: dd.to_datetime(ddf['loan_time']),
        }
    )
    df2 = read('e_user_base_loan_info_d_loan', dt_st, dt_ed,
        filters={'loan_id': df1['loan_id'].values},
        columns=[
            'loan_id',
            'user_id',
            'first_all_loan_apply_succ_time',
            'first_all_loan_succ_time',
        ],
        transforms={
            'first_all_loan_apply_succ_time': lambda ddf: dd.to_datetime(ddf['first_all_loan_apply_succ_time'])
        }
    )
    df3 = read('dim_channel_d', dt_st, dt_ed,
        filters=dcd_filters,
        columns=[
            'finance_caliber',
            'caliber_category',
            'marketing_caliber',
            'channel_name',
            'suffix_code',
        ],
        renames={'suffix_code': 'reg_second_channel'}
    )
    df = df1.merge(df2, on=['loan_id', 'user_id'], how='left')
    df = df.merge(df3, on='reg_second_channel', how='left')
    # 是否审核通过-当日申请当日审核通过口径 -> T+x
    df['diff_apply_audit_days'] = ((df.audit_time).dt.date - (df.apply_time).dt.date).dt.days
    # 贷中审核通过
    df['label_r3'] = 0
    df.loc[(df.diff_apply_audit_days <= T)
           & (df.audit_status.isin(['002002005', '002002003'])), 'label_r3'] = 1
    # 是否放款 -> T+x
    df['diff_apply_loan_days'] = ((df.loan_time).dt.date - (df.apply_time).dt.date).dt.days
    df['label_is_loan'] = 0
    df.loc[df.diff_apply_loan_days <= T, 'label_is_loan'] = 1
    df.reg_first_channel = df.reg_first_channel.astype(str)
    df.reg_second_channel = df.reg_second_channel.astype(str)
    df.audit_status = df.audit_status.astype(str)
    df.loan_status = df.loan_status.astype(str)
    # 新老客类型
    df.order_suc_no_asc.fillna(-1, inplace=True)
    df.order_suc_no_asc = df.order_suc_no_asc.astype(int)
    df['diff_apply_lstloansuccapply_days'] = ((df.apply_time).dt.date - (df.first_all_loan_apply_succ_time).dt.date).dt.days
    df['user_type'] = '新客'
    # 新客：用户首笔放款成功 或 用户从未有过放款成功的借据
    # 次新客： 该笔支用申请时间距离用户首笔放款成功对应的支用申请时间 <=100 天
    # 老客：该笔支用申请时间距离用户首笔放款成功对应的支用申请时间 >100天
    df.loc[(df.order_suc_no_asc > 1)
           & (df.diff_apply_lstloansuccapply_days <= 100), 'user_type'] = '次新客'
    df.loc[(df.order_suc_no_asc > 1)
           & (df.diff_apply_lstloansuccapply_days > 100), 'user_type'] = '老客'
    return df


def get_resultdata_key_count(dt_st, dt_ed, biz_type, biz_ids=None):
    filters = {}
    filters['biz_type'] = biz_type
    if biz_ids is not None:
        filters['biz_id'] = biz_ids
    renames = {}
    if biz_type == 'credit':
        biz_id_name = 'credit_id'
    elif biz_type == 'loan':
        biz_id_name = 'loan_id'
    else:
        raise Exception('biz_type should be in [credit, loan]!')
    renames = {'biz_id': biz_id_name}
    ddf = read('e_risk_feature_d_resultdata',
               dt_st,
               dt_st,
               filters=filters,
               columns=['biz_id', 'key', 'value'],
               renames=renames,
               compute=False)
    ddf = ddf.groupby('key')[biz_id_name].count()
    df_feature_count = ddf.compute()
    return pd.DataFrame(df_feature_count.sort_values(ascending=False)).rename(columns={biz_id_name: 'count'})


def get_resultdata(dt_st, dt_ed, keys, biz_type, biz_ids=None):
    filters = {}
    filters['biz_type'] = biz_type
    if biz_ids is not None:
        filters['biz_id'] = biz_ids
    renames = {}
    if biz_type == 'credit':
        biz_id_name = 'credit_id'
    elif biz_type == 'loan':
        biz_id_name = 'loan_id'
    else:
        raise Exception('biz_type should be in [credit, loan]!')
    renames = {'biz_id': biz_id_name}
    _df = read('e_risk_feature_d_resultdata',
               dt_st,
               dt_ed,
               filters=filters,
               columns=['biz_id', 'key', 'value'],
               renames=renames,
               compute=True)
    _df_w = _df.pivot_table(index=biz_id_name, columns='key', values='value', aggfunc='mean')
    _df_w = _df_w.reset_index()
    return _df_w
