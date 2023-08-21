'''
获取特征明细数据
'''
import pandas as pd
import json

class FeatureHour:
    # 小时级别的数据
    # bdw_tinyv_outer.b_handler_result_mongo_h 不支持spark 查询，限制为hive_engine查询
    # resultdata: 为string 类型
    def __init__(self,hive_engine,start_date,end_date,view_hour,query_source):
        self.hive_engine=hive_engine
        self.start_date=start_date
        self.end_date=end_date
        self.view_hour=view_hour
        self.query_source=query_source

    def query_src_data(self):
        sql = '''
            SELECT dtl.userid user_id,
                          type,
                          channel,
                          querysource,
                          dtl.dt,
                          bcdt.brand_name,
                          dcd.finance_caliber,dcd.caliber_category,dcd.marketing_caliber,dcd.channel_name,
                          case when to_date(dtl.dt)>to_date(base.first_all_loan_succ_time) then '老客' else '新客' end as user_type,
                          dtl.resultdata
          FROM bdw_tinyv_outer.b_handler_result_mongo_h dtl
          left join edw_tinyv.e_user_base_credit_info_d bcdt on dtl.userid=bcdt.user_id and bcdt.dt=date_sub(current_date(),1)
          LEFT JOIN dim_tinyv.dim_channel_d dcd ON dcd.suffix_code=bcdt.second_channel
            AND dcd.dt =date_sub(current_date(), 1)
          left join edw_tinyv.e_user_base_loan_info_d base ON dtl.userid=base.user_id
          AND base.dt=date_add(current_date(),-1)
          WHERE dtl.dt>='{start_date}'
             and dtl.dt < '{end_date}'
             AND dtl.dh='{view_hour}'
             AND type IN ('credit','loan')
             and querysource='{querysource}'
        '''.format(start_date=self.start_date, end_date=self.end_date, view_hour=self.view_hour,
                   querysource=self.query_source)
        print(sql)
        res = []
        df_chk = pd.read_sql(sql, con=self.hive_engine, chunksize=10000)
        for i in df_chk:
            res.append(i)
        df = pd.concat(res)
        return df


    def parse_resultdata(self,df):
        # 遍历每一行
        res = []
        for ix, rw in df.iterrows():
            t = json.loads(rw['resultdata'])
            for k in t.keys():
                if isinstance(t[k], list):
                    for j in t[k]:
                        # json
                        if isinstance(j, dict):
                            for ky in j.keys():
                                res.append([rw['user_id'], rw['querysource'], rw['brand_name'], rw['finance_caliber'],
                                            rw['caliber_category'], rw['marketing_caliber'], rw['channel_name'],
                                            rw['type'],rw['user_type'], ky, j[ky], rw['dt']])
                        else:
                            # 非json
                            print('非json', k, j, rw['user_id'])
                            res.append([rw['user_id'], rw['querysource'], rw['brand_name'], rw['finance_caliber'],
                                        rw['caliber_category'], rw['marketing_caliber'], rw['channel_name'], rw['type'],rw['user_type'],
                                        k, j, rw['dt']])
                else:
                    res.append([rw['user_id'], rw['querysource'], rw['brand_name'], rw['finance_caliber'],
                                rw['caliber_category'], rw['marketing_caliber'], rw['channel_name'], rw['type'], rw['user_type'], k,
                                t[k], rw['dt']])

        df_qs = pd.DataFrame(res, columns=['user_id', 'query_source', 'reg_brand_name', 'finance_caliber', 'caliber_category'
            , 'marketing_caliber', 'channel_name', 'type','user_type', 'fact_name', 'fact_value', 'dt'])
        # 转化为数字型--
        df_qs['fact_value_number'] = pd.to_numeric(df_qs.fact_value, errors='coerce')
        # 判断缺失率
        df_qs['is_miss'] = 0
        df_qs.loc[df_qs.fact_value_number.isin([-1, -2, -99, -999, -9999]) | df_qs.fact_value_number.isna(), 'is_miss'] = 1
        df_qs.finance_caliber.fillna('空', inplace=True)
        df_qs.loc[df_qs.fact_value_number.isin([-1, -2, -99, -999, -9999]), 'fact_value_number'] = None
        # 剔除重复值
        df_qs.drop_duplicates(['dt', 'user_id', 'type', 'fact_name', 'fact_value'], inplace=True)
        return df_qs






