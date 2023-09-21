"""
这个模块的目的是用于贷款逾期率的分析
产品：现金分期类产品
统一字段：user_id,loan_id,loan_amount,loan_time,loan_term,
term_no,repay_status,due_date,repay_date,due_principle,repay_principle
due_date:应还日
repay_date:实还款日
repay_status:还款状态，匹配为 0：未还款 1：已还款  2 ： 部分还款
"""
import datetime
from enum import Enum
import pandas as pd
import numpy as np
from dataclasses import dataclass
from model_tools.utils import toolutil, plotutil


class RepayStatus(Enum):
    NoRepay = 0  # 未还款
    Repaid = 1  # 已还款
    PartialRepay = 2  # 部分还款


@dataclass(frozen=True)
class PDPConst:
    required_cols = ['user_id', 'loan_id', 'loan_amount', 'loan_time', 'loan_term', 'term_no',
                     'repay_status', 'due_date', 'repay_date', 'due_principal', 'repay_principal']


class PDPAnalysis:
    pdp_const = PDPConst()

    @classmethod
    def get_required_cols(cls):
        return cls.pdp_const.required_cols

    def __check_data(self, df: pd.DataFrame) -> bool:
        """
        数据必须含有 required_cols
        :param df:
        :return:
        """
        if df is None:
            return True
        dc = set(self.get_required_cols()) - set(df.columns)
        if len(dc) > 0:
            print('缺少字段 %s ' % dc)
            return True
        return False

    def __preprocess(self, df, view_date=datetime.datetime.now().date()):
        """
        数据预处理
        1、如果部分还款，处理为未还款；
        2、如果还款日>= view_date 则还款日为空，还款金额为0
        3、如果应还日>= view_date 则还款日为空，还款金额为0 ：未到期情况
        :param view_date:观察日
        :return:
        """
        if self.__check_data(df):
            raise ValueError('数据为空或必须含有以下字段 %s ' % self.get_required_cols())
        # 还款状态int
        df['repay_status'] = df['repay_status'].astype(int)
        df = toolutil.parse_date(df, 'due_date')
        df = toolutil.parse_date(df, 'repay_date')
        # 如果是部分还款和未还款，则还款日置为空
        df.loc[df.repay_status != RepayStatus.Repaid.value, 'repay_date'] = None
        # 如果 还款日>=  view_date 统一置为空
        df.loc[df.repay_date >= view_date, 'repay_principal'] = 0
        df.loc[df.repay_date >= view_date, 'repay_date'] = None
        # 如果应还日 >= view_date，未到期 还款金额&还款日置为空
        df.loc[df.due_date >= view_date, 'repay_principal'] = 0
        df.loc[df.due_date >= view_date, 'repay_date'] = None
        check_n = df[(df.repay_status == RepayStatus.Repaid.value) & (df.repay_date.isna())].shape[0]
        if check_n > 0:
            print('已还款但是还款时间为空的数量：%s' % check_n)
        # 填充
        df['repay_date_fill'] = df['repay_date']
        df['repay_date_fill'] = df['repay_date_fill'].fillna(view_date)
        df['diff_days'] = pd.to_timedelta(df['repay_date_fill'] - df['due_date']).dt.days
        # 提前还款，统一设置为 -1
        df.loc[df.diff_days < 0, 'diff_days'] = -1
        return df

    def fpd_stats(self, df, view_date=datetime.datetime.now().date(),
                  overdue_days=7, group_cols=[]) -> pd.DataFrame:
        """
        首逾数据统计，指定逾期天数 如果>overdue_days 则认为逾期
        :param df:
        :param view_date:默认当日
        :param overdue_days:
        :return:maturity_rate : 有表现的占比
        """
        df = self.__preprocess(df, view_date)
        df = df[df.term_no == 1]
        # 选择有表现的
        tt = df[df.due_date < view_date + datetime.timedelta(days=-overdue_days)]
        tt['fpd_n'] = 0
        tt.loc[tt.diff_days > overdue_days, 'fpd_n'] = 1
        if group_cols is None or len(group_cols) == 0:
            # 数量，逾期率，有表现的占比
            tt.shape[0], tt[tt.diff_days > overdue_days].shape[0]
            gp = pd.DataFrame([[tt.shape[0], tt.loan_amount.sum(),
                                tt[tt.diff_days > overdue_days].shape[0],
                                tt[tt.diff_days > overdue_days].loan_amount.sum(),
                                df.shape[0]
                                ]], columns=['cnt', 'amt', 'cnt_overdue', 'amt_overdue',
                                             'cnt_total'])
        else:
            # 有表现期
            gp = tt.groupby(group_cols).agg(
                cnt=('user_id', 'count'),
                amt=('loan_amount', 'sum')
            ).reset_index()
            # 逾期情况
            gp_overdue = tt[tt.diff_days > overdue_days].groupby(group_cols).agg(
                cnt_overdue=('user_id', 'count'),
                amt_overdue=('loan_amount', 'sum')
            ).reset_index()
            # 所有的样本量
            gp_total = df.groupby(group_cols).agg(
                cnt_total=('user_id', 'count')
            ).reset_index()
            gp = gp.merge(gp_overdue, on=group_cols, how='left')
            gp = gp.merge(gp_total, on=group_cols, how='left')

        gp['overdue_days'] = overdue_days
        # 计算逾期-订单数维度
        gp['fpdn_num'] = np.round(gp['cnt_overdue'] / gp['cnt'], 2)
        # 金额维度
        gp['fpdn_amt'] = np.round(gp['amt_overdue'] / gp['amt'], 2)
        # 有表现的占比
        gp['maturity_rate'] = np.round(gp['cnt'] / gp['cnt_total'], 2)

        out_cols = [] if group_cols is None else group_cols + ['overdue_days',
                                                               'cnt', 'cnt_overdue', 'fpdn_num',
                                                               'amt', 'amt_overdue', 'fpdn_amt',
                                                               'maturity_rate']
        gp.fillna(0, inplace=True)
        gp['cnt_overdue'] = gp['cnt_overdue'].astype(int)
        return gp[out_cols]

    def fpd_analysis(self, df, view_date=datetime.datetime.now().date(),
                     min_overdue_days=0, max_overdue_days=30, group_cols=[]):
        """
        首逾分析：第一期的逾期分析：如果还款日为空，则默认为当日;部分还款默认为未还款
        以当日的角度，查看首逾逾期率
        :param df:
        :param 查看 min_overdue_days <= fpd_n < max_overdue_days 逾期率的变化
        :return:
        """
        # 数据预处理
        df = self.__preprocess(df, view_date)
        df = df[df.term_no == 1]
        min_overdue_days = max(0, min_overdue_days)
        max_overdue_days = max(max_overdue_days, 1)
        data = []
        for overdue_days in range(min_overdue_days, max_overdue_days, 1):
            tmp = self.fpd_stats(df, view_date, overdue_days, group_cols)
            data.append(tmp)

        gp = pd.concat(data)
        return gp
