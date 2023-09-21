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
from model_tools.utils import toolutil


class RepayStatus(Enum):
    NoRepay = 0  # 未还款
    Repaid = 1  # 已还款
    PartialRepay = 2  # 部分还款


@dataclass(frozen=True)
class PDPConst:
    required_cols = ['user_id', 'loan_id', 'loan_amount', 'loan_time', 'loan_term', 'term_no',
                     'repay_status', 'due_date', 'repay_date', 'due_principle', 'repay_principle']


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
            return False
        n = len(set(self.get_required_cols()) - set(df.columns))
        if n > 0:
            return False
        return True

    def fpd_analysis(self, df, group_cols=[]):
        """
        首逾分析：第一期的逾期分析：如果还款日为空，则默认为当日;部分还款默认为未还款
        以当日的角度，查看首逾逾期率
        :param df:
        :param group_cols:
        :return:
        """
        if not self.__check_data(df):
            raise ValueError('数据为空或必须含有以下字段 %s ' % self.get_required_cols())
        t = df[df.term_no == 1]
        t['repay_status'] = t['repay_status'].astype(int)
        # 如果是部分还款和未还款，则还款日置为空
        t.loc[t.repay_status != RepayStatus.Repaid.value, 'repay_date'] = None
        print('已还款但是还款时间为空的数量：%s' %
              t[(t.repay_status == RepayStatus.Repaid.value) & (t.repay_date.isna())].shape[0])
        # 还款日填充为当日时间
        cur_date = datetime.datetime.now().date()
        t['repay_date'] = t['repay_date'].fillna(cur_date)
        t = toolutil.parse_date(t, 'repay_date')
        # 计算diff_days
        t = toolutil.parse_date(t, 'due_date')
        t['diff_days'] = pd.to_timedelta(t['repay_date'] - t['due_date']).dt.days
        # 提前还款，统一设置为 -1
        t.loc[t.diff_days < 0, 'diff_days'] = -1
        # 计数统计+金额统计
        # 首逾：term=1 ：：逾期放款本金/放款本金
        if group_cols is None or len(group_cols) == 0:
            gp = t.groupby('diff_days').agg(cnt=('user_id', 'count'),
                                            amt=('loan_amount', 'sum')).reset_index()
            gp.sorted_values('diff_days', ascending=True, inplace=True)
            gp['accum_cnt'] = gp['cnt'].cumsum()
            gp['accum_amt'] = gp['amt'].cumsum()
            # 按照订单数统计，>diff_days 为1
            gp['fpdn_num'] = np.round(1 - gp['accum_cnt'] / t.shape[0], 2)
            # 按照金额统计
            gp['fpdn_amt'] = np.round(1 - gp['accum_amt'] / t['loan_amount'].sum(), 2)
        else:
            gp_all = t.groupby(group_cols).agg(
                cnt_all=('user_id', 'count'),
                amt_all=('loan_amount', 'sum')
            ).reset_index()
            gp = t.groupby(group_cols + ['diff_days']).agg(
                cnt=('user_id', 'count'),
                amt=('loan_amount', 'sum')
            ).reset_index()
            gp = gp.merge(gp_all, on=group_cols, how='left')
            gp.sorted_values(group_cols + ['diff_days'], ascending=True, inplace=True)
            gp['accum_cnt'] = gp.groupby(group_cols)['cnt'].cumsum()
            gp['accum_amt'] = gp.groupby(group_cols)['amt'].cumsum()
            # 按照订单数统计，>diff_days 为1
            gp['fpdn_num'] = np.round(1 - gp['accum_cnt'] / gp['cnt_all'], 2)
            # 按照金额统计
            gp['fpdn_amt'] = np.round(1 - gp['accum_amt'] / gp['amt_all'], 2)

        return gp
