"""
推荐系统相关算法&评估指标
"""
from typing import List
import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def recall(y_true: List[List], y_pred: List[List]):
    """
    对用户推荐N 个物品，
    召回率： 真实&预测的个数 / 真实的个数; 真实中，预测准的占比；越高越好
    """

    def _recal(y_tr: list, y_te: list):
        return len(y_tr), len(set(y_tr) & set(y_te))

    out = Parallel(n_jobs=2)(delayed(_recal)(y_tr, y_te) for y_tr, y_te in zip(y_true, y_pred))
    y_tr_cnt, y_com_cnt = np.sum(out, axis=0)
    # 全量的数据
    return np.round(y_com_cnt / y_tr_cnt, 3)

def precision(y_true: List[List], y_pred: List[List]):
    """
    对用户推荐N 个物品，
    精确率： 真实&预测的个数 / 预测的个数; 预测中，真实的是多少；越高越好
    """
    def _recal(y_tr: list, y_te: list):
        return len(y_te), len(set(y_tr) & set(y_te))
    out = Parallel(n_jobs=2)(delayed(_recal)(y_tr, y_te) for y_tr, y_te in zip(y_true, y_pred))
    y_te_cnt, y_com_cnt = np.sum(out, axis=0)
    # 全量的数据
    return np.round(y_com_cnt / y_te_cnt, 3)


def fill():
    """

    :return:
    """
