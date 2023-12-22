import os
import re
import datetime
import time
import hashlib
import base64
import ast
import numpy as np
import pandas as pd
from typing import Union

import json
from itertools import chain

import logging

# 科学计数法全部显示
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
logger = logging.getLogger(__file__)


def img_to_base64(file_path, file_name):
    """
    图片base64 编码
    如果是发送邮件，则进行图片编码'<img src="data:image/png;base64,{img_64}">'.format(img_64=img_64)
    """
    if os.path.exists(os.path.join(file_path, file_name)):
        with open(os.path.join(file_path, file_name), 'rb') as fp:
            img_data = fp.read()
            img_64 = base64.b64encode(img_data).decode()
            return img_64


def get_md5(s: str) -> str:
    """
    返回字符串的 32位 md5值
    """
    m2 = hashlib.md5()
    m2.update(str(s).encode())
    return m2.hexdigest()


def check_contain_chinese(check_str):
    """
    字符串中是否包含中文
    return : True 返回中文
    """
    # 中文
    pattern = r'[\u4e00-\u9fff]+'
    m = re.search(pattern, check_str, re.MULTILINE)
    if m is None:
        return False
    return True


def parse_date(df, date_name):
    """
    对date_name 转化为日期格式 yyyy-mm-dd
    :param df:
    :param date_name:
    :return:
    """
    df[date_name] = pd.to_datetime(df[date_name], format='%Y-%m-%d', exact=False).dt.date
    return df


def parse_timestamp(df: pd.DataFrame, date_name: str, date_name_new: str):
    """
    日期转时间戳
    date_name：日期
    date_name_new：新的日期
    """
    df[date_name] = pd.to_datetime(df[date_name])
    df[date_name] = df[date_name].astype(str)
    df[date_name_new] = df[date_name].apply(lambda x: time.mktime(time.strptime(x, '%Y-%m-%d %H:%M:%S')))
    return df


def parse_month(df, date_name: str, date_name_new: str):
    """
    日期 date_name 转为新时间 date_name_new 月份
    :return: %y-%m
    """
    columns = df.columns.tolist()
    if date_name not in columns:
        raise ('not found %' % date_name)
    df[date_name] = pd.to_datetime(df[date_name])
    df[date_name_new] = df[date_name].dt.strftime('%Y-%m')
    return df


def parse_week(df: pd.DataFrame, date_name: str, date_name_new: str):
    """
    日期 date_name 转为新时间 date_name_new 周
    :return: %y-%m-%d 每周第一天
    """
    columns = df.columns.tolist()
    if date_name not in columns:
        raise ('not found %' % date_name)
    df[date_name] = pd.to_datetime(df[date_name])
    df[date_name_new] = df[date_name].dt.strftime('%w')
    df[date_name_new] = df[date_name_new].astype(int)
    df[date_name_new] = df.apply(lambda x: x[date_name] + datetime.timedelta(days=-x[date_name_new]), axis=1)
    df[date_name_new] = pd.to_datetime(df[date_name_new]).dt.date
    return df


def cal_diff_days(df, start_date, end_date):
    """
    计算 start_date,end_date 之间的天数差，返回diff_days
    """
    df[[start_date, end_date]] = df[[start_date, end_date]].apply(pd.to_datetime)
    df['diff_days'] = (df[end_date] - df[start_date]) / np.timedelta64(1, 'D')
    return df


def cal_diff_weeks(df, start_date, end_date):
    """
    计算 start_date,end_date 之间的week 差，返回diff_weeks
    """
    df[[start_date, end_date]] = df[[start_date, end_date]].apply(pd.to_datetime)
    df['diff_weeks'] = (df[end_date] - df[start_date]) / np.timedelta64(1, 'W')
    return df


def cal_diff_months(df, start_date, end_date):
    """
    计算 start_date,end_date 之间的month 差，返回diff_months
    """
    df[[start_date, end_date]] = df[[start_date, end_date]].apply(pd.to_datetime)
    df['diff_months'] = (df[end_date] - df[start_date]) / np.timedelta64(1, 'M')
    return df


def cal_diff_years(df, start_date, end_date):
    """
    计算 start_date,end_date 之间的year 差，返回diff_years
    """
    df[[start_date, end_date]] = df[[start_date, end_date]].apply(pd.to_datetime)
    df['diff_years'] = (df[end_date] - df[start_date]) / np.timedelta64(1, 'Y')
    return df


def del_none(values: Union[list, np.array, pd.Series]) -> np.array:
    """
    删除空值
    """
    if values is None:
        return values
    values = np.array(values)
    # 剔除 None
    values = values[values != np.array(None)]
    # 剔除空字符串
    vt = values.dtype.kind
    if vt not in ['i', 'u', 'f', 'c']:
        values = values[values != '']
        values = values.astype(float)
    # 剔除 np.nan-- 必须是数字类型,如果不是，则不进行剔除
    try:
        values = values[~np.isnan(values)]
    except ValueError as e:
        # 非数字类型，不进行剔除
        logger.info('%s is %s', (values, e))
    return values


def join_list(xs: list):
    """
    ','.join(xs) 去空&排序后的处理
    :return:str
    """
    xs = [x for x in xs if not pd.isna(x)]
    if len(xs) == 0:
        return ''
    return ','.join(sorted(list(set(xs))))


def parse_cols_to_json(df: pd.DataFrame, feature_cols: list, json_column_name: str) -> pd.DataFrame:
    """
    将 df 中的 feature_cols 组合为一个大的json ,且 作为一个column of df
    """
    cols = list(set(df.columns) - set(feature_cols))
    t = pd.DataFrame.to_dict(df[feature_cols], orient='records', index=True)
    data = {json_column_name: t}
    t = pd.DataFrame(data, index=df.index)
    df = df[cols].merge(t, left_index=True, right_index=True, how='inner')
    return df


def parse_json_to_cols(df: pd.DataFrame, json_col: str) -> pd.DataFrame:
    """
    将 df 中的 json 列数据，展开，json 格式为单层的
    """
    df['tmp_json'] = df[json_col].map(lambda x: ast.literal_eval(x))
    df['tmp_json_column'] = df['tmp_json'].map(lambda x: list(x.keys()))
    # df['tmp_json_column'] = df[json_col].map(lambda x: list(json.loads(x).keys()))
    # 展开的 key
    add_columns = list(set(list(chain(*df['tmp_json_column']))))
    for c in add_columns:
        df[c] = df['tmp_json'].map(lambda x: x.get(str(c)))
    df.drop(['tmp_json_column', 'tmp_json'], axis=1, inplace=True)
    return df
