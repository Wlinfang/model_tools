import os
import re
import datetime
import time
import hashlib
import base64

import numpy as np
import pandas as pd
from typing import Union
import logging

# 科学计数法全部显示
pd.set_option('display.float_format', lambda x: '%.0f' % x)
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
    m2.update(s.encode())
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
    df[date_name_new] = df[date_name].dt.strftime('%y-%m')
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
        logger.info('%s is %s', (values,e))
    return values
