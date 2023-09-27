"""
特征工程
"""
import numpy as np
import logging


class FeatureEngineer:
    def __init__(self):
        """
        特征表
        :param feature_dict:
        """
        self.feature_dict = {}
        self.feature_cols = []

    @classmethod
    def build_cls(cls, feature_cols, feature_dict):
        """
        构造 FeatureEngineer 实例
        :param feature_dict:
        :param feature_cols:
        :return:
        """
        instan = cls()
        instan.feature_cols = list(feature_cols) if feature_cols is not None else []
        instan.feature_dict = feature_dict if feature_dict is not None else {}
        return instan

    def update_feature_dict(self, feature_name, feature_desc):
        """
        维护特征字典
        feature_name:新特征名
        feature_desc:新特征描述
        """
        # 新增特征列
        if feature_name not in self.feature_cols:
            self.feature_cols.append(feature_name)
        # 特征字典
        self.feature_dict[feature_name] = feature_desc

    def add_feature_sum(self, df, feature_name, feature_desc, exec_sum_features: list):
        """
        加法类特征：f1+f2+f3
        feature_name：新特征名
        feature_desc:新特征描述
        exec_sum_features: list ，计算累加的特征值
        """
        diff_cols = set(exec_sum_features) - set(df.columns)
        if len(diff_cols) > 0:
            logging.info('有特征不存在，不进行衍生 %s' % diff_cols)
            return df
        df[feature_name] = df[exec_sum_features].sum(axis=1, skipna=True)
        self.update_feature_dict(feature_name, feature_desc)
        return df

    def add_feature_divide(self, df, feature_name, feature_desc, feature_fenzi, feature_fenmu):
        """
        比例类特征 feature_fenzi/feature_fenmu
        feature_name:新特征名
        feature_desc:新特征描述
        feature_fenzi:计算新特征的分子
        feature_fenmu:计算新特征的分母
        """
        diff_cols = set([feature_fenzi, feature_fenmu]) - set(df.columns)
        if len(diff_cols) > 0:
            logging.info('有特征不存在，不进行衍生 %s' % diff_cols)
            return df
        # 7天机构/ 15天机构
        df[feature_name] = np.round(df[feature_fenzi] / df[feature_fenmu], 3)
        # 如果分母为0的情况，则为1
        df.loc[df[feature_fenmu] == 0, feature_name] = 1
        self.update_feature_dict(feature_name, feature_desc)
        return df

    def add_feature_relative_rate(self, df, feature_name, feature_desc, feature_fenzi, feature_fenmu):
        """
        相对变化率:feature_fenzi-feature_fenmu / feature_fenmu
        feature_name:新特征名
        feature_desc:新特征描述
        feature_fenzi:计算新特征的分子
        feature_fenmu:计算新特征的分母
        """
        diff_cols = set([feature_fenzi, feature_fenmu]) - set(df.columns)
        if len(diff_cols) > 0:
            logging.info('有特征不存在，不进行衍生 %s' % diff_cols)
            return df
        # 7天机构/ 15天机构
        df[feature_name] = np.round(
            (df[feature_fenzi] - df[feature_fenmu]) / df[feature_fenmu], 3)
        # 如果分母为0的情况
        df.loc[df[feature_fenmu] == 0, feature_name] = 1
        self.update_feature_dict(feature_name, feature_desc)
        return df

    def add_feature_max(self, df, feature_name, feature_desc, exec_sum_features: list):
        """
        选择exec_sum_features 之中最大值作为新的特征
        :param df:
        :param feature_name:特征名称
        :param feature_desc:特征中文描述
        :param exec_sum_features:提取最大值的特征
        :return:
        """
        diff_cols = set(exec_sum_features) - set(df.columns)
        if len(diff_cols) > 0:
            logging.info('有特征不存在，不进行衍生 %s' % diff_cols)
            return df
        df[feature_name] = df[exec_sum_features].max(axis=1, skipna=True)
        self.update_feature_dict(feature_name, feature_desc)
        return df

    def fill_miss_dummy(self, df, feature_name, fill_value=-999):
        """
        增加哑变量:如果缺失，则为1，非则为0
        :param feature_name 缺失值填充 fill_value
        :return:
        """
        if df is None:
            return None
        if df[df[feature_name].isna()].shape[0] == 0:
            # 无缺失值
            return df
        f_n = feature_name + '_miss'
        df[f_n] = 0
        df.loc[df[feature_name].isna(), f_n] = 1
        if f_n not in self.feature_cols:
            self.feature_cols.append(f_n)
        # 填充
        df[feature_name] = df[feature_name].fillna(fill_value)
        return df
