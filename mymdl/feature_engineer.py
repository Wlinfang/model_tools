"""
特征工程
"""
import numpy as np


class FeatureEngineer:
    def __init__(self):
        """
        特征表
        :param feature_dict:
        """
        self.feature_dict = {}
        self.feature_cols = []

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
        df[feature_name] = df[exec_sum_features].sum(axis=1, skipna=False)
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
        # 7天机构/ 15天机构
        df[feature_name] = np.round(df[feature_fenzi] / df[feature_fenmu], 3)
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
        # 7天机构/ 15天机构
        df[feature_name] = np.round(
            (df[feature_fenzi] - df[feature_fenmu]) / df[feature_fenmu], 3)
        self.update_feature_dict(feature_name, feature_desc)
        return df
