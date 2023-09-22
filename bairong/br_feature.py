"""
百融特征工程
"""
from model_tools.mymdl.feature_engineer import FeatureEngineer
from model_tools.mymdl import mdlutil

class ApplyLoanStr:
    """
    借贷意向验证接口
    """
    def __init__(self,feature_cols,feature_dict):
        self.feature_cols=feature_cols
        self.feature_dict=feature_dict
    def preprocess(self,df):
        common_cols = list(set(self.feature_cols) & set(df.columns))
        # 预过滤一部分特征
        df_feature_describe = mdlutil.describe_df(df, common_cols)
        df_feature_describe['miss_rate_float'] = df_feature_describe['miss_rate'].str.replace('%', '')
        df_feature_describe['miss_rate_float'] = df_feature_describe['miss_rate_float'].astype(float)


        mdlutil.describe_df(df,common_cols)
    def add_features(self,df):
        """
        :param df:
        :return:
        """
