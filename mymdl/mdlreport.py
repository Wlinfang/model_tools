"""
模型报告输出
"""
import pandas as pd
from model_tools.mymdl import mdlutil
from model_tools.utils import plotutil


class ModelReport:
    """
    适用于模型报告输出
    """

    def __init__(self, feature_dict={}, report_file=''):
        """
        :param feature_dict  特征字典 {'a':'申请次数'}
        :param report_file :输出的报告，html 格式
        """
        self.__feature_dict = feature_dict
        self.__report_file = report_file

    def report_liftchart(self, df_train, df_test, x, y, n_bin=10,
                         group_cols=[],
                         plot_trte=True,
                         is_show=False, is_save=False):
        """
        输出模型分的lift 表现;df_test 必须有值；
        :param group_cols 分组查看
        :param plot_trte:True,输出训练集、验证集[可选]的表现  False : 只看 test 的表现
        返回 fig,gp
        """
        if group_cols is None:
            group_cols = []
        cols = [x, y, 'sample_type'] + group_cols
        if df_test is None:
            raise ValueError('df_test is None！！！')
            return None
        if plot_trte and df_train is None:
            raise ValueError('df_train is None！！！')
            return None
        df_test['sample_type'] = 'test'
        if plot_trte:
            # 训练集分组
            feature_grid = mdlutil.get_feature_grid(df_train[x], cut_type=1, n_bin=n_bin)
            df_train['sample_type'] = 'train'
            t = pd.concat([df_train[cols], df_test[cols]])
        else:
            feature_grid = mdlutil.get_feature_grid(df_test[x], cut_type=1, n_bin=n_bin)
            t = df_test[cols]
        fig, gp = plotutil.plot_liftvar(t,  y,x, group_cols+['sample_type'], feature_grid=feature_grid, is_show=is_show)
        if is_save:
            plotutil.save_fig_tohtml(self.__report_file, fig)
        return fig, gp

    def report_feature(self, df_train, df_test, feature_name, y_true, y_pred, group_cols=[], n_bin=10, plot_trte=True,
                       is_show=False, is_save=False) -> pd.DataFrame:
        """
        分析单个变量::测试集的情况; feature_name & [y_true,y_pred]
        :param plot_trte : True 表示 plot train+test; False: 只plot test
        :return: dataframe
        """
        if group_cols is None:
            group_cols = []
        cols = [feature_name, y_true, y_pred, 'sample_type'] + group_cols
        if plot_trte and df_train is None:
            return None
        if df_test is None:
            return None
        df_test['sample_type'] = 'test'
        if plot_trte:
            feature_grid = mdlutil.get_feature_grid(df_train[feature_name], cut_type=1, n_bin=n_bin)
            df_train['sample_type'] = 'train'
            t = pd.concat([df_train[cols], df_test[cols]])
        else:
            feature_grid = mdlutil.get_feature_grid(df_test[feature_name], cut_type=1, n_bin=n_bin)
            t = df_test[cols]
        feature_desc = self.__feature_dict.get(feature_name, feature_name)
        fig, gp = plotutil.plot_univar_and_pdp(t, x=feature_name, y_true=y_true, y_pred=y_pred,
                                               group_cols=group_cols+['sample_type'],
                                               feature_grid=feature_grid,
                                               title=feature_desc, is_show=is_show)
        if is_save:
            plotutil.save_fig_tohtml(self.__report_file, fig)
        return gp

    def report_features(self, df_train, df_test, features, y_true, y_pred, plot_trte=True, group_cols=[], n_bin=10):
        """
        保存所有的特征 univar + pdp 到 html 中
        """
        for feature_name in features:
            print(feature_name)
            self.report_feature(df_train, df_test, feature_name, y_true, y_pred, group_cols=group_cols,
                                n_bin=n_bin,
                                plot_trte=plot_trte,
                                is_show=False, is_save=True)
