"""
模型报告输出
"""
import pandas as pd
import numpy as np
from typing import Union
import mdlutil
from model_tools.utils import plotutil


class ModelReport:
    def __int__(self, estimator, features: Union[list, np.array, pd.Series],
                feature_dict: dict,
                y: str,
                df_train, df_val, df_test,
                report_file):
        """
        :param estimator 学习器，必须有 predict_proba() 方法
        :param features:特征名
        :param feature_dict  特征字典 {'a':'申请次数'}
        :param y 目标变量名
        :param report_file :输出的报告，html 格式
        :return:
        """
        self.__estimator = estimator
        self.__features = features
        self.__feature_dict = feature_dict
        self.__label = y
        self.__train = df_train.copy()
        self.__val = None or df_val.copy()
        self.__test = df_test.copy()
        self.__report_file = report_file
        # 预测值
        self.__pred = 'y_pred'
        self.__train[self.__pred] = self.__estimator.predict_proba(self.__train[features])[:, 1]
        if self.__val is not None:
            self.__val[self.__pred] = self.__estimator.predict_proba(self.__val[features])[:, 1]
        self.__test[self.__pred] = self.__estimator.predict_proba(self.__test[features])[:, 1]

    def __stats_univar(self, df, feature_name, group_cols=[], n_bin=10, feature_grid=[]):
        """
        统计单个变量的 y_true,y_pred 的情况
        :param df: self.__train,self.__val,self.__test
        :param feature_name:
        :param group_col:
        :param n_bin:
        :param feature_grid:
        :return:
        """
        if df is None:
            return None
        gp_true = mdlutil.univar(df, feature_name, self.__label, feature_grid=feature_grid,
                                 n_bin=n_bin, cut_type=1, group_cols=group_cols)
        gp_pred = mdlutil.univar(df, feature_name, self.__pred, feature_grid=feature_grid,
                                 n_bin=n_bin, cut_type=1, group_cols=group_cols)
        if len(group_cols) == 0:
            cls_col = ['lbl', 'lbl_index', 'lbl_left']
        else:
            cls_col = group_cols + ['lbl', 'lbl_index', 'lbl_left']
        gp_true.rename(columns={'avg': 'rate_bad'}, inplace=True)
        gp_pred.rename(columns={'avg': 'score_avg'}, inplace=True)
        gp_true = gp_true.merge(gp_pred[cls_col + ['score_avg']], on=cls_col, how='left')
        return gp_true

    def __stats_univar_trte(self, feature_name, group_col, n_bin=10, plot_trte=True):
        """
        统计单个变量的 y_true,y_pred 的情况
        :param feature_name:
        :param group_col:
        :param n_bin:
        :param plot_trte:
        :return: sample_type,group_col,lbl,lbl_index,lbl_left,cnt,sum,rate_bad,score_avg
        """
        feature_grid = []
        if plot_trte:
            feature_grid = mdlutil.get_feature_grid(self.__train[feature_name], cut_type=1,
                                                    n_bin=n_bin)
        # 训练集
        gp_train = self.__stats_univar(self.__train, feature_name, [group_col], n_bin, feature_grid)
        gp_train['sample_type'] = 'train'
        # 验证集
        gp_val = self.__stats_univar(self.__val, feature_name, [group_col], n_bin, feature_grid)
        if gp_val is not None:
            gp_val['sample_type'] = 'val'
        # 测试集
        gp_test = self.__stats_univar(self.__test, feature_name, [group_col], n_bin, feature_grid)
        gp_test['sample_type'] = 'test'
        if plot_trte:
            if gp_val is not None:
                return pd.concat([gp_train, gp_val, gp_test], ignore_index=True)
            else:
                return pd.concat([gp_train, gp_test], ignore_index=True)
        else:
            return gp_test

    def __stats_liftvar(self, df, group_cols=[], n_bin=10, feature_grid=[]):
        """
        统计数据集 df 的lift 表现
        :param df:
        :param group_cols:
        :param n_bin:
        :param feature_grid:
        :return:
        """
        if df is None:
            return None, None, None
        gp = mdlutil.liftvar(df, self.__pred, self.__label, feature_grid=feature_grid, group_cols=group_cols)
        auc, ks, gini = mdlutil.evaluate_binary_classier(df[self.__label], df[self.__pred])
        return gp, auc, ks

    def report_liftchart(self, n_bin=10, plot_trte=True):
        """
        输出模型分的lift 表现
        :param n_bin:
        :param plot_trte:True,输出训练集，测试集、验证集的表现  False : 只看 test 的表现
        :return:
        """
        feature_grid = []
        if plot_trte:
            feature_grid = mdlutil.get_feature_grid(self.__train[self.__pred], cut_type=1, n_bin=n_bin)
        # train
        gp_train, auc, ks = self.__stats_liftvar(self.__train, group_cols=None, n_bin=n_bin, feature_grid=feature_grid)
        train_title = 'train:::cnt:{} auc:{} ks:{}'.format(len(self.__train), auc, ks)
        gp_train['group_col'] = train_title
        # val
        val_title = ''
        gp_val, auc, ks = self.__stats_liftvar(self.__val, group_cols=None, n_bin=n_bin, feature_grid=feature_grid)
        if gp_val:
            val_title = 'val:::cnt:{} auc:{} ks:{}'.format(len(self.__val),
                                                           auc, ks)
            gp_val['group_col'] = val_title
        # test
        gp_test, auc, ks = self.__stats_liftvar(self.__test, group_cols=None, n_bin=n_bin, feature_grid=feature_grid)
        test_title = 'test:::cnt:{} auc:{} ks:{}'.format(len(self.__test),
                                                         auc, ks)
        gp_test['group_col'] = test_title

        if gp_val:
            gp = pd.concat([gp_train, gp_val, gp_test], ignore_index=True)
        else:
            gp = pd.concat([gp_train, gp_test], ignore_index=True)

        fig = plotutil.plot_liftvars(gp, x1='lbl', y1='rate_bad', x2='lbl', y2='accum_lift_bad', group_col='group_col',
                                     is_show=False, title='liftchart')

        return fig


    def update_data(self,df_train,df_val,df_test):
        """
        更新数据集
        :param df_train:
        :param df_val:
        :param df_test:
        :return:
        """
        self.__train=df_train
        self.__val=df_val
        self.__test=df_test

    def report_feature(self, feature_name, group_col: str, n_bin=10, plot_trte=True):
        """
        分析单个变量::只看测试集的情况
        :param feature_name:
        :param group_col:当前只支持 单维度分组，后续增加多维度分组
        :param plot_trte : True 表示 plot train+test; False: 只plot test
        :return: go.Figure
        """
        gp = self.__stats_univar(feature_name, group_col, n_bin, plot_trte)
        fig = plotutil.plot_univar_and_pdp(gp, x=feature_name, y_true='rate_bad', y_pred='score_avg',
                                           group_col=group_col, is_show=False,
                                           title=self.__feature_dict[feature_name])
        return fig

    def report_features(self):
        """
        保存所有的特征 univar + pdp 到 html 中
        保存方式是 dash app 运行
        :param plot_trte:
        :return:
        """
        figs = []
        for feature_name in self.__features:
            fig = self.report_feature(feature_name, None, n_bin=10, plot_trte=True)
            figs.append(fig)
        plotutil.show_dash(figs)
