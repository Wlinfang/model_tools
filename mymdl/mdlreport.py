"""
模型报告输出
"""
import pandas as pd
import numpy as np
from typing import Union
from model_tools.mymdl import mdlutil
from model_tools.utils import plotutil


class ModelReport:
    """
    适用于模型报告输出
    """

    def __init__(self, estimator, features: Union[list, np.array, pd.Series],
                 feature_dict: dict, y: str, report_file):
        """
        :param estimator 学习器，必须有 predict_proba() 方法
        :param features:入模的特征名 Union[list, np.array, pd.Series]
        :param feature_dict  特征字典 {'a':'申请次数'}
        :param y 目标变量名
        :param report_file :输出的报告，html 格式
        :return:
        """
        self.__estimator = estimator
        self.__features = features
        self.__feature_dict = feature_dict
        self.__label = y
        self.__report_file = report_file
        self.__pred = 'y_pred'

    def __predict_proba(self, df):
        """
        预测值
        :param df:
        :return:
        """
        if df is None:
            return None
        df[self.__pred] = self.__estimator.predict_proba(df[self.__features])[:, 1]
        return df

    def __stats_univar(self, df, feature_name, group_cols=[], n_bin=10, feature_grid=[]):
        """
        统计单个变量的 y_true,y_pred 的情况
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
        if group_cols is None or len(group_cols) == 0:
            cls_col = ['lbl', 'lbl_index', 'lbl_left']
        else:
            cls_col = group_cols + ['lbl', 'lbl_index', 'lbl_left']
        gp_true.rename(columns={'avg': 'rate_bad'}, inplace=True)
        gp_pred.rename(columns={'avg': 'score_avg'}, inplace=True)
        gp_true = gp_true.merge(gp_pred[cls_col + ['score_avg']], on=cls_col, how='left')
        return gp_true

    def __stats_liftvar(self, df, group_cols=[], n_bin=10, feature_grid=[]):
        """
        统计数据集 df 的lift 表现
        :param df:
        :return: df_lift, df_auc
        """
        if df is None:
            return None, None
        gp = mdlutil.liftvar(df, self.__pred, self.__label, feature_grid=feature_grid, cut_type=1, n_bin=n_bin,
                             group_cols=group_cols)
        gp = gp.reset_index()
        # 分组计算 auc,ks,gini
        if group_cols is not None and len(group_cols) > 0:
            gp_auc = df.groupby(group_cols).apply(
                lambda x: mdlutil.evaluate_binary_classier(x[self.__label], x[self.__pred], is_show=False))
            gp_auc = gp_auc.reset_index().rename(columns={0: 'value'})
            gp_auc.loc[:, ['cnt', 'auc', 'ks', 'gini']] = gp_auc['value'].apply(pd.Series,
                                                                                index=['cnt', 'auc', 'ks', 'gini'])
            gp_auc.drop(['value'], axis=1, inplace=True)
            return gp, gp_auc
        else:
            cnt, auc, ks, gini = mdlutil.evaluate_binary_classier(df[self.__label], df[self.__pred])
            gp_auc = pd.DataFrame([[cnt, auc, ks, gini]], columns=['cnt', 'auc', 'ks', 'gini'], index=['all'])
        return gp, gp_auc

    def report_liftchart(self, df_train, df_test, n_bin=10,
                         group_cols=[],
                         plot_trte=True,
                         is_show=False, is_save=False) -> pd.DataFrame:
        """
        输出模型分的lift 表现;df_test 必须有值；
        :param group_cols 分组查看
        :param plot_trte:True,输出训练集、验证集[可选]的表现  False : 只看 test 的表现
        :return:
        """
        if df_test is None:
            raise ValueError('df_test is None！！！')
            return None
        if plot_trte and df_train is None:
            raise ValueError('df_train is None！！！')
            return None
        # 模型分预测
        for df in [df_train, df_test]:
            df = self.__predict_proba(df)
        feature_grid = []
        if plot_trte:
            # 训练集分组
            feature_grid = mdlutil.get_feature_grid(df_train[self.__pred], cut_type=1, n_bin=n_bin)
        # 初始化输出值
        gp = pd.DataFrame()
        gp_auc = pd.DataFrame()
        # train test
        for df, sample_type in zip([df_train, df_test], ['train', 'test']):
            tmp, tmp_auc = self.__stats_liftvar(df, group_cols=group_cols, n_bin=n_bin,
                                                feature_grid=feature_grid)
            if tmp is not None and len(tmp) > 0:
                tmp['sample_type'] = sample_type

                tmp_auc['auc_title'] = tmp_auc.apply(
                    lambda x: 'cnt:{} auc:{} ks:{}'.format(x['cnt'], x['auc'], x['ks']), axis=1)
                tmp_auc['sample_type'] = sample_type
                gp = pd.concat([gp, tmp])
                gp_auc = pd.concat([gp_auc, tmp_auc])
        # 如果分组的情况
        if group_cols is not None and len(group_cols) > 0:
            gp[group_cols] = gp[group_cols].astype(str)
            gp['group_cols_str'] = gp[group_cols].apply(lambda x: ':'.join(x), axis=1)
            gp_auc[group_cols] = gp_auc[group_cols].astype(str)
            gp_auc['group_cols_str'] = gp_auc[group_cols].apply(lambda x: ':'.join(x), axis=1)
        else:
            gp['group_cols_str'] = ''
            gp_auc['group_cols_str'] = ''
        # 合并 gp,gp_auc
        gp = gp.merge(gp_auc[['group_cols_str', 'sample_type', 'auc_title']], on=['group_cols_str', 'sample_type'],
                      how='left')
        # 生成图例
        gp['legend_title'] = gp[['sample_type', 'group_cols_str', 'auc_title']].fillna('').apply(lambda x: '::'.join(x),
                                                                                                 axis=1)
        n = gp['group_cols_str'].nunique()
        m = gp['sample_type'].nunique()
        if m > 1 and n > 1:
            # 多数据集+多维组合
            for gcs in gp['group_cols_str'].unique():
                fig = plotutil.plot_liftvar(gp[gp['group_cols_str'] == gcs], x1='lbl', y1='rate_bad', x2='lbl',
                                            y2='accum_lift_bad',
                                            title=gcs, group_col='legend_title',
                                            is_show=is_show)
                if is_save:
                    plotutil.save_fig_tohtml(self.__report_file, fig)
        else:
            fig = plotutil.plot_liftvar(gp, x1='lbl', y1='rate_bad', x2='lbl', y2='accum_lift_bad',
                                        group_col='legend_title',
                                        title='liftchart',
                                        is_show=is_show)
            plotutil.save_fig_tohtml(self.__report_file, fig)

        return gp

    def report_feature(self, df_train, df_test, feature_name, group_cols=[], n_bin=10, plot_trte=True,
                       is_show=False, is_save=False):
        """
        分析单个变量::测试集的情况; feature_name & [y_true,y_pred]
        :param feature_name:
        :param group_cols:
        :param plot_trte : True 表示 plot train+test; False: 只plot test
        :return: go.Figure
        """
        if plot_trte and df_train is None:
            return None
        if df_test is None:
            return None

        feature_name_desc = self.__feature_dict[feature_name]
        # 初始化输出值
        gp = pd.DataFrame()
        feature_grid = []
        if plot_trte:
            feature_grid = mdlutil.get_feature_grid(df_train[feature_name], cut_type=1, n_bin=n_bin)
        for df, sample_type in zip([df_train, df_test], ['train', 'test']):
            if df is not None:
                # 预测分值
                df = self.__predict_proba(df)
                # 统计数据
                tmp = self.__stats_univar(df, feature_name, group_cols, n_bin, feature_grid)
                tmp['sample_type'] = sample_type
                gp = pd.concat([gp, tmp])
        # group_cols 汇总为1个字段
        if group_cols is not None and len(group_cols) > 0:
            gp[group_cols] = gp[group_cols].astype(str)
            gp['group_cols_str'] = gp[group_cols].apply(lambda x: ':'.join(x), axis=1)
        else:
            gp['group_cols_str'] = ''

        # 图例标题
        gp['legend_title'] = gp[['sample_type', 'group_cols_str']].apply(lambda x: '::'.join(x), axis=1)

        if plot_trte and gp['group_cols_str'].nunique() > 1:
            # 多数据集+多维
            for gcs in gp['group_cols_str'].unique():
                tmp = gp[gp['group_cols_str'] == gcs]
                # univar
                fig = plotutil.plot_univar_with_bar(tmp, x='lbl', y_line='rate_bad', y_bar='cnt',
                                                    group_col='legend_title',
                                                    title='::'.join([feature_name, feature_name_desc]),
                                                    is_show=is_show)
                if is_save:
                    plotutil.save_fig_tohtml(self.__report_file,fig)
                # pdp
                fig = plotutil.plot_univar(tmp, x='lbl', y='score_avg', group_col='legend_title',
                                           title='pdp-{}'.format(feature_name),
                                           is_show=is_show)
                if is_save:
                    plotutil.save_fig_tohtml(self.__report_file,fig)
        else:
            # univar
            fig = plotutil.plot_univar_with_bar(gp, x='lbl', y_line='rate_bad', y_bar='cnt',
                                                group_col='legend_title',
                                                title='::'.join([feature_name, feature_name_desc]),
                                                is_show=is_show)
            if is_save:
                plotutil.save_fig_tohtml(self.__report_file, fig)
            # pdp
            fig = plotutil.plot_univar(gp, x='lbl', y='score_avg', group_col='legend_title',
                                       title='pdp-{}'.format(feature_name),
                                       is_show=is_show)
            if is_save:
                plotutil.save_fig_tohtml(self.__report_file, fig)

        return gp

    def report_features(self,df_train,df_test,plot_trte=True,group_cols=[], n_bin=10):
        """
        保存所有的特征 univar + pdp 到 html 中
        :param plot_trte:
        :return:
        """
        figs = []
        for feature_name in self.__features:
            fig = self.report_feature(df_train, df_test, feature_name, group_cols=group_cols, n_bin=n_bin, plot_trte=plot_trte,
                       is_show=False, is_save=True)
            figs.append(fig)
        plotutil.show_dash(figs)
