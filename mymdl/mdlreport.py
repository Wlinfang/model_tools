"""
模型报告输出
"""
import pandas as pd
import numpy as np
from typing import Union
import mdlutil
from model_tools.utils import plotutil

class ModelReport:
    def __int__(self, estimator,features: Union[list, np.array, pd.Series],y:str,
                df_train,df_val,df_test,
                report_file):
        """
        :param estimator 学习器，必须有 predict_proba() 方法
        :param features:特征名
        :param y 目标变量名
        :param report_file :输出的报告，html 格式
        :return:
        """
        self.__estimator = estimator
        self.__features = features
        self.__label = y
        self.__train=df_train.copy()
        self.__val=df_val.copy()
        self.__test = df_test.copy()
        self.__report_file = report_file

    def report_liftchart(self,n_bin=10):
        """
        整体看 训练集、验证集、测试集的lift 值,等频分组
        :param df_train:
        :param df_val:
        :param df_test:
        :param n_bin:
        :return:
        """
        # 预测值
        self.__train['y_pred']=self.__estimator.predict_proba(self.__train[self.__features])[:, 1]
        feature_grid = mdlutil.get_feature_grid(self.__train['y_pred'],cut_type=1,n_bin=n_bin)
        # 分组
        gp_train=mdlutil.liftvar(self.__train,'y_pred',self.__label,feature_grid=feature_grid)
        auc, ks, gini = mdlutil.evaluate_binary_classier(self.__train[self.__label], self.__train['y_pred'])
        train_title = 'train:::cnt:{} auc:{} ks:{}'.format(len(self.__train['y_pred']),
                                                                   auc,ks)
        # 初始化
        gp_val=pd.DataFrame()
        gp_test=pd.DataFrame()

        val_title=''
        test_title=''
        if self.__val:
            self.__val['y_pred']=self.__estimator.predict_proba(self.__val[self.__features])[:, 1]
            gp_val = mdlutil.liftvar(self.__val, 'y_pred', self.__label, feature_grid=feature_grid)

            auc, ks, gini=mdlutil.evaluate_binary_classier(self.__val[self.__label],self.__val['y_pred'])

            val_title = 'val:::cnt:{} auc:{} ks:{}'.format(len(self.__val['y_pred']),
                                                                   auc,ks)
        if self.__test:
            self.__test['y_pred'] = self.__estimator.predict_proba(self.__test[self.__features])[:, 1]
            gp_test = mdlutil.liftvar(self.__test, 'y_pred', self.__label, feature_grid=feature_grid)
            auc, ks, gini = mdlutil.evaluate_binary_classier(self.__test[self.__label], self.__test['y_pred'])
            test_title = 'test:::cnt:{} auc:{} ks:{}'.format(len(self.__test['y_pred']),
                                                                   auc,ks)

        fig = plotutil.plot_liftvars([
            gp_train['lbl'],gp_val['lbl'],gp_test['lbl']
        ],[gp_train['rate_bad'],gp_val['rate_bad'],gp_test['rate_bad']],
        [gp_train['lbl'],gp_val['lbl'],gp_test['lbl']],
        [gp_train['accum_lift_bad'],gp_val['accum_lift_bad'],gp_test['accum_lift_bad']],
        title='lift',fig_titles=[train_title,val_title,test_title],is_show=False)
        return fig




    def report_feature(self,feature_name,is_save=False,plot_trte=True):
        """
        分析单个变量::只看测试集的情况
        :param feature_name:
        :param is_save:True  表示 保存到文件 report_file
        :param plot_trte : True 表示 plot train+test; False: 只plot test
        :return: go.Figure
        """
        plot_univar_with_pdp



    def report_features(self,df_test,df_):


