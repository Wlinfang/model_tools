import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sn
from mpl_toolkits.axes_grid1 import host_subplot
import lightgbm as lgb
# 刻度方向配置
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'inout'
# 字体配置--支持中文
plt.rcParams['font.sans-serif']=['SimHei','YouYuan']
# 显示负号
plt.rcParams['axes.unicode_minus']=False # 负数
# padding bbox
plt.rcParams['figure.autolayout']=True

from sklearn.metrics import *
import xgboost as xgb

from sklearn import tree
# 决策树
import pydotplus
import graphviz

import shap
shap.initjs()

def buildTree(df,feature,target,max_depth=3):
    '''
    构建决策树
	# 170 jupyter 上有权限问题，graph 无法显示图像
	返回：dtr graph ftr_imp
	dtr:决策树；graph:决策树图；ftr_imp:特征重要性
    '''
    dtr = tree.DecisionTreeRegressor(max_depth=max_depth,min_samples_split=200,random_state=1024)
    dtr.fit(df[feature],df[target])
    dot_data = tree.export_graphviz(dtr,out_file=None,feature_names=feature,class_names=target,
                                   filled=True,impurity=False,rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    #	 graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    graph = graphviz.Source(dot_data)
    #	 graph.write_pdf("tree.pdf")
    ftr_imp=pd.Series(dtr.feature_importances_,feature)
    return dtr,graph,ftr_imp


class FindAbnormalFeature:
    '''
    改算法的核心思想是：利用机器学习算法得到特征的重要性，提取特征变化大的特征
    '''
    def __init__(self,df_train,df_test,features,label):
        '''
        Parameters
        features:list；入模的特征
        label：boolean ; 标签
        '''
        self.df_train=df_train
        self.df_test = df_test
        self.features=features
        self.label=label
        # categorical_feature lgb 中可以直接进行计算，无需one-hot 识别 categorical_feature 规则为 取值个个数比较少
        tmp=df_train[features].nunique(axis=0).to_frame().reset_index().rename(
            columns={0: 'unique', 'index': 'feature_name'})
        self.categorical_feature=tmp[tmp['unique']<10].feature_name.unique().tolist()
        # 特征数字型转换
        num_list = df_train[features].select_dtypes(include=np.number).columns.tolist()
        non_num_list=list(set(features)-set(num_list))
        if len(non_num_list) > 0:
            for i in non_num_list:
                self.df_train[i]=pd.to_numeric(self.df_train[i],errors='ignore')

    def lgb_model(self,):
        train_data=lgb.Dataset(self.df_train,self.label,feature_name=self.features,categorical_feature=self.categorical_feature)


    def xgb_model(self,nfold=5,max_depth=2,learning_rate=0.05,n_estimators=200):
        '''
        	算法为xgboost 这个未作任何的调参，仅仅是分析使用，无需达到模型的严格要求
        	返回clf 算法对象, feat_imp 特征重要性
        	'''
        param = {
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'verbosity': 2,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'n_jobs': -1,
            'gamma': 0.0001,
            'random_state': 1024,
            'early_stopping_rounds': 20
        }
        X_train = self.df_test[self.features]
        Y_label = self.df_test[[self.label]]
        d_train = xgb.DMatrix(X_train, label=Y_label)
        clf = xgb.XGBClassifier(**param, eval_metric='auc', n_estimators=n_estimators)
        #	 cvres=xgb.cv(clf.get_xgb_params(), d_train, num_boost_round = n_estimators, nfold =nfold,metrics= (['auc']), early_stopping_rounds = 20,seed=1024,)
        #	 print (cvres, cvres.shape)

        #	 clf.set_params(n_estimators=cvres.shape[0])
        print('Model params: -----------')
        print(clf.n_estimators)
        print(clf.get_xgb_params())
        clf.fit(X_train, Y_label, eval_metric='auc',)
        # Predict training set:
        dtrain_predictions = clf.predict(df[features])
        dtrain_predprob = clf.predict_proba(df[features])[:, 1]









def xgb_modelfit(df,features,label,nfold=5,max_depth=2,learning_rate=0.05,n_estimators=200):
	'''
	算法为xgboost 这个未作任何的调参，仅仅是分析使用，无需达到模型的严格要求
	返回clf 算法对象, feat_imp 特征重要性
	'''



	# Print Model Report:
	print ("\nModel Report")
	print ("Accuracy : %.4g" % accuracy_score(df[label].values, dtrain_predictions))
	print ("AUC Score (Train): %f" % roc_auc_score(df[label], dtrain_predprob))
	# Print Feature Importance:
	feat_imp = pd.Series(clf.get_booster().get_fscore(), features).sort_values(ascending=False, na_position='last')
	# feat_imp = pd.Series(alg.booster().get_fscore(), predictors).sort(ascending=False)
	feat_imp = feat_imp[feat_imp > 0]
	print ('----------- Feature importance -------------')
	print (feat_imp)

	return clf, feat_imp

def cal_shap_value(clf,df,features,top_num=5,is_show=False):
	'''
	计算shap value 值
	clf:estimators 算法对象
	features: 入模特征
	top_num : 默认提取前5个特征
	is_show : 是否画图shap value False : 不画图
	'''
	# # 可视化第一个样本预测的解释
	# shap.force_plot(explainer.expected_value,
	#				   shap_values[0,:],
	#				   X.iloc[0,:], matplotlib=True)

	# # 所有样本Shap图
	# shap.force_plot(explainer.expected_value, shap_values, X)

	# # 计算所有特征的影响
	# shap.summary_plot(shap_values, X)

	explainer = shap.TreeExplainer(clf)
	shap_values = explainer.shap_values(df[features])

	f=pd.DataFrame(shap_values,columns=features)
	f=np.abs(f).mean()
	ftr_imp=pd.Series(clf.get_booster().get_fscore(), features).sort_values(ascending=False, na_position='last').head(top_num)
	f=f[f>0.000].sort_values(ascending=False).reset_index().rename(columns={'index':'f',0:'value'})
	f=f[f['f'].isin(ftr_imp.index.values)]

	if is_show == True:
		shap.summary_plot(shap_values, df[features], plot_type="bar")
	return f


def get_top_feature(df_base,df_test,features,label):

