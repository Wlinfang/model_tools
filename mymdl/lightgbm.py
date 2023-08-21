import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
import numpy 
import pandas

params = {
    'task': 'train',   #用途
    'application':'binary',   #用于二分类
    'boosting_type': 'gbdt',  # 设置提升类型
    'num_boost_round':100,   #迭代次数
    'learning_rate': 0.01,  # 学习速率
    'metric': {'logloss', 'auc'},  # 评估函数
    'early_stopping_rounds':None,
#         'objective': 'regression', # 目标函数
    'max_depth':4,
    'num_leaves': 20,   # 叶子节点数   
    'feature_fraction': 0.9, # 建树的特征选择比例
    'bagging_fraction': 0.8, # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    
    'verbose': 1 # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }


'''
instructions : training lightgbm model with specified params

Parameters : 
    dataset -
    features - feature list of dataset
    target - tagert column or label list of samples

'''
def lgb_train(params,training_set,features,target):
    lgb_train = lgb.Dataset(training_set[features],training_set[target])
    #lgb.train(params,)
    return 1


'''
instructions : build a lgb classifier

Params : 
    
'''   
def buildClf(params):
    return lgb.LGBMClassifier(params)

'''
'''
def automodelfit(clf,param_grid,dftrain,features,resp, kfold=10,scoring='roc_auc'):

    # kflod=StratifiedKFold(n_splits=kfold,shuffle=True,random_state=7)
    grid_search=GridSearchCV(clf,param_grid,scoring=scoring,n_jobs=2,cv=kfold,verbose=2,iid=True,refit=True)
    #== 模型训练
    grid_search.fit(dftrain[features],dftrain[resp])
    #== 获取最优参数
    return grid_search
    

def modelfit(clf, dftrain, features, resp,useTrainCV = True, kfold=10, eval_metric='auc',early_stopping_rounds=20):
    '''
    模型训练
    :type useTrainCV: object
    :param clf:XGBClassifier
    :param dftrain:训练集
    :param features: 特征
    :param resp:label
    :param useTrainCV:if True  call cv function,目的是调节参数 n_estimators
    :param cv_folds: N 折交叉验证
    :param early_stopping_rounds:添加数loss变化不大这个状态持续的轮数，达到这个数就退出训练过程
    :param eval_metric 同 目标函数 objective 有关，取值https://xgboost.readthedocs.io/en/latest/python/python_api.html#
    :return:
    '''
    if useTrainCV:
        # kflod = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=7)
        xgb_param = clf.get_xgb_params()
        xgtrain = lgb.DMatrix(dftrain[features].values, label=dftrain[resp].values)
        cvresult = lgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=kfold,
            metrics=eval_metric, early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
        clf.set_params(n_estimators=cvresult.shape[0])

    clf.fit(dftrain[features], dftrain[resp],eval_metric=eval_metric)
    return clf

