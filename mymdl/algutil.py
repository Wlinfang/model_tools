import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import tree
from xgboost import DMatrix, cv, XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
import pydotplus
import graphviz
import shap


def build_tree(df, feature_names, target, max_depth=3):
    """
    构建决策回归树
    target:二分类
    返回：dtr graph ftr_imp
    dtr:决策树；graph:决策树图；ftr_imp:特征重要性
    """
    dtr = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_split=200, random_state=1024)
    dtr.fit(df[feature_names], df[target])
    dot_data = tree.export_graphviz(dtr, out_file=None, feature_names=feature_names,
                                    class_names=target, filled=True, impurity=False, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    #     graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    graph = graphviz.Source(dot_data)
    #     graph.write_pdf("tree.pdf")
    ftr_imp = pd.Series(dtr.feature_importances_, feature_names)
    return dtr, graph, ftr_imp


def xgboost_fit(df_train, df_val, feature_cols, target, cv_folds=5, max_depth=3,
                learning_rate=0.05, n_estimators=500):
    """
    利用cv 获取最优 n_estimators
    feature_cols：输入的X 特征变量名
    resp：目标变量名
    """
    # 利用cv 进行调参
    params = {
        'n_estimators': n_estimators,  # 调参
        'max_depth': max_depth,
        'grow_policy': 'lossguide',  # leaf-wise 生长策略
        'learning_rate': learning_rate,
        'verbosity': 1,  # 日志输出
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'gamma': 0.0001,
        #     'tree_method':'hist',
        #     'reg_alpha':,
        #     'reg_lambda':,
        'random_state': 1024,
        'eval_metric': 'auc'
    }
    alg = XGBClassifier(**params)
    xgb_param = alg.get_xgb_params()
    xgtrain = DMatrix(df_train[feature_cols].values, label=df_train[target].values)

    cvresult = cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'],
                  nfold=cv_folds, metrics='auc',
                  early_stopping_rounds=10, verbose_eval=100, seed=1024, shuffle=True)
    # 更新参数
    alg.set_params(n_estimators=cvresult.shape[0])
    alg.set_params(eval_metric='auc')
    print(cvresult, cvresult.shape)

    #     eval_set=[(df_val[feature_cols],df_val[resp])]
    #     evallist = [(dtrain, 'train'), (dtest, 'eval')]

    # Fit the algorithm on the data and save the model
    alg.fit(df_train[feature_cols], df_train[target])
    print('Model params: -----------')
    print(alg.n_estimators, alg.max_depth, alg.learning_rate)

    print("\nModel Report")
    dtrain_predprob = alg.predict_proba(df_train[feature_cols])[:, 1]
    print("AUC Score (Train): %f" % metrics.roc_auc_score(df_train[target], dtrain_predprob))
    if df_val is not None:
        # Predict validation Set:
        dval_predprob = alg.predict_proba(df_val[feature_cols])[:, 1]
        print("AUC Score (Validation): %f" % metrics.roc_auc_score(df_val[target], dval_predprob))

    # Print Feature Importance:
    feat_imp = pd.Series(alg.get_booster().get_fscore(), feature_cols).sort_values(ascending=False, na_position='last')
    feat_imp = feat_imp[feat_imp > 0]

    return alg, feat_imp


def logit_fit(df_train, df_val, feature_cols, target,
              penalty='l2', Cs=10, fit_intercept=True, cv=5):
    """
    sklearn Logit regression
    :param feature_cols:
    :param target:
    :param penalty 正则化项， ‘l1’, ‘l2’, None
    :param fit_intercept : 截距；如果X 进行了中心化处理，设置为 False
    :param Cs : 值越小 正则化越强; 如果为int , list；cv 用于调参
    :return:
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1024)
    lr = LogisticRegressionCV(Cs=Cs, fit_intercept=fit_intercept, cv=kf,
                              penalty=penalty, scoring='auc', max_iter=500,
                              solver='saga', tol=1e-4, class_weight='balanced', refit=True,
                              random_state=1024, verbose=1)
    lr.fit(df_train[feature_cols], target)
    # 参数 coef_ c_

    return lr



def cal_shap_value(alg, df, feature_names, top_num=5, is_show=False):
    """
    计算shap value 值 主要用于分析贡献度
    alg:estimators 算法对象 限制为树类算法
    features: 入模特征
    top_num : 默认提取前5个特征
    is_show : 是否画图shap value False : 不画图
    """
    explainer = shap.TreeExplainer(alg)
    shap_values = explainer.shap_values(df[feature_names])

    f = pd.DataFrame(shap_values, columns=feature_names)
    f = np.abs(f).mean()
    ftr_imp = pd.Series(alg.get_booster().get_fscore(), feature_names).sort_values(ascending=False,
                                                                                   na_position='last').head(
        top_num)
    f = f[f > 0.000].sort_values(ascending=False).reset_index().rename(columns={'index': 'f', 0: 'value'})
    f = f[f['f'].isin(ftr_imp.index.values)]

    if is_show:
        shap.summary_plot(shap_values, df[feature_names], plot_type="bar")
    return f
