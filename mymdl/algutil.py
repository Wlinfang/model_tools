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


def split_train_val(df: pd.DataFrame, split_type=1, split_ratio=0.8, sort_col=None):
    """
    将数据集进行切分
    :param split_type  1：随机切分  2：按照时间切分
    :param split_ratio  切分比例； 取值范围：(0,1)
    :param sort_col：如果 split_type=2 根据sort_col 排序然后切分
    :return: df_train,df_val
    """
    dftrain = df.reset_index()
    # == dftrain 中划分 训练集，验证集
    if split_type == 1:
        # 随机分配 train / val
        train = dftrain.sample(frac=split_ratio, random_state=7)
        val = dftrain[~dftrain.index.isin(train.index)]
    elif split_type == 2:
        # 按时间序列分配 train /val
        train = dftrain.sort_values(by=sort_col).head(int(len(dftrain) * split_ratio))
        val = dftrain[~dftrain.index.isin(train.index)]
    else:
        raise ValueError('param error or data error')
    return train, val


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
        'booster': 'gbtree',
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'grow_policy': 'lossguide',  # leaf-wise 生长策略
        'max_leaves': 10,
        'learning_rate': learning_rate,  # [0,1] 0.01~0.2
        'verbosity': 1,  # 日志输出
        'objective': 'binary:logistic',
        'gamma': 0.0001,
        'tree_method': 'hist',
        'min_child_weight': 1,  # cv
        # 'reg_alpha':,# l1
        # 'reg_lambda': 0.1,  # l2
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

    # Print Feature Importance: 按照平均增益方式
    # feat_imp = pd.Series(alg.get_booster().get_fscore(), feature_cols).sort_values(ascending=False, na_position='last')
    # feat_imp = feat_imp[feat_imp > 0]
    feat_imp = pd.Series(alg.get_booster().get_score(importance_type='gain'), feature_cols).sort_values(ascending=False,
                                                                                                        na_position='last')
    feat_imp = feat_imp[feat_imp.notna()]

    return alg, feat_imp


def logit_fit(df_train, df_val, feature_cols, target,
              penalty='l2', Cs=10, fit_intercept=True, cv=5):
    """
    sklearn Logit regression,本code 只在二分类上测试，如果多分类，待验证
    :param feature_cols:
    :param target:
    :param penalty 正则化项， ‘l1’, ‘l2’, None
    :param fit_intercept : 截距；如果X 进行了中心化处理，设置为 False
    :param Cs : 值越小 正则化越强; 如果为int 1e-4~1e4等比10个数字, list；cv 用于调参
    :return:
    """
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1024)
    lr = LogisticRegressionCV(Cs=Cs, fit_intercept=fit_intercept, cv=kf,
                              penalty=penalty, scoring='roc_auc', max_iter=100,
                              solver='saga', tol=1e-4, class_weight='balanced', refit=True,
                              random_state=1024, verbose=0)
    lr.fit(df_train[feature_cols], df_train[target])
    # 训练集auc,验证集auc
    train_proba = lr.predict_proba(df_train[feature_cols])[:, 1]
    print('AUC Score (Train): %f' % metrics.roc_auc_score(df_train[target], train_proba))
    if df_val is not None:
        val_proba = lr.predict_proba(df_val[feature_cols])[:, 1]
        print('AUC Score (Validation): %f' % metrics.roc_auc_score(df_val[target], val_proba))
    # c_
    df_c = pd.DataFrame(lr.Cs_, columns=['C'])
    df_c = df_c.reset_index()
    df_c = df_c.rename(columns={'index': 'c_name'})
    df_c['c_name'] = df_c['c_name'].apply(lambda x: 'c' + str(x))
    # 最优参数
    data = []
    for ky in lr.scores_.keys():
        t = lr.scores_.get(ky)
        if t.ndim == 2:
            n_folds, n_cs = t.shape
            index_names = ['c%d' % i for i in range(0, n_cs, 1)]
            td = pd.DataFrame(np.mean(t, axis=0), columns=['score'], index=index_names)
            td = td.reset_index()
            td.rename(columns={'index': 'c_name'}, inplace=True)
            td['classes'] = ky
            data.append(td)
    df_scores = pd.concat(data)
    df_scores = df_c.merge(df_scores, on='c_name', how='right')
    print(df_scores)
    print(' best params \n')
    print(' C::', lr.C_, ' intercept::', lr.intercept_, )
    # 特征 & 系数
    df_feat = pd.DataFrame(lr.coef_[0], index=lr.feature_names_in_, columns=['coef'])
    return lr, df_feat


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
