from xgboost import DMatrix, cv
from xgboost import XGBClassifier
import pandas as pd
from sklearn import metrics


def xgboost_fit(df_train, df_val, feature_cols, resp, cv_folds=5, max_depth=3,
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
    xgtrain = DMatrix(df_train[feature_cols].values, label=df_train[resp].values)

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
    alg.fit(df_train[feature_cols], df_train[resp])
    print('Model params: -----------')
    print(alg.n_estimators, alg.max_depth, alg.learning_rate)

    print("\nModel Report")
    dtrain_predprob = alg.predict_proba(df_train[feature_cols])[:, 1]
    print("AUC Score (Train): %f" % metrics.roc_auc_score(df_train[resp], dtrain_predprob))
    if df_val is not None:
        # Predict validation Set:
        dval_predprob = alg.predict_proba(df_val[feature_cols])[:, 1]
        print("AUC Score (Validation): %f" % metrics.roc_auc_score(df_val[resp], dval_predprob))

    # Print Feature Importance:
    feat_imp = pd.Series(alg.get_booster().get_fscore(), feature_cols).sort_values(ascending=False, na_position='last')
    feat_imp = feat_imp[feat_imp > 0]

    return alg, feat_imp
