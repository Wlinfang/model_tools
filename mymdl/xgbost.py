

def xgb_modelfit(df,features,label,nfold=5,max_depth=2,learning_rate=0.05,n_estimators=200):
    '''
    算法为xgboost 这个未作任何的调参，仅仅是分析使用，无需达到模型的严格要求
    返回clf 算法对象, feat_imp 特征重要性
    '''
    param = {
        'max_depth': max_depth,
        'learning_rate':learning_rate,
        'verbosity':2,
        'objective': 'binary:logistic',
        'booster':'gbtree',
        'n_jobs':-1,
        'gamma':0.0001,
        'random_state':1024,
        'early_stopping_rounds':20
    }
    X_train = df[features]
    Y_label = df[[label]]
    d_train = xgb.DMatrix(X_train, label=Y_label)
    clf = xgb.XGBClassifier(**param,eval_metric='auc',n_estimators=n_estimators)

    #	 cvres=xgb.cv(clf.get_xgb_params(), d_train, num_boost_round = n_estimators, nfold =nfold,metrics= (['auc']), early_stopping_rounds = 20,seed=1024,)
    #	 print (cvres, cvres.shape)

    #	 clf.set_params(n_estimators=cvres.shape[0])
    print ('Model params: -----------')
    print (clf.n_estimators)
    print(clf.get_xgb_params())
    clf.fit(X_train,Y_label,eval_metric='auc')
    # Predict training set:
    dtrain_predictions = clf.predict(df[features])
    dtrain_predprob = clf.predict_proba(df[features])[:,1]

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