"""
决策树相关算法
"""
import pandas as pd
from sklearn import tree
import pydotplus
import graphviz
import shap


def build_tree(df, features, target, max_depth=3):
    """
    构建决策回归树
    target:二分类
    返回：dtr graph ftr_imp
    dtr:决策树；graph:决策树图；ftr_imp:特征重要性
    """
    dtr = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_split=200, random_state=1024)
    dtr.fit(df[feature], df[target])
    dot_data = tree.export_graphviz(dtr, out_file=None, feature_names=feature, class_names=target,
                                    filled=True, impurity=False, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    #     graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    graph = graphviz.Source(dot_data)
    #     graph.write_pdf("tree.pdf")
    ftr_imp = pd.Series(dtr.feature_importances_, feature)
    return dtr, graph, ftr_imp


def cal_shap_value(clf, df, features, top_num=5, is_show=False):
    """
    计算shap value 值
    clf:estimators 算法对象 限制为树类算法
    features: 入模特征
    top_num : 默认提取前5个特征
    is_show : 是否画图shap value False : 不画图
    """
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

    f = pd.DataFrame(shap_values, columns=features)
    f = np.abs(f).mean()
    ftr_imp = pd.Series(clf.get_booster().get_fscore(), features).sort_values(ascending=False, na_position='last').head(
        top_num)
    f = f[f > 0.000].sort_values(ascending=False).reset_index().rename(columns={'index': 'f', 0: 'value'})
    f = f[f['f'].isin(ftr_imp.index.values)]

    if is_show == True:
        shap.summary_plot(shap_values, df[features], plot_type="bar")
    return f
