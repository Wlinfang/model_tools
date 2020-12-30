'''
分类算法
'''

from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score

def cal_evaluate_classier(y_real,y_pred):
	
	'''
	二分类模型评估指标
	@param y_real : list，真实值
	@param y_pred : list, 预测值
	混淆矩阵：
			真实值
			1	0
	预测值 1 TP  FP
		  0 FN  TN
	准确率 Accuracy = (TP+TN) / (TP+FP+FN+TN)
	精确率(Precision) = TP / (TP+FP) --- 预测为正样本 分母为预测正样本
	召回率(Recall) = TP / (TP+FN) -- 实际为正样本 分母为真实正样本
	F-Meauter = (a^2 + 1) * 精确率 * 召回率 / [a^2 * (精确率 + 召回率)]
	a^2 如何定义
	F1分数(F1-Score) = 2 *  精确率 * 召回率 / (精确率 + 召回率)
	P-R曲线 ： 平衡点即为 F1分数；y-axis = 精确率；x-axis= 召回率
	平衡点 ： 精确率 = 召回率
	真正率(TPR) = TP / (TP+FN)-- 以真实样本 分母为真实正样本
	假正率(FPR) = FP / (FP+TN)-- 以真实样本 分母为真实负样本
	Roc 曲线：y-axis=真正率 ; x-axis=假正率； 无视样本不均衡问题
	AUC = Roc 曲线面积
	'''
	accuracy = accuracy_score(y_real, y_pred)
	# p=precision_score(y_real, y_pred)
	# f1=f1_score(y_real, y_pred)
	# 返回confusion matrix
	cm = confusion_matrix(y_real,y_pred)
	# 返回 精确率，召回率，F1
	cr = classification_report(y_real,y_pred)
	auc = roc_auc_score(y_real,y_pred) 

	fpr, tpr, thr = roc_curve(ytrue, yprob)
	ks = max(abs(tpr - fpr))
	# roc 图，ks 图

	gini = 2 * auc - 1

	