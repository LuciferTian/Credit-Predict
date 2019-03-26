# -*- coding:utf-8 -*-
# 分割数据集
import pandas as pd
import numpy as np
import codecs
from math import *
import graphviz
import pydotplus
from IPython.display import Image
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, classification_report
data = pd.read_csv('new_data.csv')
data['APPLY_TERM_TIME'] = data['APPLY_TERM_TIME'].astype(str)
np.random.seed(12)
# 离散属性编码
le = LabelEncoder()
data[['GENDER','MARITAL_STATUS','LOANTYPE','PAYMENT_TYPE',
      'APPLY_TERM_TIME']] = data[['GENDER','MARITAL_STATUS','LOANTYPE','PAYMENT_TYPE',
                                                 'APPLY_TERM_TIME']].apply(le.fit_transform)

# 按违约比例进行随机分层抽样,抽100个样本
data1 = data[data.eval('Label == 1')]
data0 = data[data.eval('Label == 0')]
testing = pd.concat([data1.sample(46),
                     data0.sample(54)])
training = data.drop(index=testing.index)
training_y = training.iloc[:,-1]
training_x = training.drop('Label', axis=1)
testing = testing.reset_index(drop = True)
testing_y = list(testing.iloc[:,-1])
testing_x = testing.drop('Label', axis=1)

# 模型
def tree_model(model='DT'):
    if model == 'DT':
        clf = DecisionTreeClassifier()
        clf_tree = clf.fit(training_x, training_y)
        pred = clf_tree.predict(testing_x)
        # 可视化
        data_feature_name = training_x.columns
        training['Label'] = training['Label'].astype(str)
        print(type(training['Label']))
        data_target_name = np.unique(training['Label'])
        dot_tree = tree.export_graphviz(clf, out_file=None,
                                        feature_names=data_feature_name, class_names=data_target_name,
                                        filled=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_tree)
        img = Image(graph.create_png())
        graph.write_png("tree.png")
    elif model == 'GBDT':
        clf = GradientBoostingClassifier(n_estimators=3000, max_depth=4, learning_rate=0.1)
        clf.fit(training_x, training_y)
        pred = clf.predict(testing_x)
    else:
        clf = XGBClassifier()
        clf.fit(training_x, training_y)
        pred = clf.predict(testing_x)
    return clf, pred
clf, pred = tree_model('XGB')
# 混淆矩阵、查准率、查全率
print(confusion_matrix(testing_y, pred))
print(classification_report(testing_y, pred))


