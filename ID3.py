# -*- coding:utf-8 -*-
'''
可以做二叉树，多叉树，多类别树
'''

# 分割数据集
import pandas as pd
from math import *
data = pd.read_csv('new_data.csv')
data['APPLY_TERM_TIME'] = data['APPLY_TERM_TIME'].astype(str)

# 按违约比例进行随机分层抽样,抽100个样本做测试集
data1 = data[data.eval('Label == 1')]
data0 = data[data.eval('Label == 0')]
testing = pd.concat([data1.sample(46, random_state=10),
                     data0.sample(54, random_state=10)])
training = data.drop(index=testing.index)
testing = testing.reset_index(drop=True)
testing_y = list(testing.iloc[:,-1])
testing_x = testing.drop('Label',axis=1)
# 求数据集的信息熵
def getInfoEntropy(data):
    # 根据标签统计不同样本的个数
    count_class = pd.value_counts(data.iloc[:,-1], sort=True)
    #print(count_class):0,702;1,600
    size = data.shape[0]
    Entropy = 0.0
    for i in range(len(count_class)):
        p = count_class.iloc[i] / size
        Entropy += (-p * log(p, 2))
    return Entropy
getInfoEntropy(training)

# 划分数据：离散型特征根据类别划分，连续性特征用二分法
# 离散型数据划分:每一个类别生成一个数据集
def split_data(data, column):
    splt_datas = pd.Series()
    str_values = data.iloc[:, column].unique()
    # print(str_values):'Female','Male'
    for i in range(len(str_values)):  # 遍历对应类别，找出类别对应的数据集
        df = data.loc[data.iloc[:,column] == str_values[i]]
        splt_datas[str(i)] = df
    return splt_datas

# 连续型数据划分
def split_continues(data, column):
    splt_datas = pd.Series()
    series_data_sort = data.iloc[:,column].sort_values()
    split_values = []  # 存储所有的划分点
    # 求划分点
    for i in range(len(series_data_sort)-1):
        split_values.append((series_data_sort.iloc[i] +
                             series_data_sort.iloc[i+1]) / 2)
    # 二分法寻找最佳划分点
    best_split_value = 0
    minInfoGain = 100
    for i in range(len(split_values)):
        # 根据划分点将数据划分为左右两个子集
        left_data = data.loc[data.iloc[:,column] <= split_values[i]]
        right_data = data.loc[data.iloc[:,column] > split_values[i]]
        InfoGain = len(left_data) / len(data) * getInfoEntropy(left_data) + \
                   len(right_data) / len(data) * getInfoEntropy(right_data)  # 不是信息增益，跟信息增益的方向相反
        # 判断该划分点是否是最佳划分点
        if InfoGain < minInfoGain:
            minInfoGain = InfoGain
            best_split_value = split_values[i]
    left_data = data.loc[data.iloc[:,column] <= best_split_value]
    right_data = data.loc[data.iloc[:,column] > best_split_value]
    series = pd.Series()
    series['0'] = left_data
    series['1'] = right_data
    return series, best_split_value, minInfoGain
# seris存储左右两个划分集，best_split_value存储最有划分点,minInfoGain存储划分后的信息熵

# 求当前最优划分属性
def find_best_feature(data):
    best_feature_index = 0  # 保存最有划分特征的索引
    minInfoGain = 100
    size = data.shape[0]
    best_split_value_return = 0  # 如果是连续性属性，还需保存对应的最有划分点
    best_split_value = 0
    for i in range(data.shape[1]-1):
        InfoGain = 0
        series = pd.Series()
        if i < data.shape[1] - 6:  # 离散型属性
            series = split_data(data, i)
            for j in range(len(series)):  # 该属性的信息增益
                df = series[j]  # 每个类别组成的数据集
                InfoGain += len(df) / size * (getInfoEntropy(df))
            print('属性%d的信息增益是：%.3f' % (i,InfoGain))
        else:  # 连续性属性
            series,best_split_value,InfoGain = split_continues(data, i)
            print('属性%d的信息增益是：%.3f' % (i,InfoGain))
        if InfoGain < minInfoGain:
            minInfoGain = InfoGain
            InfoGain = 0.0
            best_feature_index = i
            best_Series = series
            if i >= data.shape[1] - 6:  # 连续性属性要存储最有划分点
                best_split_value_return = best_split_value
    # 返回最优属性、属性类别划分数据集、连续性属性的最优划分点
    return data.columns[best_feature_index],best_Series,best_split_value_return
# 构建决策树
def creat_Tree(data):
    y_values = data.iloc[:,-1].unique()  # 当前数据集标签的类别
    if len(y_values) == 1:  # 若只剩一个类别，数据纯净，返回
        return y_values[0]
    flag = 0
    for i in range(1, data.shape[1]):  # 若当前节点在所剩属性中取值相同，则返回
        if len(data.iloc[:,i].unique()) != 1:
            flag = 1
            break
    if (flag == 0):
        value_count = pd.value_counts(data.iloc[:, -1])  # 当前属性的各个类别
        return value_count.index[0]
    # 当前属性可分，寻找最优特征
    best_feature,best_Series,best_split_value = find_best_feature(data)
    Tree = {best_feature:{}}
    for j in range(len(best_Series)):  # 遍历划分后的数据集
        split_data = best_Series.iloc[j]
        value = ''
        if best_split_value == 0.0:  # 离散特征
            value = split_data.loc[:,best_feature].unique()[0]  # 获取划分特征中对应的特征类别
            split_data = split_data.drop(best_feature, axis=1)  # 离散型特征用完后即删除
        else:
            if j == 0:
                value = '<=' + str(best_split_value)
            else:
                value = '>' + str(best_split_value)
        Tree[best_feature][value] = creat_Tree(split_data)
    return Tree
Tree = creat_Tree(training)
# 测试
# 预测一个样本
def classification_one(Tree, data):
    # print(Tree)
    first_key = list(Tree.keys())[0]  # 获取根节点的key
    # print('根节点的key是:',first_key)
    first_value = Tree[first_key] # 获取根节点对应的value
    # print('根节点的value是：',first_value.keys())
    result = -1
    if ('<' in list(first_value.keys())[0]):  # 连续性特征
        left_key = list(first_value.keys())[0]  # 连续性特征中分割点左边的键
        right_key = list(first_value.keys())[1]
        split_poit = float(''.join(list(left_key)[2:]))  # 分割点
        if data[first_key] <= split_poit:  # 如果属于左分支
            # 判断是否是叶子节点，如果对应的value还是一个字典，说明是非叶子节点
            if isinstance(first_value[left_key], dict):
                result = classification_one(first_value[left_key], data)
            else:
                result = first_value[left_key]
        else:
            if isinstance(first_value[right_key], dict):
                result = classification_one(first_value[right_key], data)
            else:
                result = first_value[right_key]
    else:  # 离散型特征
        # 可以换成判断key值是否存在
        if(isinstance(first_value[data[first_key]], dict)):
            result = classification_one(first_value[data[first_key]], data)
        else:
            result = first_value[data[first_key]]
    return result

# 预测多个样本
def classification_more(Tree, data):
    result_list = []
    for i in range(data.shape[0]):
        result = classification_one(Tree, data.iloc[i])
        result_list.append(result)
    return result_list
# 类别预测值
pred = classification_more(Tree, testing_x)
print('预测值：',pred)
print('真实值：',testing_y)

# 计算查全率和查准率
TP = 0
for i in range(100):
    if pred[i] == 1 and testing_y[i] == 1:
        TP += 1
precision = TP / sum(pred)
recall = TP / sum(testing_y)
print('查全率为：%.2f,查准率为：%.2f' % (recall, precision))

