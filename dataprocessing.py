# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import accuracy_score
import pydotplus
from IPython.display import Image
pd.set_option('display.max_columns',500)

import graphviz
'''
数据预处理
'''
# 缺失值
data = pd.read_csv('credit.csv')
data = pd.DataFrame(data)
print(data.dtypes)
print(data.isnull().any())
null_nums = sum(data['MONTHLY_INCOME_WHITHOUT_TAX'].isnull())
ratio = round(null_nums / data.shape[0] * 100, 3)
print('月收入数据缺失个数为' + str(null_nums) + ',缺失率为'+ str(ratio) + '%')
# 用中位数填充
median = round(data.median()['MONTHLY_INCOME_WHITHOUT_TAX'])
data.fillna(median, inplace=True)
# 异常值
col_proc = ['ID','Label','AGE','MONTHLY_INCOME_WHITHOUT_TAX',
            'GAGE_TOTLE_PRICE','APPLY_AMOUNT','APPLY_TERM_TIME',
           'APPLY_INTEREST_RATE']
data1 = data[col_proc]
data_scal = (data1 - data1.mean()) / (data1.std())
sns.set()
plt.boxplot(x=data_scal.values,labels=data_scal.columns,sym='r',vert=False,patch_artist=True)
plt.show()
# 返回月收入、申请额度和总价值的异常值索引
def Outrange(ser):
    low = ser.quantile(0.25) - 1.5 * (ser.quantile(0.75) - ser.quantile(0.25))
    up = ser.quantile(0.75) + 1.5 * (ser.quantile(0.75) - ser.quantile(0.25))
    index = (ser < low) | (ser > up)
    return index
# 月收入异常值索引130个
ser1 = data1['MONTHLY_INCOME_WHITHOUT_TAX']
outrange1 = Outrange(ser1)
# 总价值异常值索引77个
ser2 = data1['GAGE_TOTLE_PRICE']
outrange2 = Outrange(ser2)
# 申请额度异常值索引97个
ser3 = data1['APPLY_AMOUNT']
outrange3 = Outrange(ser3)
# 求三个属性异常值的并集196个
print('异常样本数为：',sum(outrange1|outrange2|outrange3))
# 展示异常样本
data_interest = data1[outrange1|outrange2|outrange3]
ratio_interest = sum(data_interest['Label']) / 196
print('异常样本中违约人的比例:%.2f' % ratio_interest)
# 异常样本散点图
plt.subplot(1,3,1)
plt.scatter(x=data_interest.index,y=data_interest['MONTHLY_INCOME_WHITHOUT_TAX'])
plt.subplot(1,3,2)
plt.scatter(x=data_interest.index,y=data_interest['GAGE_TOTLE_PRICE'])
plt.subplot(1,3,3)
plt.scatter(x=data_interest.index,y=data_interest['APPLY_AMOUNT'])
plt.show()
# 去除月收入中的两个最大值
data = data.drop(data['MONTHLY_INCOME_WHITHOUT_TAX'].idxmax())
data = data.drop(data['MONTHLY_INCOME_WHITHOUT_TAX'].idxmax())
# 去除总价值大于6000000的值
data = data.drop(data[data['GAGE_TOTLE_PRICE'] > 6000000].index)
# 去除申请额度大于3000000的值
data = data.drop(data[data['APPLY_AMOUNT'] > 3000000].index)
size = data.shape[0]
bad = data['Label'].sum()
data = data.drop('ID', axis=1)
print('违约客户数为：%d,未违约客户数为:%d,违约比例为:%.2f' % (bad,size-bad,bad/size))
data = data[['GENDER','MARITAL_STATUS','LOANTYPE',
            'PAYMENT_TYPE','APPLY_TERM_TIME','AGE','MONTHLY_INCOME_WHITHOUT_TAX',
            'GAGE_TOTLE_PRICE','APPLY_AMOUNT','APPLY_INTEREST_RATE','Label']]
print(data.dtypes)
data.to_csv('/Users/lucifer/PycharmProjects/datascience/datamining/DT/new_data.csv', index=False)