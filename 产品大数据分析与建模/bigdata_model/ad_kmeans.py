# -*- coding: UTF-8 -*-

'''
步骤1 导入库
'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

'''
步骤2 读取数据
'''
raw_data = pd.read_table('data/ad_performance.txt', delimiter='\t')

'''
步骤3 读取数据
'''
print(raw_data.head(2)) #查看前2条数据
print(pd.DataFrame(raw_data.dtypes).T) #查看数据类型
print(pd.DataFrame(raw_data.isnull().sum()).T) #查看缺失值
print(raw_data.describe().round(2).T) #查看数据描述性统计信息
print(raw_data.corr().round(2).T) #使用corr()做相关性分析

'''
步骤4 数据预处理
'''
#1 缺失值替换
data_fillna = raw_data.fillna(raw_data['平均停留时间'].mean()) #用均值替换缺失值

#2 字符串分类转整数类型
#定义要转换的数据
conver_cols = ['素材类型','广告类型','合作方式','广告尺寸','广告卖点']
convert_matrix = data_fillna[conver_cols]
lines = data_fillna.shape[0]
dict_list = []
unique_list = []

#获得所有列的唯一值列表
for col_name in conver_cols:
    cols_unique_value = data_fillna[col_name].unique().tolist()
    unique_list.append(cols_unique_value)

print(unique_list)  #查看前2条数据

#将每条记录的具体值跟其在唯一值列表中的索引做映射
for line_index in range(lines):
    each_record = convert_matrix.iloc[line_index]
    for each_index, each_data in enumerate(each_record):
        list_value = unique_list[each_index]
        each_record[each_index] = list_value.index(each_data)
    each_dict = dict(zip(conver_cols, each_record))
    dict_list.append(each_dict)