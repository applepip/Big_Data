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
print raw_data.head(2) #查看前2条数据
print pd.DataFrame(raw_data.dtypes).T #查看数据类型
print pd.DataFrame(raw_data.isnull().sum()).T #查看缺失值
print raw_data.describe().round(2).T #查看数据描述性统计信息
print raw_data.corr().round(2).T #使用corr()做相关性分析

'''
步骤4 数据预处理
'''
#1 缺失值替换
data_fillna = raw_data.fillna(raw_data['平均停留时间'].mean()) #用均值替换缺失值

#2 字符串分类转整数类型
#定义要转换的数据
conver_cols = ['素材类型','广告类型','合作方式','广告尺寸','广告卖点']