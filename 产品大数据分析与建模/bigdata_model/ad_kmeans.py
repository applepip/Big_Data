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

print(data_fillna)

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

print(unique_list)

#将每条记录的具体值跟其在唯一值列表中的索引做映射
for line_index in range(lines):
    each_record = convert_matrix.iloc[line_index]
    for each_index, each_data in enumerate(each_record):
        list_value = unique_list[each_index]
        each_record[each_index] = list_value.index(each_data)
    each_dict = dict(zip(conver_cols, each_record))
    dict_list.append(each_dict)

print(dict_list[0:5])

#使用DictVectorizer将字符串转换为整数
model_dvtransform = DictVectorizer(sparse=False, dtype=np.int64)
data_dictvec = model_dvtransform.fit_transform(dict_list)

print(data_fillna)

#3 数据标准化
scale_matrix = data_fillna.iloc[:, 1:8] #获得要转换的矩阵
minmax_scaler = MinMaxScaler() #
data_scaled = minmax_scaler.fit_transform(scale_matrix)

#4 合并所有输入维度
X = np.hstack((data_scaled, data_dictvec))

print(X[0:5])


'''
步骤5 通过平均轮廓系数检验得到最佳KMeans聚类模型
'''

score_list = list() #用来存储每个k下模型的平均轮廓系数

silhouette_int = -1 #初始化的平均轮廓阀值

for n_clusters in range(2, 10): #遍历从2-10几个有限组
    model_kmeans = KMeans(n_clusters=n_clusters, random_state=0) # 建立聚类模型对象
    cluster_labels_tmp = model_kmeans.fit_predict(X) #训练聚类模型
    silhouette_tmp = metrics.silhouette_score(X, cluster_labels_tmp)#得到每个k下的平均轮廓系数
    if silhouette_tmp > silhouette_int:#如果平均轮廓系数更高
        best_k = n_clusters #将最好的k存储下来
        silhouette_int = silhouette_tmp #将最好的平均轮廓得分存储下来
        best_kmeans = model_kmeans #将最好的模型存储下来
        cluster_labels_k = cluster_labels_tmp #将最好的聚类标签存储下来
    score_list.append([n_clusters, silhouette_tmp])

print('{:-^60}'.format('K值和平均轮廓系数:'))
print(np.array(score_list)) #打印输出k下的详细得分
print('最优K值是:{0} 该K值的平均轮廓系数是{1}'.format(best_k, silhouette_int.round(4)))

'''
步骤6 针对聚类结果的特征分析
'''
# 6.1 将原始数据与聚类标签整合
cluster_labels = pd.DataFrame(cluster_labels_k, columns=['clusters'])  # 获得训练集下的标签信息
merge_data = pd.concat((data_fillna, cluster_labels), axis=1)  # 将原始处理过的数据跟聚类标签整合

# 6.2 计算每个聚类类别下的样本量和样本占比
clustering_count = pd.DataFrame(merge_data['渠道代号'].groupby(merge_data['clusters']).count()).T.rename({'渠道代号': 'counts'})
clustering_ratio = (clustering_count / len(merge_data)).round(2).rename({'counts': 'percentage'})

# 6.3 计算各个聚类类别内部最显著特征值
cluster_features = []  # 存储最终合并后的所有特征信息
for line in range(best_k):  # 读取每个类索引
    label_data = merge_data[merge_data['clusters'] == line]  # 获得特定类的数据
    part1_data = label_data.iloc[:, 1:8]  # 获得数值型数据特征
    part1_desc = part1_data.describe().round(3)  # 描述性统计信息
    merge_data1 = part1_desc.iloc[2, :]  # 得到数值型特征的均值

    part2_data = label_data.iloc[:, 8:-1]  # 获得字符型数据特征
    part2_desc = part2_data.describe(include='all')  # 获得描述统计信息
    merge_data2 = part2_data.iloc[2, :]  # 获得字符型特征的最频繁值

    merge_line = pd.concat((merge_data1, merge_data2), axis=0)  # 按行合并
    cluster_features.append(merge_line)  # 将每个类别下的数据特征追加到列表

# 6.4 输出完整的类别特征信息
cluster_pd = pd.DataFrame(cluster_features).T  # 将列表转化为矩阵
print('{:-^60}'.format('所有clusters的具体特征:'))
all_cluster_set = pd.concat((clustering_count, clustering_ratio, cluster_pd), axis=0)  # 将每个列别的所有信息合并
print(all_cluster_set)


'''
步骤7 各类别显著值特征对比
'''
#7.1 各类别数据预处理（标准化）
num_sets = cluster_pd.iloc[:6, :].T.astype(np.float64) #获取要展示的数据
num_sets_max_min = minmax_scaler.fit_transform(num_sets) #获得标准化后的数据

#7.2 画布基本设置
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, polar=True)
labels = np.array(merge_data1.index[:-1])
color_list = ['r','g','b','c']
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

#7.3 画雷达图
for i in range(len(num_sets)):
    data_tmp = num_sets_max_min[i, :]
    data = np.concatenate((data_tmp, [data_tmp[0]]))
    ax.plot(angles, data, 'o-', c=color_list[i], label=i,linewidth=2.5)
    ax.set_thetagrids(angles*180/np.pi, labels, fontproperties='SimHei',fontsize=14)
    ax.set_title('各聚类类别显著特征对比',fontproperties='SimHei',fontsize=18)
    ax.set_rlim(-0.2, 1.2)
    plt.legend(loc=0)
    #plt.show()

plt.show()