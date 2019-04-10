#-*- coding: utf-8 -*-
#数据清洗，过滤掉不符合规则的数据

import pandas as pd
from sklearn.cluster import KMeans #导入K均值聚类算法
import numpy as np
import matplotlib.pyplot as plt

def cleanData(datafile, cleanedfile):
    '''
    @ Datoo
    进行数据清洗，丢弃票价为空记录，丢弃票价为0，折扣不为0且飞行距离大于0的距离
    将结果写入 cleanedfile
    '''
    data = pd.read_csv(datafile,encoding='utf-8')
    data = data[data['SUM_YR_1'].notnull()*data['SUM_YR_2'].notnull()]  # 票价非空值才保留
    index1 = data['SUM_YR_1'] != 0                                      # 只保留票价非零的
    index2 = data['SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)    # 平均折扣率与总飞行公里数同时为0的记录，该规则是“与”
    data = data[index1 | index2 | index3]                               # 该规则是“或”

    data.to_csv(cleanedfile)
    print("cleanData is finished...")


def reductionData(datafile, reductionfile):
    '''
    @ Datoo
    将清洗后的数据进行属性规约
    将结果写入 reductionfile
    '''
    data = pd.read_csv(datafile, encoding='utf-8')
    data = data[['LOAD_TIME', 'FFP_DATE', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount']]

    # 计算 L = FFP_DATE - LOAD_TIME
    d_ffp = pd.to_datetime(data['FFP_DATE'])
    d_load = pd.to_datetime(data['LOAD_TIME'])
    res = d_load - d_ffp
    data['L'] = res.map(lambda x: x / np.timedelta64(30 * 24 * 60, 'm'))

    data['R'] = data['LAST_TO_END']
    data['F'] = data['FLIGHT_COUNT']
    data['M'] = data['SEG_KM_SUM']
    data['C'] = data['avg_discount']
    data = data[['L', 'R', 'F', 'M', 'C']]
    data.to_csv(reductionfile, index=False)
    print("reduction is finished ...")

def exploreData(datafile, resultfile):
    '''
    @ Datoo
    对数据进行缺失值分析与异常值分析
    将结果写入 resultfile
    '''
    data = pd.read_csv(datafile, encoding='utf-8')
    explore = data.describe(percentiles=[], include='all').T            # describe()：返回统计信息
    explore['null'] = len(data) - explore['count']                      # describe()：自动计算非空值数，需要手动计算空值数
    explore = explore[['null', 'max', 'min']]
    explore.columns = [u'空值数', u'最大值', u'最小值']                 # 表头重命名

    explore.to_excel(resultfile)  # 导出结果
    print("exploreData is finished ...")

def zscoreData(datafile ,zscoredfile):
    '''
    @ Datoo
    对数据进行标准化处理，此处选择 zscore 标准化
    将结果写入 zscoredfile
    '''
    data = pd.read_csv(datafile)
    data = (data - data.mean(axis=0)) / (data.std(axis=0))
    data.columns = ['Z' + i for i in data.columns]

    data.to_excel(zscoredfile, index=False)
    print("zscoreData is finished ...")

def KMeansCluster(inputfile, clusterfile):
    '''
    @ Datoo
    利用Kmeans聚类算法对客户进行分类
    '''

    plt.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    k = 5                                                               # 需要进行的聚类类别数
    data = pd.read_excel(inputfile)                                     # 读取数据并进行聚类分析
    kmodel = KMeans(n_clusters=k, n_jobs=4)                             # n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(data)                                                    # 训练模型

    # print('center: \n ', kmodel.cluster_centers_)                       # 查看聚类中心
    # print('labels: \n ', kmodel.labels_)                                # 查看各样本对应的类别

    r1 = pd.Series(kmodel.labels_).value_counts()
    r2 = pd.DataFrame(kmodel.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(data.columns) + ['类别数目']
    # print(r)
    index = ["","第一类客户","第二类客户","第三类客户","第四类客户","第五类客户"]
    r.to_csv(clusterfile,index=index, encoding='utf_8_sig')


    print("k-means is finished ...")


    labels = data.columns                                               # 标签
    plot_data = kmodel.cluster_centers_
    color = ['b', 'g', 'r', 'c', 'y']  # 指定颜色

    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    plot_data = np.concatenate((plot_data, plot_data[:, [0]]), axis=1)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)                               # polar参数！！
    for i in range(len(plot_data)):
        ax.plot(angles, plot_data[i], 'o-', color=color[i], label=u'客户群' + str(i), linewidth=2)  # 画线

    ax.set_rgrids(np.arange(0.01, 3.5, 0.5), np.arange(-1, 2.5, 0.5), fontproperties="SimHei")
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontproperties="SimHei")
    plt.legend(loc=4)
    plt.show()


datafile= '../data/air_data.csv'
cleanedfile = '../tmp/data_cleaned0321.csv'
# cleanData(datafile, cleanedfile)

reductionfile = '../tmp/reduction.csv'
# reductionData(cleanedfile, reductionfile)

resultfile = '../tmp/explore0321.xls'
# exploreData(reductionfile, resultfile)



zscoredfile = '../tmp/zscoreddata0321.xls'
# zscoreData(reductionfile, zscoredfile)

clusterfile = '../tmp/clusterResult0321.csv'
KMeansCluster(zscoredfile, clusterfile)