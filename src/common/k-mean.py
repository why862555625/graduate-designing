# n_clusters : 聚类的个数k，default：8.
# init : 初始化的方式，default：k-means++
# n_init : 运行k-means的次数，最后取效果最好的一次, 默认值: 10
# max_iter : 最大迭代次数, default: 300
# tol : 收敛的阈值, default: 1e-4
# n_jobs : 多线程运算, default=None，None代表一个线程，-1代表启用计算机的全部线程。
# algorithm : 有“auto”, “full” or “elkan”三种选择。"full"就是我们传统的K-Means算法， “elkan”是我们讲的elkan K-Means算法。默认的"auto"则会根据数据值是否是稀疏的，来决定如何选择"full"和“elkan”。一般数据是稠密的，那么就是“elkan”，否则就是"full"。一般来说建议直接用默认的"auto"。

# 算法伪代码


# 创建 k 个点作为起始质心（随机选择）
# 当任意一个点的簇分配结果发生改变时（不改变时算法结束）
#     对数据集中的每个数据点
#         对每个质心
#             计算质心与数据点之间的距离
#         将数据点分配到距其最近的簇
#     对每一个簇, 计算簇中所有点的均值并将均值作为质心


import numpy as np
import matplotlib.pyplot as plt
import glog as log

from sklearn.cluster import DBSCAN  # 进行DBSCAN聚类
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # 计算 轮廓系数，CH 指标，DBI


# 定义一个进行DBSCAN的函数
def DBSCAN_Cluster(embedding_image_feats):
    """
    dbscan cluster
    :param embedding_image_feats:  # 比如形状是（9434,4）表示9434个像素点
    :return:
    """
    db = DBSCAN(eps=0.35, min_samples=600)
    try:
        features = StandardScaler().fit_transform(embedding_image_feats)  # 将特征进行归一化
        db.fit(features)
    except Exception as err:
        log.error(err)
        ret = {
            'origin_features': None,
            'cluster_nums': 0,
            'db_labels': None,
            'cluster_center': None
        }
        return ret

    db_labels = db.labels_  # 获取聚类之后没一个样本的类别标签
    unique_labels = np.unique(db_labels)  # 获取唯一的类别

    num_clusters = len(unique_labels)
    cluster_centers = db.components_

    ret = {
        'origin_features': features,  # (9434,4)
        'cluster_nums': num_clusters,  # 5  它是一个标量，表示5类，包含背景
        'db_labels': db_labels,  # (9434,)
        'unique_labels': unique_labels,  # (5,)
        'cluster_center': cluster_centers  # (6425,4)
    }

    return ret


# 画出聚类之后的结果
def plot_dbscan_result(features, db_labels, unique_labels, num_clusters):
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, color in zip(unique_labels, colors):
        if k == -1:
            color = 'k'  # 黑色的，这代表噪声点

        index = np.where(db_labels == k)  # 获取每一个类别的索引位置
        x = features[index]

        plt.plot(x[:, 0], x[:, 1], 'o', markerfacecolor=color, markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % num_clusters)
    plt.show()


if __name__ == '__main__':
    embedding_features = np.load("./tools/features_logits/lane_embedding_feats.npy")  # 导入数据，数据格式为（samples，）

    ret = DBSCAN_Cluster(embedding_features)  # 进行 DBSCAN聚类

    plot_dbscan_result(ret['origin_features'], ret['db_labels'], ret['unique_labels'], ret['cluster_nums'])  # 展示聚类之后的结果

    s1 = silhouette_score(embedding_features, ret['db_labels'], metric='euclidean')  # 计算轮廓系数
    s2 = calinski_harabasz_score(embedding_features, ret['db_labels'])  # 计算CH score
    s3 = davies_bouldin_score(embedding_features, ret['db_labels'])  # 计算 DBI

    print(s1)
    print(s2)
    print(s3)

'''运行结果为：
0.7971864
48375.80213812995
0.8799878743935938
'''
