import numpy as np
from src.utils.cluster_utils import init_cluster_data
from sklearn.cluster import KMeans
from src.utils.cluster_utils import bench_k_means
import matplotlib.pyplot as plt

# 导入数据集
data, labels = init_cluster_data()
# 将训练集转化为numpy
data = np.array(data)
labels = np.array(labels)
# 算出分类数量
(n_samples, n_features), n_digits = data.shape, np.unique(labels).size

# 打印
print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
# 初始化为k-means++的聚类
kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", rate=1.2, data=data, labels=labels)
# 初始化为random的聚类

kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", rate=1.2, data=data, labels=labels)

print(82 * "_")

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(data)

# centroids = kmeans.cluster_centers_

centroids = [[1.88682042, 0.10928035],
             [2.4022796, 0.4929822],
             [2.85314042, 0.745028569],
             [2.11730389, 0.27801713]]
print(centroids)
# 画聚类中心
plt.scatter(
    [i[0] for i in centroids],
    [i[1] for i in centroids],
    marker="x",
    s=169,
    linewidths=3,
    color="red",
    zorder=10,
)
x = []
y = []
# 画数据集

for i in data:
    x.append(i[0])
    y.append(i[1])
plt.scatter(x, y, s=6, c=labels)
plt.savefig('./聚类后标签显示.jpg')
plt.show()
