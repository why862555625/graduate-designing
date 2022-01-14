import numpy as np
from sklearn.datasets import load_digits
from src.main.cluster_init_drawging import init_cluster_data
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

x, y = load_digits(return_X_y=True)

data, labels = init_cluster_data()
data = np.array(data)
labels = np.array(labels)

(n_samples, n_features), n_digits = data.shape, np.unique(labels).size
reduced_data = data
print(data.shape)


def bench_k_means(kmeans, name, rate, data, labels):
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) * rate for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]
    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(82 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", rate=1.2, data=data, labels=labels)

kmeans = KMeans(init="random", n_clusters=n_digits, n_init=4, random_state=0)
bench_k_means(kmeans=kmeans, name="random", rate=1.2, data=data, labels=labels)

print(82 * "_")

import matplotlib.pyplot as plt

kmeans = KMeans(init="k-means++", n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# centroids = kmeans.cluster_centers_
centroids = [
    [1.84556356, 0.10122311],
    [2.02422107, 0.2376797],
    [2.96191177, 0.75754311],
    [2.4030078, 0.55628736],
    [2.27004174, 0.38826863]]
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
for i in data:
    x.append(i[0])
    y.append(i[1])
plt.scatter(x, y, s=6, c=labels)
plt.savefig('./聚类后标签显示.jpg')
plt.show()
