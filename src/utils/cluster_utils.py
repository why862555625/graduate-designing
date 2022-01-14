from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from src.common.init_train_data import train_data
import matplotlib.pyplot as plt
import random
import numpy as np

keras = tf.keras
layers = tf.keras.layers

def bench_k_means(kmeans, name, rate, data, labels):
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [rate * m(labels, estimator[-1].labels_) for m in clustering_metrics]
    results += [
        rate * metrics.silhouette_score(
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




def dispersed_point(rate):
    return rate * random.random() * random.random()


def init_cluster_data():
    X = []
    tf.random.set_seed(0)
    x_data, _, y = train_data()
    x_data = tf.convert_to_tensor(x_data) / 8000
    # 标准化  =>   (0,1)
    x_data = tf.reshape(x_data, shape=(-1, 16, 16))
    x_data = tf.expand_dims(x_data, -1)

    # 训练集形状(7956, 257)
    shape = x_data.shape

    inputs = layers.Input(shape=(shape[1], shape[2], 1), name='inputs')
    # 加入 批标准化
    encode = keras.layers.BatchNormalization()(inputs)
    encode = keras.layers.Conv2D(32, 3, strides=1, padding="same", activation='relu')(encode)
    encode = keras.layers.BatchNormalization()(encode)
    encode = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation='relu')(encode)
    encode = keras.layers.BatchNormalization()(encode)
    # input = (8*8)
    encode = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation='relu')(encode)
    encode = keras.layers.BatchNormalization()(encode)
    # input = (4*4)
    encode = keras.layers.Conv2D(64, 2, strides=2, padding="same", activation='relu')(encode)
    encode = keras.layers.BatchNormalization()(encode)
    # input = (2*2)
    encode = keras.layers.Conv2D(32, 1, strides=1, padding="same", activation='relu')(encode)
    encode = keras.layers.BatchNormalization()(encode)
    # input = (2*2)
    encode_output = keras.layers.Conv2D(1, 2, strides=1, padding="same", activation='relu')(encode)

    encoder = keras.Model(inputs, encode_output)
    encoder.load_weights("../main/sava/encoder")
    X_trained = encoder(x_data)
    x = X_trained.numpy()
    # 让点更加分散
    for i in x:
        X.append([float(i[0][0]) + dispersed_point(0.5),
                  float(i[1][0]) + dispersed_point(0.05)])
    return X, y


# 2d
data, label = init_cluster_data()
data = np.array(data)
label = np.array(label)

x = []
y = []
for i in data:
    x.append(i[0])
    y.append(i[1])
plt.scatter(x, y, s=6, c=label)
plt.savefig('../main/压缩后标签显示.jpg')
plt.show()