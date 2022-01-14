import tensorflow as tf
from src.common.init_train_data import train_data
import matplotlib.pyplot as plt
import random
import numpy as np

keras = tf.keras
layers = tf.keras.layers


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
    encoder.load_weights("./sava/encoder")
    X_trained = encoder(x_data)
    x = X_trained.numpy()
    # 让点更加分散
    for i in x:
        X.append([float(i[0][0]) + 0.5 * random.random() * random.random(),
                  float(i[1][0]) + 0.02 * random.random() * random.random()])
    return X, y


# # 2d
# data, label = init_cluster_data()
# data = np.array(data)
# label = np.array(label)
#
# x = []
# y = []
# for i in data:
#     x.append(i[0])
#     y.append(i[1])
# plt.scatter(x, y, s=6, c=label)
# plt.savefig('./压缩后标签显示.jpg')
# plt.show()
