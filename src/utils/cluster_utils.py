from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from src.common.init_train_data import train_data
import matplotlib.pyplot as plt
import random
import numpy as np
import os.path as path
keras = tf.keras
layers = tf.keras.layers


model_data = path.join(path.dirname(path.abspath(__file__)), "../main/autoencoer/sava/DBLP/encoder")
model_data1 = path.join(path.dirname(path.abspath(__file__)), "../main/autoencoer/sava/DBLP/res_encoder")

save_img_path = path.join(path.dirname(path.abspath(__file__)), "../main/cluster/压缩后的标签显示.jpg")


## 正则卷积
def regurlarized_padded_conv(*args, **kwargs):
    return layers.Conv2D(*args, **kwargs, padding="same",
                         use_bias=False,
                         kernel_initializer="he_normal",
                         kernel_regularizer=keras.regularizers.l2(5e-4))


def regurlarized_padded_conv1(*args, **kwargs):
    return layers.Conv2DTranspose(*args, **kwargs, padding="same",
                                  use_bias=False,
                                  kernel_initializer="he_normal"
                                  )


class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ration=16):
        super(ChannelAttention, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()
        self.conv1 = layers.Conv2D(in_planes // ration, kernel_size=1, strides=1,
                                   padding="same",
                                   kernel_regularizer=keras.regularizers.l2(1e-4),
                                   use_bias=True, activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1,
                                   padding="same",
                                   kernel_regularizer=keras.regularizers.l2(1e-4),
                                   use_bias=True)

        def call(self, inputs):
            avg = self.avg(inputs)
            max = self.max(inputs)
            avg = layers.Reshape((1, 1, avg.shape[1]))(avg)
            max = layers.Reshape((1, 1, max.shape[1]))(max)
            avg_out = self.conv2(self.conv1(avg))
            max_out = self.conv2(self.conv1(max))
            out = avg_out + max_out
            out = tf.nn.sigmoid(out)
            return out


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regurlarized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)
        out = self.conv1(out)
        return out


class ChannelAttention1(layers.Layer):
    def __init__(self, in_planes, ration=16):
        super(ChannelAttention1, self).__init__()
        self.avg = layers.GlobalAveragePooling2D()
        self.max = layers.GlobalMaxPooling2D()

        self.conv1 = layers.Conv2D(in_planes // ration, kernel_size=1, strides=1,
                                   padding="same",
                                   kernel_regularizer=keras.regularizers.l2(1e-4),
                                   use_bias=True, activation=tf.nn.relu)

        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1,
                                   padding="same",
                                   kernel_regularizer=keras.regularizers.l2(1e-4),
                                   use_bias=True)

        def call(self, inputs):
            avg = self.avg(inputs)
            max = self.max(inputs)
            avg = layers.Reshape((1, 1, avg.shape[1]))(avg)

            max = layers.Reshape((1, 1, max.shape[1]))(max)

            avg_out = self.conv2(self.conv1(avg))
            max_out = self.conv2(self.conv1(max))

            out = avg_out + max_out
            out = tf.nn.sigmoid(out)
            return out


class SpatialAttention1(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()
        self.conv1 = regurlarized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)
    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        # 使用 tf.stack 将平均和最大堆叠到一起
        out = tf.stack([avg_out, max_out], axis=3)
        out = self.conv1(out)
        return out


class BasicBlock(layers.Layer):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = regurlarized_padded_conv(out_channels, kernel_size=3,
                                          strides=stride)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = regurlarized_padded_conv(out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()

        #注意力机制
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        # 3.判断stride是否等于1，如果为1就是没有降采样
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = keras.Sequential([regurlarized_padded_conv(self.expansion * out_channels,
                                                                 kernel_size=1, strides=stride),
                                        layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    def call(self, inputs, training=False):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        ########注意力机制###########
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = out + self.shortcut(inputs, training)
        out = tf.nn.relu(out)
        return out


class BasicBlock1(layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1, self).__init__()

        self.conv1 = regurlarized_padded_conv1(out_channels, kernel_size=3,
                                               strides=stride)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = regurlarized_padded_conv1(out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()

        ########注意力机制#################

        # 3.判断stride是否等于1，如果为1就是没有降采样
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = keras.Sequential([regurlarized_padded_conv1(self.expansion * out_channels,
                                                                  kernel_size=1, strides=stride),
                                        layers.BatchNormalization()])

        else:
            self.shortcut = lambda x, _: x

    def call(self, inputs, training=False):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        ########注意力机制###########

        out = out + self.shortcut(inputs, training)
        out = tf.nn.relu(out)

        return out


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=6):
        super(ResNet, self).__init__()
        self.in_channels = 16

        # 预测理卷积
        self.stem = keras.Sequential([
            regurlarized_padded_conv(16, kernel_size=3, strides=1),
            layers.BatchNormalization()
        ])
        # 创建4个残差网络
        self.layer1 = self.build_resblock(16, layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(16, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(16, layer_dims[1], stride=2)
        self.layer1 = self.build_resblock(16, layer_dims[0], stride=1)
        self.layer1 = self.build_resblock(1, layer_dims[0], stride=1)

        #         self.layer3 = self.build_resblock(256,layer_dims[2],stride=2)
        #         self.layer4 = self.build_resblock(512,layer_dims[3],stride=2)

    def call(self, inputs, training=False):
        out = self.stem(inputs, training)
        out = tf.nn.relu(out)

        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        #         out = self.layer3(out,training=training)
        #         out = self.layer4(out,training=training)
        return out

    #         self.final_bn = layers.BatchNormalization()
    #         self.avgpool =
    # 1.创建resBlock
    def build_resblock(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        res_blocks = keras.Sequential()
        for stride in strides:
            res_blocks.add(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return res_blocks



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
    encoder.load_weights(model_data)
    X_trained = encoder(x_data)
    x = X_trained.numpy()
    # 让点更加分散
    for i in x:
        X.append([float(i[0][0]) + dispersed_point(0.5),
                  float(i[1][0]) + dispersed_point(0.05)])
    return X, y


def init_cluster_data2():
    X = []
    tf.random.set_seed(0)
    x_data, _, y = train_data()
    x_data = tf.convert_to_tensor(x_data) / 8000
    # 标准化  =>   (0,1)
    x_data = tf.reshape(x_data, shape=(-1, 16, 16))
    x_data = tf.expand_dims(x_data, -1)

    # 训练集形状(7956, 257)
    shape = x_data.shape

    inputs = tf.keras.layers.Input(shape=(shape[1], shape[2], 1), name='inputs')
    encoder = BasicBlock(1, 16, 1)(inputs)
    encoder = BasicBlock(16, 32, 2)(encoder)
    encoder = BasicBlock(32, 32, 2)(encoder)
    encoder = BasicBlock(32, 32, 2)(encoder)
    encode_output = BasicBlock(32, 1, 1)(encoder)
    encoder = keras.Model(inputs, encode_output)

    encoder.load_weights(model_data1)
    X_trained = encoder(x_data)
    x = X_trained.numpy()
    # 让点更加分散
    for i in x:
        X.append([float(i[0][0]) + dispersed_point(0.1),
                  float(i[1][0]) + dispersed_point(0.1)])
    return X, y


# 2d
data, label = init_cluster_data2()
data = np.array(data)
label = np.array(label)

x = []
y = []
for i in data:
    x.append(i[0])
    y.append(i[1])
plt.scatter(x, y, s=6, c=label)
plt.savefig(save_img_path)
plt.show()