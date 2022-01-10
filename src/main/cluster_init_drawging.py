import tensorflow as tf
from src.common.init_train_data import train_data

tf.random.set_seed(0)
_, x_data = train_data()
keras = tf.keras
layers = tf.keras.layers
# 将训练集转化为tensor
x_data = tf.convert_to_tensor(x_data)
# 训练集形状(7956, 257)
shape = x_data.shape
print(shape)
# (257,1)
inputs = layers.Input(shape=(shape[1], 1), name='inputs')
encode = layers.Flatten()(inputs)
encode = keras.layers.Dense(1024, activation='sigmoid')(encode)
# 加入 批标准化
encode = keras.layers.BatchNormalization()(encode)
encode = keras.layers.Dense(256, activation='sigmoid')(encode)
encode = keras.layers.BatchNormalization()(encode)
encode = keras.layers.Dense(2, activation='sigmoid')(encode)
encode_output = keras.layers.BatchNormalization()(encode)

encoder = keras.Model(inputs, encode_output)
encoder.load_weights("./sava/encoder")

X_trained = encoder(x_data)
x = X_trained.numpy()
print(x)
