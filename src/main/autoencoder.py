import tensorflow as tf
from src.common.init_train_data import train_data

tf.random.set_seed(0)
x_data, _ = train_data()
keras = tf.keras
layers = tf.keras.layers
# 将训练集转化为tensor
x_data = tf.convert_to_tensor(x_data)
# 训练集形状(7956, 257)
shape = x_data.shape
# (257,1)
inputs = layers.Input(shape=(shape[1], 1), name='inputs')
encode = layers.Flatten()(inputs)
encode = keras.layers.Dense(1024, activation='sigmoid')(encode)
# 加入 批标准化
encode = keras.layers.BatchNormalization()(encode)
encode = keras.layers.Dense(256, activation='sigmoid')(encode)
encode = keras.layers.BatchNormalization()(encode)
encode = keras.layers.Dense(257, activation='sigmoid')(encode)
encode_output = keras.layers.BatchNormalization()(encode)
decode = keras.layers.Dense(256, activation='sigmoid')(encode_output)
# 加入稀疏链接   比率为30%
decode = keras.layers.AlphaDropout(rate=0.3)(decode)
decode = keras.layers.Dense(1024, activation='sigmoid')(decode)
decode_output = keras.layers.Dense(shape[1], activation='sigmoid')(decode)

encoder = keras.Model(inputs, encode_output)
autoencoder = keras.Model(inputs, decode_output)
# 显示网络中数据
autoencoder.summary()

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=keras.losses.MeanSquaredError())
# 模型可视化
tf.keras.utils.plot_model(autoencoder, show_shapes=True)
# validation_split 测试集的比重
autoencoder.fit(x_data, x_data, batch_size=32, epochs=64)
# 保存
encoder.save_weights("./sava/encoder")
