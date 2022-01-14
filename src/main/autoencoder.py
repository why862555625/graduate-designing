import tensorflow as tf
from src.common.init_train_data import train_data

# 随机种子
tf.random.set_seed(0)
x_data, _, _ = train_data()
x_data = x_data
keras = tf.keras
layers = tf.keras.layers
# 将训练集转化为tensor   并转化为(0,1)
x_data = tf.convert_to_tensor(x_data) / 8000
# 标准化  =>   (0,1)
x_data = tf.reshape(x_data, shape=(-1, 16, 16))
x_data = tf.expand_dims(x_data, -1)
# 训练集形状(7956, 257)
shape = x_data.shape
print(shape)
#  (257,1)
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
# input = (2*2)
decode = keras.layers.Conv2DTranspose(32, 1, strides=1, padding="same", activation='relu')(encode_output)
decode = keras.layers.BatchNormalization()(decode)
# input = (2*2)
decode = keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same", activation='relu')(decode)
decode = keras.layers.BatchNormalization()(decode)
# input = (4*4)
decode = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation='relu')(decode)
decode = keras.layers.BatchNormalization()(decode)
# input = (8*8)
decode = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation='relu')(decode)
decode = keras.layers.BatchNormalization()(decode)
# input = (16*16)
decode = keras.layers.Conv2DTranspose(32, 3, strides=1, padding="same", activation='relu')(decode)
decode = keras.layers.BatchNormalization()(decode)
decode_output = keras.layers.Conv2DTranspose(1, 3, strides=1, padding="same", activation='sigmoid')(decode)

encoder = keras.Model(inputs, encode_output)
autoencoder = keras.Model(inputs, decode_output)
# 显示网络中数据
# autoencoder.summary()
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001,),
                    loss=keras.losses.MeanSquaredError())
# 模型可视化
tf.keras.utils.plot_model(autoencoder, show_shapes=True)
# validation_split 测试集的比重
autoencoder.fit(x_data, x_data, batch_size=64, epochs=64)
# 保存
encoder.save_weights("./sava/encoder")
