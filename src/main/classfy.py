import tensorflow as tf
from src.common.init_train_data import train_data

tf.random.set_seed(0)
keras = tf.keras
layers = tf.keras.layers
x_data, y_data, _ = train_data()
# 将训练集转化为tensor
x_data = tf.convert_to_tensor(x_data)
# 输入形状(7956, 257)
shape = x_data.shape
# 将标签转化为tensor
y_data = tf.convert_to_tensor(y_data)
# 构建神经网络网络形状
model = keras.Sequential()
# (256,1)
model.add(layers.Flatten(input_shape=(shape[1], 1)))
model.add(keras.layers.Dense(1024, activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(512, activation='sigmoid'))
model.add(keras.layers.AlphaDropout(rate=0.1))
model.add(keras.layers.Dense(256, activation='sigmoid'))
model.add(keras.layers.Dense(64, activation='sigmoid'))
model.add(keras.layers.Dense(5, activation='softmax'))
model.summary()
# 使用tensorboard进行 数据可视化
log_dir = "E:\BIYE\LOSS\Autoencoder"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              # Accuracy 准确率   Recall召回率
              metrics=[
                  'accuracy',
                  'Recall',
              ]
              )
# 模型可视化
tf.keras.utils.plot_model(model, show_shapes=True)
# validation_split 测试集的比重
# epochs 训练总次数  batch_size ： 每个epochs训练的样本数量
# callbacks 如果连续3个批次loss没有下降  就提前结束
model.fit(x_data, y_data, batch_size=64, epochs=64, callbacks=[tensorboard_callback], validation_split=0.15)
