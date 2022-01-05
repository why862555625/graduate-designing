import tensorflow as tf
from src.common.init_train_data import train_data

x_data, y_data = train_data()

x_data = tf.convert_to_tensor(x_data)
y_data = tf.convert_to_tensor(y_data)
print(x_data)
print(y_data)
layers = tf.keras.layers
inputs = layers.Input(shape=(40,1), name='inputs')
D1 = layers.Dense(256, activation='relu')(inputs)
D2 = layers.Dense(128, activation='relu')(D1)
D3 = layers.Dense(64, activation='relu')(D2)
D4 = layers.Dense(5, activation='softmax')(D3)
Model = tf.keras.Model(inputs, D4)
Model.summary()


log_dir = "E:\BIYE\LOSS\Autoencoder"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

Model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy())
# 模型可视化
tf.keras.utils.plot_model(Model, show_shapes=True)
# 提前结束训练      patience: 没有进步的训练轮数，在这之后训练就会被停止。           monitor: 被监测的数据。
# early_stop =tf. keras.callbacks.EarlyStopping(patience=2, monitor='loss')
# batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
# epochs训练总轮数
# validation_freq如果为整数，则指定在执行新的验证运行之前要运行多少个训练时期，例如validation_freq=2，每2个时期运行一次验证。
Model.fit(x_data, y_data, batch_size=2, epochs=15, callbacks=[tensorboard_callback])

# log_dir = "E:\BIYE\LOSS\classfy"
# optimizer = tf.keras.optimizers.Adam()
# loss = tf.keras.losses.BinaryCrossentropy()
# x_data = iter(x_data)
# y_data = iter(y_data)
# for epoch in range(14000):
#
#     batch_x = tf.convert_to_tensor(next(x_data))
#     print(batch_x)
#     batch_y = tf.convert_to_tensor(next(y_data))
#     print(batch_y)
#
#     with tf.GradientTape() as tape:
#         y_train = Model(batch_x)
#         loss_value = loss(y_train, batch_y)
#         print(loss_value)
#     grads = tape.gradient(loss_value, Model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, Model.trainable_variables))
#     if epoch % 100 == 0:
#         print(epoch, 'loss_10:', float(loss_value))
# Model.save_weights('./save/classify')
