import tensorflow as tf
from src.common.init_train_data import train_data

tf.random.set_seed(0)
keras = tf.keras
layers = tf.keras.layers

x_data, y_data = train_data()

x_data = tf.convert_to_tensor(x_data)
y_data = tf.convert_to_tensor(y_data)

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(251, 1)))
model.add(keras.layers.Dense(1024, activation='sigmoid'))
model.add(keras.layers.Dense(512, activation='sigmoid'))
model.add(keras.layers.Dense(256, activation='sigmoid'))
model.add(keras.layers.Dense(5, activation='sigmoid'))
model.summary()

log_dir = "E:\BIYE\LOSS\Autoencoder"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy']
              )
# 模型可视化
tf.keras.utils.plot_model(model, show_shapes=True)
# validation_split 测试集的比重
model.fit(x_data, y_data, batch_size=2, epochs=200, callbacks=[tensorboard_callback], validation_split=0.15)

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
