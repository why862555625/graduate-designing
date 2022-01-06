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
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(2, activation='relu'))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(1024, activation='relu'))
model.add(keras.layers.Dense(251, activation='relu'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              )
# 模型可视化
tf.keras.utils.plot_model(model, show_shapes=True)
# validation_split 测试集的比重
model.fit(x_data, x_data, batch_size=1, epochs=200, )
model.save_weights("./sava/autoencoder")
