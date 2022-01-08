import tensorflow as tf
from src.common.init_train_data import train_data

tf.random.set_seed(0)
keras = tf.keras
layers = tf.keras.layers
x_data, y_data = train_data()
x_data = tf.convert_to_tensor(x_data)
y_data = tf.convert_to_tensor(y_data)

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(5, 1)))
model.add(keras.layers.Dense(1024, activation='sigmoid'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(256, activation='sigmoid'))
model.add(keras.layers.Dense(2, activation='sigmoid'))
model.add(keras.layers.Dense(256, activation='sigmoid'))
model.add(keras.layers.Dense(1024, activation='sigmoid'))
model.add(keras.layers.Dense(5, activation='sigmoid'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy')
# 模型可视化
tf.keras.utils.plot_model(model, show_shapes=True)
# validation_split 测试集的比重
model.fit(y_data, y_data, batch_size=32, epochs=200, )
model.save_weights("./sava/autoencoder")
