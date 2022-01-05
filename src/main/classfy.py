import tensorflow as tf
from src.common.init_train_data import train_data

x_data, y_data = train_data()

# x_data = tf.convert_to_tensor(x_data)
# y_data = tf.convert_to_tensor(y_data)

layers = tf.keras.layers
inputs = layers.Input(shape=(40, 1), name='inputs')
D1 = layers.Dense(256, activation='relu')(inputs)
D2 = layers.Dense(128, activation='relu')(D1)
D3 = layers.Dense(64, activation='relu')(D2)
D4 = layers.Dense(5, activation='softmax')(D3)
Model = tf.keras.Model(inputs, D4)
Model.summary()
log_dir = "E:\BIYE\LOSS\classfy"
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()
x_data = iter(x_data)
y_data = iter(y_data)

for epoch in range(14000):

    batch_x = tf.convert_to_tensor(next(x_data))
    print(batch_x)
    batch_y = tf.convert_to_tensor(next(y_data))
    print(batch_y)

    with tf.GradientTape() as tape:
        y_train = Model(batch_x)
        loss_value = loss(y_train, batch_y)
        print(loss_value)
    grads = tape.gradient(loss_value, Model.trainable_variables)
    optimizer.apply_gradients(zip(grads, Model.trainable_variables))
    if epoch % 100 == 0:
        print(epoch, 'loss_10:', float(loss_value))
Model.save_weights('./save/classify')
