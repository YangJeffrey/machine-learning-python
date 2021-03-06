import tensorflow as tf 
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(128, activation='relu'))
model.add(tf.keras.layer.Dropout(0.2))

model.add(tf.keras.layer.Dense(32, activation='relu'))
model.add(tf.keras.layer.Dropout(0.2))

model.add(tf.keras.layer.Dense(10, activation='softmax')) #how many classes (hard coded)

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

#mean_squared_error
model.compile(loss='sparse_categorical_crossentropy',
    optimizer = opt,
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
