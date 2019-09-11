import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow import keras
import pickle
import time

DATADIR = "" #data directory 
CATEGORIES = [""] #categories array list
EPOCHS = 10
BATCH_SIZE = 1
VALIDATION_SPLIT = 0.1



for category in CATEGORIES:
  path = os.path.join(DATADIR, category)
  for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
    break
  break

IMG_SIZE = 50
new_array = cv2.cvtColor(cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)

training_data = []
def create_training_data():
    for category in CATEGORIES:
      path = os.path.join(DATADIR, category)
      class_num = CATEGORIES.index(category)
      for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_num])
create_training_data() 

X = []
y = []

for features, label in training_data:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

dense_layers = [] #integer number of dense layers
layer_sizes = [] #integer layer sizes
conv_layers = [] #integer number of convolutional layers

for dense_layer in dense_layers:
  for layer_size in layer_sizes:
    for conv_layer in conv_layers:
      NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
      tensorboard = tf.keras.callbacks.TensorBoard(log_dir='C:\\logs\\{}'.format(NAME))
      print(NAME)
      model = tf.keras.models.Sequential()

      # customize layers for optiminization
      model.add(tf.keras.layers.Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
      model.add(tf.keras.layers.Activation("relu"))
      model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

      for l in range(conv_layer-1):
        model.add(tf.keras.layers.Conv2D(layer_size, (3,3)))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

      model.add(tf.keras.layers.Flatten()) #3D feature maps to 1D feature wectors
      for l in range(dense_layer):
        model.add(tf.keras.layers.Dense(layer_size))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Dropout(0.2))

      model.add(tf.keras.layers.Dense(1))
      model.add(tf.keras.layers.Activation('sigmoid'))

      model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=['accuracy'])

      model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, callbacks=[tensorboard])

model.save('convolutionalneuralnetwork.model')
