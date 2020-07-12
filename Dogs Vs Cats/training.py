from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import numpy as np

pickle_in = open("X_train.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y = pickle.load(pickle_in)

X = np.array(X)
X = X/255.0

y = np.array(y)

dense_layers = 0
conv_layers = 3
layer_size = 64

NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layers, layer_size, dense_layers, int(time.time()))

model = Sequential()

model.add(Conv2D(layer_size, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

for x in range(conv_layers-1):
    model.add(Conv2D(layer_size, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

for x in range(dense_layers):
    model.add(Dense(layer_size))
    model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(X, y,
    batch_size=32,
    epochs=10,
    validation_split=0.3,
    callbacks=[tensorboard])

model.save('64x3-CNN.model')