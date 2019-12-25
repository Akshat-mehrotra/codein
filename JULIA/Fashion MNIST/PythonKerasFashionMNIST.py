#
import pickle

import keras
from keras.datasets import fashion_mnist
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt



(train_x,train_y), (test_x,test_y) = fashion_mnist.load_data()
train_x = train_x.reshape(-1,28*28)
test_x = test_x.reshape(-1,28*28)

model = keras.Sequential([
    keras.layers.Dense(28*28, input_shape=(28*28,), activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(32, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=30)

test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)

keras.models.save_model(
    model,
    "./KerasFinalModel"
)

with open("history.pickle","wb") as h:
    pickle.dump((history.history,test_acc), h)
