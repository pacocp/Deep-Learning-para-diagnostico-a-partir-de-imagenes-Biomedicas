#!/usr/bin/python3.6

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy
import math
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import keras
import PIL

"""
Different parameters that allow to change variables of the whole network.

@param: dimension_first_conv, how many dimensions do you want in the first convolutional layer of the network
@param: dimension_second_conv, how many dimensions do you want in the second convolutional layer of the network
@param: dimension_fc, how many neurons do you want in the fully conected layer

"""

dimension_first_conv = 16
dimension_second_conv = 32
dimension_fc = 64

'''
Creation of the architecture for the CNN.

- One convolutional layer
- Activation ReLU.
- Max Pooling (2,2)
- One convolutional layer
- Activation ReLU
- Max Pooling (2,2)
- One convolutional layer
- Activation Relu
- Max Pooling (2,2)
- Fully conected layer
- Dropout layer
- Output layer
- Activation sigmoid

'''

def create_model():
    print("Creating the model")

    print("creating first layer")
    model = Sequential()
    model.add(Conv2D(dimension_first_conv,(10, 10), input_shape=(166, 256,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    print("creating second layer")
    model.add(Conv2D(dimension_second_conv,(10, 10)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    print("creating third layer")
    model.add(Conv2D(dimension_fc, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    print("creating output layer")
    model.add(Flatten())
    model.add(Dense(dimension_fc))
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    return model
