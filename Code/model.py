#!/usr/bin/python3.6

'''
Francisco Carrillo Pérez (2017)
carrilloperezfrancisco@gmail.com
Universidad de Granada
TFG
'''

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy
import math
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, merge
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Nadam
import keras
import PIL
from keras.regularizers import l2

def create_model():
    '''
    Method for building the keras model used in the experiments

    Parameters
    ------------
    None
    
    Returns
    ------------
    model: keras model
    '''
    print("Creating the model")

    model = Sequential()

    model.add(Conv2D(8, (3, 3), input_shape=(110, 110,3)))
    model.add(Activation("relu"))

    model.add(Conv2D(8, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(16, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=1))

    model.add(Dropout(0.7))

    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dense(2))

    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
      optimizer=keras.optimizers.Adam(lr=27*1e-04,clipnorm=1., clipvalue=0.5),
      metrics=['accuracy'])
    return model

'''
The following implementation of the residual block is from:
https://github.com/relh/keras-residual-unit/blob/master/residual.py
'''

def conv_block(feat_maps_out, prev):
    prev = BatchNormalization(axis=1)(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, 3, 3, border_mode='same')(prev)
    prev = BatchNormalization(axis=1)(prev) # Specifying the axis and mode allows for later merging
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, 3, 3, border_mode='same')(prev)
    return prev


def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv2D(feat_maps_out, 1, 1, border_mode='same')(prev)
    return prev


def Residual(feat_maps_in, feat_maps_out, prev_layer):
    '''
    A customizable residual unit with convolutional and shortcut blocks
    Args:
      feat_maps_in: number of channels/filters coming in, from input or previous layer
      feat_maps_out: how many output channels/filters this block will produce
      prev_layer: the previous layer
    '''

    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)

    print('Residual block mapping '+str(feat_maps_in)+' channels to '+str(feat_maps_out)+' channels built')
    return merge([skip, conv], mode='sum') # the residual connection

def create_model_RES():

    inp = Input((110, 110, 3))
    cnv1 = Conv2D(64, 3, 3, subsample=[2,2], activation='relu', border_mode='same')(inp)
    r1 = Residual(64, 64, cnv1)
    # An example residual unit coming after a convolutional layer. NOTE: the above residual takes the 64 output channels
    # from the Convolutional2D layer as the first argument to the Residual function
    r2 = Residual(64, 64, r1)
    cnv2 = Conv2D(64, 3, 3, activation='relu', border_mode='same')(r2)
    r3 = Residual(64, 64, cnv2)
    r4 = Residual(64, 64, r3)
    cnv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same')(r4)
    r5 = Residual(128, 128, cnv3)
    r6 = Residual(128, 128, r5)
    maxpool = MaxPooling2D(pool_size=(7, 7))(r6)
    flatten = Flatten()(maxpool)
    dense1 = Dense(128, activation='relu')(flatten)
    out = Dense(2, activation='softmax')(dense1)

    model = Model(input=inp, output=out)
    model.compile(loss='categorical_crossentropy',
    optimizer=Nadam(lr=1e-4), metrics=['accuracy'])

    return model
