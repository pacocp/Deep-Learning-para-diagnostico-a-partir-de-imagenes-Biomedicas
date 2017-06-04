#!/usr/bin/python3.5

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import keras
import PIL

"""
Different parameters that allow to change variables of the whole network.

@param: BATCH SIZE_TRAIN, is going to depend on the number of samples that we have
@param: IMAGE_HEIGHT, height of the images
@param: IMAGE_WIDTH, width of the images
@param: IMAGE_WIDTH_ORIGINAL, original width of the SEM images
@param: IMAGE_HEIGHT_ORIGINAL, original height of the SEM images
@param: dimension_first_conv, how many dimensions do you want in the first convolutional layer of the network
@param: dimension_second_conv, how many dimensions do you want in the second convolutional layer of the network
@param: dimension_fc, how many neurons do you want in the fully conected layer

"""

BATCH_SIZE_TRAIN = 2000
NUM_EPOCHS = 3
IMAGE_HEIGHT =29
IMAGE_WIDTH = 29
dimension_first_conv = 16
dimension_second_conv = 32
dimension_fc = 64

'''
Full path of the different directories for loading the dataset to the network, and also values of the dataset.

@param: nb_train_samples, number of train samples
@param: nb_validation_samples, number of validation samples
@param: nb_test_samples, number of test samples

'''
nb_train_samples = 86 + 100 + 197
nb_validation_samples = 18 + 21 + 42
nb_test_samples = 19 + 21 + 43

'''Options for performing training, restore a model or test'''
restore = True

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
def model():
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

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Test the neural network that ")
	parser.add_argument("-i", "--inputFile",
						help="input file",
						dest='inputFile')
	parser.add_argument("-a", "--allImages",
						help="all images in the folder(0,1)",
						dest='allImages')
    parser.add_argument("-d", "--directory",
                        help="directory where the images are"
                        dest='test_data_dir')

	args = parser.parse_args()

	inputFile = args.inputFile
	allImages = args.allImages
    test_data_dir = args.test_data_dir

    '''
    Restoring the weights, the name could be changed depending the name of out file,
    but they are saved as weights.h5
    '''
    #creating the model
    model = model()
    model.load_weights('weights.h5')
    print("augmentation configuration for testing")
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(166, 256),
            class_mode='binary')
    # Then evaluating in the data
    print("Evaluating in test data...")
    predictions = model.predict(test_generator)
    print("Classes predict are: ")
    print(predictions)
