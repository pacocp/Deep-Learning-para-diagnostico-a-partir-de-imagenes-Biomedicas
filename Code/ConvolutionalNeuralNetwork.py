#!/usr/bin/python3.5

# Libraries that I need
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy
import math
import matplotlib
matplotlib.use('TkAgg') # This is for being able to use a virtualenv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import keras
import PIL

# My own files's imports
from model import create_model

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

@param: train_data_dir, full path where the training samples are
@param: validation_data_dir, full path where the validation samples are
@param: test_data_dir, full path where the validation samples are
@param: nb_train_samples, number of train samples
@param: nb_validation_samples, number of validation samples
@param: nb_test_samples, number of test samples

The order for the summation is: ad mci normal
'''
train_data_dir = './dataset/train'
validation_data_dir = './dataset/validation'
test_data_dir = './dataset/test'
nb_train_samples = 149 + 443 + 269
nb_validation_samples = 22 + 94 + 63
nb_test_samples = 21 + 96 + 62

'''Options for performing training, restore a model or test'''
restore = True
train = False
test = True

'''
For visualizing the model

@param: mode, the model we have declared above
@param: to_file, name of the file is going to be saved to

'''
'''
from keras.utils.visualize_util import plot
print("Plotting the model")
plot(model, to_file='model.png')
'''

print("augmentation configuration for training")
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
print("augmentation configuration for testing")
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(166, 256),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(166, 256),
        batch_size=18,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(166, 256),
        batch_size=18,
        class_mode='binary')

'''
Restoring the weights, the name could be changed depending the name of out file, but they are saved as weights.h5
'''
#creating the model
model = create_model()
if(restore == True):
    model.load_weights('weights.h5')

if(train == True):
    #Using the early stopping technique to prevent overfitting
    earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
    print("Fitting the model")
    history = model.fit_generator(
            train_generator,
            samples_per_epoch=BATCH_SIZE_TRAIN,
            #callbacks=[earlyStopping],
            epochs=NUM_EPOCHS,
            steps_per_epoch=1,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples)

    print("Saving the weights")
    # always save your weights after training or during training
    model.save_weights('weights.h5')

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

if(test == True):
    print("Evaluating in test data...")
    test_loss = model.evaluate_generator(test_generator,steps = nb_test_samples)
    print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
