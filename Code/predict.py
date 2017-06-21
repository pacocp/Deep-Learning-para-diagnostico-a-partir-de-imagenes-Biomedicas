#!/usr/bin/python3.5

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
import math
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
import keras
import PIL
import tempfile
import shutil
# Imports from my files
from model import create_model
from save_load_results import create_temp_dir,delete_temp_dir
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

'''Options for performing training, restore a model or test'''
restore = True

def predict_class(inputFile):
		'''
		Predict the class for a new image that the user choose.
		'''
		img = image.load_img(inputFile,target_size=(166, 256))
		img_predict = image.img_to_array(img)
		img_predict = np.expand_dims(img_predict,axis=0)
		#creating the model
		model = create_model()
		model.load_weights('weights.h5')
		print("augmentation configuration for testing")
		# Then evaluating in the data
		print("Evaluating in test data...")
		predictions = model.predict_classes(img_predict)
		print("Classes predict are: ")
		print(predictions)
		return predictions
