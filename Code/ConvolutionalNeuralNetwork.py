#!/usr/bin/python3.6

# For seeing the training tensorboard --logdir Graph

# Libraries that I need

import theano
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np
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
import argparse
import pandas as pd
from keras.utils.np_utils import to_categorical

# My own files's imports
from model import create_model, create_model_RES
from train import train_model, train_model_CV, train_model_CV_generator, train_model_CV_MV
from save_load_results import create_file,read_from_file,write_to_file
from utils import read_images_and_labels

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

BATCH_SIZE_TRAIN = 5
NUM_EPOCHS = 150
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
+ 443
+ 94
+ 96
'''
train_data_dir = './dataset_separation/train'
test_data_dir = './dataset_separation/test'
train_data_dir_small = './dataset_small/train'
validation_data_dir_small = './dataset_small/validation'
test_data_dir_small = './dataset_small/test'
nb_train_samples = 122 + 190
nb_validation_samples = 23 + 35
nb_test_samples = 34 + 38
nb_train_samples_small = 52 + 55
nb_validation_samples_small = 6 + 25
nb_test_samples_small = 8 + 10
steps_per_epoch = nb_train_samples // 5
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
train_datagen = ImageDataGenerator(horizontal_flip=True)
print("augmentation configuration for testing")
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		target_size=(110, 110),
		batch_size=5,
		shuffle=True)
'''
validation_generator = test_datagen.flow_from_directory(
		validation_data_dir,
		target_size=(110, 110),
		shuffle=True)
'''
test_generator = test_datagen.flow_from_directory(
		test_data_dir,
		target_size=(110, 110),
		shuffle=True)

print("DATA GENERATOR----------------------")
print(train_generator.class_indices)

# Reading images and labels
images1,labels = read_images_and_labels('dataset/')
images2,labels2 = read_images_and_labels('dataset_slice_45/')
images3,labels3 = read_images_and_labels('dataset_slice_65/')
#test_images,test_labels = read_images_and_labels('./dataset/test/ad/metadata.csv')
if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Train convolutional neural network for predicting NIFTI images")
	parser.add_argument("-tr", "--train",
						help="if you want to train the network(0,1)",
						default="false",
						dest='train')
	parser.add_argument("-r", "--restore",
						help="if you want to restore the weights(0,1)",
						default="false",
						dest='restore')
	parser.add_argument("-tst", "--test",
						help="if you want to test the network(0,1)",
						default="false",
						dest='test')
	parser.add_argument("-o", "--outputFile",
						help="outputname for saving the experiments results",
						default="",
						dest='name_of_file')

	args = parser.parse_args()
	train = args.train
	restore = args.restore
	test = args.test
	name_of_file = args.name_of_file


'''
# Trying to open the results file, if it doesn't exist create it
try:
	df_experiments = read_from_file(name_of_file)
except:
	# Creating the columns for the dataframe
	columns = ['ID','BATCH_SIZE_TRAIN','STEPS_PER_EPOCH','NUM_EPOCHS','ACCURACY_TRAIN','VAL_ACC_TRAIN','LOSS_TRAIN','VAL_LOSS_TRAIN','VAL_ACC_TEST','VAL_LOSS_TEST']
	# Creating the file for the dataframe
	create_file(columns,name_of_file)
	# Opening the experiments file
	df_experiments = read_from_file(name_of_file)
'''
#creating the model
model = create_model()
if(restore == "true"):
	'''
	Restoring the weights, the name could be changed depending the name of out file, but they are saved
	as weights.h5
	'''
	model.load_weights('weights.h5')
if(train == "true"):

	#Training the model defining all the parameters. The method could be found in train.py file.

	print("Training the model...")
	'''
	train_model(model,BATCH_SIZE_TRAIN,NUM_EPOCHS,train_generator,steps_per_epoch)

	train_model_CV(images,labels,model,train_generator)

	train_model_CV_generator(images,labels,model)
	'''
	train_model_CV_MV(images1,images2,images3,labels,model)
if(test == "true"):
	'''
	for i in range(len(test_labels)):
		if test_labels[i] == "AD":
			test_labels[i] = 0
		else:
			test_labels[i] = 1
	print("Evaluating in test data...")
	test_X = np.array(test_images)
	test_Y = np.array(test_labels)
	test_Y = to_categorical(test_Y)
	test_loss = model.evaluate(test_X,test_Y)
	print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
	# Writting it to the dataframe
	'''

	test_loss = model.evaluate_generator(test_generator, steps=1)
	print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
	print(model.metrics_names)
