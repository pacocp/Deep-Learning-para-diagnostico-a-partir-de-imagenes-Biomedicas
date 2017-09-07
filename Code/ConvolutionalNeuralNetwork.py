#!/usr/bin/python3.6

'''
Francisco Carrillo Pérez (2017)
carrilloperezfrancisco@gmail.com
Universidad de Granada
TFG
'''

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
from train import train_model, train_model_CV, train_model_CV_generator, train_model_CV_MV, train_LOO
from train import train_LOO_pacient
from utils import read_images_and_labels, read_slices


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
train_data_dir = './Slices/alltheSlices/train'
test_data_dir = './Slices/alltheSlices/test'
train_data_dir_small = './dataset_small/train'
validation_data_dir_small = './dataset_small/validation'
test_data_dir_small = './dataset_small/test'
nb_train_samples = 2197 + 2822
nb_test_samples = 611 + 684
nb_train_samples_small = 52 + 55
nb_validation_samples_small = 6 + 25
nb_test_samples_small = 8 + 10
steps_per_epoch = nb_train_samples // 32


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
		train_data_dir,
		target_size=(110, 110),
		batch_size=32)
'''
validation_generator = test_datagen.flow_from_directory(
		validation_data_dir,
		target_size=(110, 110),
		shuffle=True)
'''
test_generator = test_datagen.flow_from_directory(
		test_data_dir,
		target_size=(110, 110),
		batch_size=32)


# Reading images and labels
#images1,labels,list_of_images1 = read_images_and_labels('dataset/')
images2,labels2,names = read_images_and_labels('Slices/dataset_slice_45')
#images3,labels3,list_of_images3 = read_images_and_labels('dataset_slice_65/')
#slices_images,slices_labels = read_slices('dataset_slice_65/')
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
	
	train_model(model,train_generator,steps_per_epoch)

	'''
	train_model_CV(slices_images,slices_labels)

	train_model_CV_generator(images,labels,model)

	train_model_CV_MV(images1,images2,images3,labels,model)

	train_LOO(images2,labels2)
	
	train_LOO_pacient(images2,labels2,names)
	'''
	'''
	i = 10
	f = open("Results_CV.txt","w")
	while i <= 100:
		print("Reading the slices "+str(i))
		path = "Slices/dataset_slice_"+str(i)+"/"
		slices_images,slices_labels = read_slices(path)
		slice_number = i
		print("Training with slices"+str(i))
		f.write("TRAINING MODEL WITH SLICE: "+str(i)+"\n")
		train_model_CV(slices_images,slices_labels,slice_number,f)
		f.write("END TRAINING MODEL WITH SLICE: "+str(i)+"\n")
		i = i + 5
	f.close()
	'''
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

	test_loss = model.evaluate_generator(test_generator, steps=nb_test_samples//32)
	print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
	print(model.metrics_names)
