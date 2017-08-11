#!/usr/bin/python3.6

# Libraries that I need
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
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier


from model import create_model
from utils import reorderRandomly

def train_model(model,BATCH_SIZE_TRAIN,NUM_EPOCHS,train_generator,validation_generator,steps_per_epoch,nb_validation_samples, nb_train_samples):

	#Using the early stopping technique to prevent overfitting
	earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
	print("Fitting the model")
	#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
	history = model.fit_generator(
			train_generator,
			steps_per_epoch= 2,
			#callbacks=[earlyStopping],
			epochs=NUM_EPOCHS,
			validation_data=validation_generator,
			validation_steps=nb_validation_samples)

	print("Saving the weights")
	# always save your weights after training or during training
	model.save_weights('weights.h5')


def train_model_CV(images,labels,model,BATCH_SIZE_TRAIN,NUM_EPOCHS,train_generator,validation_generator,steps_per_epoch,nb_validation_samples, nb_train_samples):
	'''
	Training model using cross-validation

	Parameters
	----------

	'''
	
	images,labels = reorderRandomly(images,labels)

	for i in range(len(labels)):
		if labels[i] == "AD":
			labels[i] = 0
		else:
			labels[i] = 1

	slices_images = [images[i::5] for i in range(5)]
	slices_labels = [labels[i::5] for i in range(5)]

	models = {}
	histories = {}
	for i in range(5):
		model = create_model()
		X_test = slices_images[i]
		Y_test = slices_labels[i]
		X_train = [item
					for s in slices_images if s is not X_test
                    for item in s]
		Y_train = [item
					for s in slices_labels if s is not Y_test
                    for item in s]

		X_train = np.array(X_train)
		Y_train = np.array(Y_train)
		X_test = np.array(X_test)
		Y_test = np.array(Y_test)
		from keras.utils.np_utils import to_categorical
		Y_train = to_categorical(Y_train)
		Y_test = to_categorical(Y_test)
		history = model.fit(X_train,Y_train,epochs=50,batch_size=5,verbose=2)
		models['model'+str(i)] = model
		test_loss = model.evaluate(X_test,Y_test)
		print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
		histories['test_acc'+str(i)] = test_loss
		if (i != 0):
			hist_ant = histories['test_acc'+str(i-1)]
		if(i == 0):
			best_model = model
		elif (test_loss[1] > hist_ant[1]):
			best_model = model

	X_all = np.array(images)
	Y_all = np.array(labels)
	Y_all = to_categorical(Y_all)
	print("Training the best model in the whole dataset")
	history = best_model.fit(X_all,Y_all,epochs=50,batch_size=5)
	print("Saving the weights")
	# always save your weights after training or during training
	model.save_weights('weights.h5')
	return best_model
	#Y = labels[0:8]
	#model.fit(X,Y,epochs=5)
