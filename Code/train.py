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




from model import create_model
from utils import reorderRandomly,calculate_mean

def train_model(model,BATCH_SIZE_TRAIN,NUM_EPOCHS,train_generator,steps_per_epoch):

	#Using the early stopping technique to prevent overfitting
	earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
	print("Fitting the model")
	#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
	history = model.fit_generator(
			train_generator,
			steps_per_epoch= steps_per_epoch,
			#callbacks=[earlyStopping],
			epochs=50)

	print("Saving the weights")
	# always save your weights after training or during training
	model.save_weights('weights.h5')


def train_model_CV(slices_images,slices_labels):
	'''
	Training model using cross-validation

	Parameters
	----------

	'''

	#images,labels,list_of_images = reorderRandomly(images,labels,list_of_images)
	'''
	for i in range(len(labels)):
		if labels[i] == "AD":
			labels[i] = 0
		else:
			labels[i] = 1

	slices_images = [images[i::5] for i in range(5)]
	slices_list_of_images = [list_of_images[i::5] for i in range(5)]
	slices_labels = [labels[i::5] for i in range(5)]

	print(slices_list_of_images)
	'''
	models = {}
	histories = {}
	values_acc = []
	print(len(slices_labels[0]))
	print(len(slices_labels[1]))
	print(len(slices_labels[2]))
	print(len(slices_labels[3]))
	print(len(slices_labels[4]))
	print(len(slices_images[0]))
	print(len(slices_images[1]))
	print(len(slices_images[2]))
	print(len(slices_images[3]))
	print(len(slices_images[4]))
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
		history = model.fit(X_train,Y_train,epochs=50,batch_size=5)
		models['model'+str(i)] = model
		test_loss = model.evaluate(X_test,Y_test)
		print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
		histories['test_acc'+str(i)] = test_loss
		values_acc.append(test_loss[1])

	mean = calculate_mean(values_acc)
	print("The mean of all the test values is: %g"%mean)

def train_model_CV_generator(images,labels,model,train_datagen):
	'''
	Training model using cross-validation

	Parameters
	----------

	'''
	train_datagen = ImageDataGenerator(horizontal_flip=True)
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
	values_acc = []
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
		history = model.fit_generator(train_datagen.flow(X_train,Y_train,batch_size = 5),
		epochs=70,steps_per_epoch= len(X_train)//5)
		models['model'+str(i)] = model
		test_loss = model.evaluate(X_test,Y_test)
		print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
		histories['test_acc'+str(i)] = test_loss
		values_acc.append(test_loss[1])

	mean = calculate_mean(values_acc)
	print("The mean of all the test values is: %g"%mean)

def train_model_CV_MV(images1,images2,images3,labels,model):
	'''
	Training model using cross-validation

	Parameters
	----------

	'''
	f = open('results_cross_validation.txt','w')
	labels_aux = labels
	images1,labels = reorderRandomly(images1,labels)
	images2,labels_aux = reorderRandomly(images2,labels_aux)
	images3,labels_aux = reorderRandomly(images3,labels_aux)

	for i in range(len(labels)):
		if labels[i] == "AD":
			labels[i] = 0
		else:
			labels[i] = 1

	slices_images1 = [images1[i::5] for i in range(5)]
	slices_images2 = [images2[i::5] for i in range(5)]
	slices_images3 = [images3[i::5] for i in range(5)]
	slices_labels = [labels[i::5] for i in range(5)]

	models = {}
	histories = {}
	values_acc = []
	for i in range(5):
		f.write("Results for the fold: "+str(i))
		f.write("\n")
		model1 = create_model()
		model2 = create_model()
		model3 = create_model()
		X_test1 = slices_images1[i]
		X_test2 = slices_images2[i]
		X_test3 = slices_images3[i]
		Y_test = slices_labels[i]
		X_train1 = [item
					for s in slices_images1 if s is not X_test1
					for item in s]
		X_train2 = [item
					for s in slices_images2 if s is not X_test2
					for item in s]
		X_train3 = [item
					for s in slices_images3 if s is not X_test3
					for item in s]
		Y_train = [item
					for s in slices_labels if s is not Y_test
					for item in s]

		X_train1 = np.array(X_train1)
		X_train2 = np.array(X_train2)
		X_train3 = np.array(X_train3)
		Y_train = np.array(Y_train)
		X_test1 = np.array(X_test1)
		X_test2 = np.array(X_test2)
		X_test3 = np.array(X_test3)
		Y_test = np.array(Y_test)
		from keras.utils.np_utils import to_categorical
		Y_train = to_categorical(Y_train)
		Y_test = to_categorical(Y_test)
		history1 = model1.fit(X_train1,Y_train,epochs=70,batch_size=5)
		history2 = model2.fit(X_train2,Y_train,epochs=70,batch_size=5)
		history3 = model3.fit(X_train3,Y_train,epochs=70,batch_size=5)
		#models['model'+str(i)] = model
		prediction_1 = model1.predict_classes(X_test1)
		prediction_2 = model2.predict_classes(X_test2)
		prediction_3 = model3.predict_classes(X_test3)

		count = 0
		for index in range(len(prediction_1)):

			vote = int(prediction_1[index]+prediction_2[index]+prediction_3[index])

			if(vote >= 2):
				major_vote = 1
			else:
				major_vote = 0

			if(int(Y_test[index][0]) == 1):
				true = 0
			else:
				true = 1
			f.write("The value of the prediction is: " + str(major_vote) +"and the true value is: "+str(true))
			f.write("\n")
			if(major_vote == true):
				count = count + 1

		values_acc.append(count/len(Y_test))

	mean = calculate_mean(values_acc)
	print("The mean of all the test values is: %g"%mean)
	f.close()

def train_LOO(images,labels):
	values_acc = []
	for i in range(len(labels)):
		if labels[i] == "AD":
			labels[i] = 0
		else:
			labels[i] = 1
	print("The lenght of images is "+str(len(images)))
	for i in range(len(images)):
		model = create_model()
		X_test = []
		Y_test = []
		X_test.append(images[i])
		Y_test.append(labels[i])
		X_train = []
		Y_train = []
		for j in range(len(images)):
			if j != i:
				X_train.append(images[j])
				Y_train.append(labels[j])


		X_train = np.array(X_train)
		Y_train = np.array(Y_train)
		X_test = np.array(X_test)
		Y_test = np.array(Y_test)
		from keras.utils.np_utils import to_categorical
		Y_train = to_categorical(Y_train,2)
		Y_test = to_categorical(Y_test,2)


		history = model.fit(X_train,Y_train,epochs=70,batch_size=10)
		test_loss = model.evaluate(X_test,Y_test)
		print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
		values_acc.append(test_loss[1])

	mean = calculate_mean(values_acc)
	print("The mean of all the test values is: %g"%mean)

def train_LOO_pacient(images,labels,names):
	print("Training LOO with pacient name")
	values_acc = []
	already_tested = []
	for i in range(len(labels)):
		if labels[i] == "AD":
			labels[i] = 0
		else:
			labels[i] = 1
	print("The lenght of images is "+str(len(images)))
	for i in range(len(images)):
		model = create_model()
		X_test = []
		Y_test = []
		# If we haven't tested that pacient already
		# we insert it to the test set and also
		# all the images with the same name
		if(not(names[i] in already_tested)):
			already_tested.append(names[i])
			X_test.append(images[i])
			Y_test.append(labels[i])
			for j in range(len(images)):
				if j!=i and names[j] == names[i]:
					X_test.append(images[j])
					Y_test.append(labels[j])
			X_train = []
			Y_train = []
			for j in range(len(images)):
				if names[j] != names[i]:
					X_train.append(images[j])
					Y_train.append(labels[j])

			X_train = np.array(X_train)
			Y_train = np.array(Y_train)
			X_test = np.array(X_test)
			Y_test = np.array(Y_test)
			from keras.utils.np_utils import to_categorical
			Y_train = to_categorical(Y_train,2)
			Y_test = to_categorical(Y_test,2)
			print("The len of the training set is: "+str(len(X_train)))
			print("The len of the test set is: "+str(len(X_test)))

			history = model.fit(X_train,Y_train,epochs=70,batch_size=10)
			test_loss = model.evaluate(X_test,Y_test)
			print("Loss and accuracy in the test set: Loss %g, Accuracy %g"%(test_loss[0],test_loss[1]))
			values_acc.append(test_loss[1])

	mean = calculate_mean(values_acc)
	print("The mean of all the test values is: %g"%mean)
