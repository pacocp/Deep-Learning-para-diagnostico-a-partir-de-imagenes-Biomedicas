#!/usr/bin/python3.6

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
import argparse
import pandas as pd

def train_model(model,BATCH_SIZE_TRAIN,NUM_EPOCHS,train_generator,validation_generator,steps_per_epoch,nb_validation_samples, df):
	#Using the early stopping technique to prevent overfitting
	earlyStopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
	print("Fitting the model")
	tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=1, write_graph=True, write_images=True)
	history = model.fit_generator(
			train_generator,
			samples_per_epoch=BATCH_SIZE_TRAIN,
			#callbacks=[earlyStopping],
			epochs=NUM_EPOCHS,
			steps_per_epoch=1,
			validation_data=validation_generator,
			validation_steps=nb_validation_samples,
			callbacks=[tbCallBack])

	print("Saving the weights")
	# always save your weights after training or during training
	model.save_weights('weights.h5')

	print("Saving in the file the training results...")
	# First we have to create the dictionary with the results
	data_of_training = {'BATCH_SIZE_TRAIN': BATCH_SIZE_TRAIN, 'STEPS_PER_EPOCH': steps_per_epoch,
						'NUM_EPOCHS': NUM_EPOCHS, 'ACCURACY_TRAIN': history.history['acc'],
						'VAL_ACC_TRAIN': history.history['val_acc'], 'LOSS_TRAIN': history.history['loss'],
						'VAL_LOSS_TRAIN': history.history['val_loss']}
	df_aux = pd.DataFrame(data=[data_of_training])
	df = df.append(df_aux)
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

	return df
