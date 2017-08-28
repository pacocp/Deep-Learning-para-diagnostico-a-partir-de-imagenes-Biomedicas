#!/usr/bin/env python

'''
Francisco Carrillo Pérez (2017)
carrilloperezfrancisco@gmail.com
Universidad de Granada
TFG
'''

import nibabel
import numpy as np
import pandas as pd
from tqdm import tqdm
import PIL
import cv2
from sklearn.utils import shuffle
from imread import imread
import itertools

def read_images_and_labels(path):
	'''
	Reading the images and the labels.

	Parameters
	-----------
	path: string. Path contanining the images

	Returns
	-------
	numpy.ndarray, numpy.ndarray
		images, and labels
	'''

	metadata = pd.read_csv(path+'metadata.csv')
	print("Reading labels...")
	smc_mask = ((metadata.Label == "AD") |
				(metadata.Label == "Normal")).values.astype('bool')
	#data = np.zeros(shape=(smc_mask.sum(), 110, 110,3), dtype='float32')
	list_of_images = metadata['Path']
	names = []
	for name in list_of_images:
		name_split = name.split("_")
		names.append(name_split[3])
	array_list = []
	print("Reading images...")
	#for it, im in tqdm(enumerate(metadata[smc_mask].Path.values),
					  # total=smc_mask.sum(), desc='Reading MRI to memory'):
	for im in list_of_images:

		img = cv2.imread(str(path+im))
		print(img.shape)
		#img = np.arange(1 * 3 * 110 * 110).reshape((110, 110, 3))
		#img = img.astype('float32')

		img_resize=cv2.resize(img,(110,110))
		array_list.append(img)
		#data[it, :, :, :] = data_img

	print(len(array_list))
	labels = metadata["Label"]

	return array_list,labels,names

def read_slices(path):
	'''
	Reading the slices of images and labels to perform 5-Fold Cross Validation

	Parameters
	-----------
	path: string. Path contanining the images

	Returns
	-------
	list of numpy.array, list of numpy.array
		slices_of_images, and slices_of_labels
	'''

	metadata_slice1 = pd.read_csv(path+"sep1.csv")
	metadata_slice2 = pd.read_csv(path+"sep2.csv")
	metadata_slice3 = pd.read_csv(path+"sep3.csv")
	metadata_slice4 = pd.read_csv(path+"sep4.csv")
	metadata_slice5 = pd.read_csv(path+"sep5.csv")

	print("Reading labels...")

	#data = np.zeros(shape=(smc_mask.sum(), 110, 110,3), dtype='float32')
	list_of_images1 = metadata_slice1['Path']
	list_of_images2 = metadata_slice2['Path']
	list_of_images3 = metadata_slice3['Path']
	list_of_images4 = metadata_slice4['Path']
	list_of_images5 = metadata_slice5['Path']
	slice1= []
	slice2= []
	slice3= []
	slice4= []
	slice5= []
	slices_images=[]
	print("Reading images...")
	#for it, im in tqdm(enumerate(metadata[smc_mask].Path.values),
					  # total=smc_mask.sum(), desc='Reading MRI to memory'):
	for i in range(len(list_of_images1)):

		img1 = cv2.imread(str(path+list_of_images1[i]))
		img2 = cv2.imread(str(path+list_of_images2[i]))
		img3 = cv2.imread(str(path+list_of_images3[i]))
		img4 = cv2.imread(str(path+list_of_images4[i]))

		#img = np.arange(1 * 3 * 110 * 110).reshape((110, 110, 3))
		#img = img.astype('float32')

		img_resize1=cv2.resize(img1,(110,110))
		img_resize2=cv2.resize(img2,(110,110))
		img_resize3=cv2.resize(img3,(110,110))
		img_resize4=cv2.resize(img4,(110,110))
		slice1.append(img1)
		slice2.append(img2)
		slice3.append(img3)
		slice4.append(img4)

		#data[it, :, :, :] = data_img
	for i in range(len(list_of_images5)):
		img5 = cv2.imread(str(path+list_of_images5[i]))
		img_resize5=cv2.resize(img5,(110,110))
		slice5.append(img5)

	slices_images.append(slice1)
	slices_images.append(slice2)
	slices_images.append(slice3)
	slices_images.append(slice4)
	slices_images.append(slice5)
	label1 = metadata_slice1["Label"]
	label2 = metadata_slice2["Label"]
	label3 = metadata_slice3["Label"]
	label4 = metadata_slice4["Label"]
	label5 = metadata_slice5["Label"]
	for i in range(len(label1)):
		if label1[i] == "AD":
			label1[i] = 0
		else:
			label1[i] = 1
	for i in range(len(label2)):
		if label2[i] == "AD":
			label2[i] = 0
		else:
			label2[i] = 1
	for i in range(len(label3)):
		if label3[i] == "AD":
			label3[i] = 0
		else:
			label3[i] = 1
	for i in range(len(label4)):
		if label4[i] == "AD":
			label4[i] = 0
		else:
			label4[i] = 1
	for i in range(len(label5)):
		if label5[i] == "AD":
			label5[i] = 0
		else:
			label5[i] = 1
	slices_labels = []
	slices_labels.append(label1)
	slices_labels.append(label2)
	slices_labels.append(label3)
	slices_labels.append(label4)
	slices_labels.append(label5)
	return slices_images,slices_labels

def reorderRandomly(X,Y,list_of_images):
	'''
	Reorder in the same way the vector of images and labels

	Parameters
	------------
	numpy.ndarray, numpy.ndarray
		images and labels

	Returns
	------------
	numpy.ndarray, numpy.ndarray
		images and labels shuffled in the same way

	'''
	X, Y,list_of_images = shuffle(X, Y,list_of_images, random_state=43)
	return (X,Y,list_of_images)

def calculate_mean(values):
	'''
	Calculate the mean of a vector of values

	Parameters
	------------
	values: List of values

	Returns
	------------
	mean: float number, the mean of the values
	'''
	sum_mean = 0
	for value in values:
		sum_mean = sum_mean + float(value)

	mean = sum_mean/(len(values))

	return mean
