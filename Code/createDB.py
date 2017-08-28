#!/usr/bin/env python

'''
Francisco Carrillo Pérez (2017)
carrilloperezfrancisco@gmail.com
Universidad de Granada
TFG
'''

'''
Python script for converting nii images to jpg and save them in the different folders depening on their labels and test/train/evaluation
'''

import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import math
import matplotlib.image

#example from: https://github.com/ZheweiMedia/DL_experiments/blob/691aa805f292f7f2a6f56e20fdd25ea5bc7b3a6a/fMRI_CSV_Analysis/SIEMENS/utility01_output_image.py
def convert_images(image_name, folder_type, label_of_images, allImages):
	#load the image_name
	epi_img = nib.load(image_name)
	#gest the data from the image
	epi_img_data = epi_img.get_data()
	#getting the slices
	print("Getting the slices")
	index2 = np.array(epi_img_data[1], dtype=int)
	std_image = epi_img.get_data()[:,:,45]
	slice_1 = epi_img_data[:,:,45]
	#plotting the slice that we care for
	print("Plotting the slices")
	plot = plt.imshow(slice_1, cmap="gray")
	plt.axis('off')
	plot.axes.get_xaxis().set_visible(False)
	plot.axes.get_yaxis().set_visible(False)
	if allImages == "1":
		full_path = image_name + '.png'

	else:
		full_path = folder_type + '/' +  label_of_images + '/' + image_name + '.png'
	#sving it
	print("Saving the slice")
	#plt.savefig(full_path, bbox_inches='tight', pad_inches = 0)
	matplotlib.image.imsave(full_path, std_image, cmap="gray")

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Convert nii images and saved them for using Keras")
	parser.add_argument("-i", "--inputFile",
						help="input file",
						dest='inputFile')
	parser.add_argument("-a", "--allImages",
						help="all images in the folder(0,1)",
						dest='allImages')
	parser.add_argument("-o", "--outputFileStem",
						help="output file",
						default="output.jpg",
						dest='outputFileStem')
	parser.add_argument("-d", "--outputDir",
						help="output image directory(train,test,evaluation)",
						dest='outputDir',
						default='.')
	parser.add_argument("-l", "--labelImage",
						help="label of the image (ad,normal or mci)",
						dest='labelImage',
						default='normal')

	args = parser.parse_args()

	inputFile = args.inputFile
	allImages = args.allImages
	folderName = args.outputDir
	label = args.labelImage

# If the user wants to convert all the images of a folder
if(allImages == '1'):
	print("All Images")
	list_name_of_images = glob.glob("*.nii")
	for inputFile in list_name_of_images:
		convert_images(inputFile,folderName,label, allImages)
else:
	#only converts one image
	convert_images(inputFile,folderName,label,allImages)
