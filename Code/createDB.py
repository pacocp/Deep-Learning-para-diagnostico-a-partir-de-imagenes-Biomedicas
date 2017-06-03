#!/usr/bin/env python

'''
Python script for converting nii images to jpg and save them in the different folders depening on their labels and test/train/evaluation
'''

import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import os
import glob

def convert_images(image_name, folder_type, label_of_images):
	#load the image_name
	epi_img = nib.load(image_name)
	#gest the data from the image
	epi_img_data = epi_img.get_data()
	#getting the slices
	slice_0 = epi_img_data[epi_img_data[0]/2,:,:]
	slice_1 = epi_img_data[:,epi_img_data[1]/2,:]
	slice_2 = epi_img_data[:,:,epi_img_data[2]/2]
	#plotting the slice that we care for
	plot = plt.imshow(slice_1,  cmap="gray")
	plt.axis('off')
	plot.axes.get_xaxis().set_visible(False)
	plot.axes.get_yaxis().set_visible(False)
	full_path = folder_type + '/' +  label_of_images + '/' + image_name + '.png'
	#sving it
	plt.savefig(full_path, bbox_inches='tight', pad_inches = 0)


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
	list_name_of_images = glob.glob("./*.nii")
	for inputFile in list_name_of_images:
		convert_images(inputFile,folderName,label)
else:
	#only converts one image
	convert_images(inputFile,folderName,label)
