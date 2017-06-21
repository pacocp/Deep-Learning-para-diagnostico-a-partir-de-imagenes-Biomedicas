#!/usr/bin/python3.6

import pandas as pd
import numpy as np
import tempfile
import shutil

'''
If the result file doesn't exist, this function is going to create it.
This method is going to be used both by the experiment script, and the software
'''
def create_file(columns,name_of_file):
	# create the pandas daframe
	df = pd.DataFrame(columns=columns)
	#saving it to a csv
	df.to_csv("results/"+name_of_file,sep=',', encoding='utf-8')

'''
Method for reading the csv file for experiments or results of predictions
'''
def read_from_file(name_of_file):
	# read the csv of the file
	df = pd.read_csv("results/"+name_of_file, sep=',')
	# return the dataframe
	return df

'''
Method for writting the csv file with experiments or results of predictions
'''
def write_to_file(df,name_of_file):
	# write the csv to a file
	df.to_csv("results/"+name_of_file,sep=',', encoding='utf-8')

'''
Function for creating a temporal directory so I could read the images and reshape them with keras,
and move image there.
'''
def create_temp_dir(filename):
	# create a temporal directory
	temp = tempfile.TemporaryDirectory()
	temp_name = temp.name
	# copying the file to the temporal directory
	shutil.copy2(filename,temp_name)
	return temp_name

'''
Function for deleting the temp repository that has been created before for predicting an image
'''
def delete_temp_dir(dirpath):
	# deleting the temporl directory
	shutil.rmtree(dirpath)
