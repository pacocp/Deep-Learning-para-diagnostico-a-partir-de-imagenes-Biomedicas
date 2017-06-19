#!/usr/bin/python3.6

import pandas as pd

'''
If the result file doesn't exist, this function is going to create it.
This method is going to be used both by the experiment script, and the software
'''
def create_file(columns,name_of_file):
	# create the pandas daframe
	df = pd.DataFrame(index=columns[0],columns=columns)
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
