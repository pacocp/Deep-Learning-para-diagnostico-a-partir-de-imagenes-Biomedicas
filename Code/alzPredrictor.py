#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#----------------------------------------------------------------------
# Francisco Carrillo Pérez <franciscocp@correo.ugr.es>
# https://github.com/pacocp
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Predicting ALzheimer images with Keras, software
#----------------------------------------------------------------------

from sys import argv
from sys import exit
from sys import platform as _platform
import subprocess
import os
import errno
from tkinter import *
from tkinter import filedialog as fd
import PIL
import pandas as pd
from PIL import ImageTk, Image

# Imports from my files
from predict import predict_class
from save_load_results import create_file,write_to_file,read_from_file
class MainWindow():

	#----------------

	def __init__(self, main,filename, df_results):

		#attributes
		self.main = main
		self.filename = filename
		self.prediction = ""
		self.df_results = df_results
		self.screen_width = main.winfo_screenwidth()
		self.screen_height = main.winfo_screenheight()
		self.filename_name_only = filename.split("/")
		#Fullscreen
		#main.attributes("-fullscreen", True)

		# canvas for image
		self.canvas = Canvas(main, width=self.screen_width, height=self.screen_height)
		self.canvas.grid(row=self.screen_height, column=self.screen_width)
		self.state = False

		self.image = ImageTk.PhotoImage(PIL.Image.open(self.filename))
		# set first image on canvas
		""" You can change the place of the images giving
		different values to the self.screen_width and self.screen_height """
		self.image_on_canvas = self.canvas.create_image((self.screen_width/2)-100, (self.screen_height/2)-200, anchor = NW, image = self.image)
		self.texto = self.canvas.create_text((self.screen_width)-(self.screen_width/2),50,font=("Purisa", 16),
		text = self.filename_name_only[len(self.filename_name_only)-1])
		self.prediction_text = self.canvas.create_text((self.screen_width)-(self.screen_width/2),(self.screen_height/2)-300,font=("Purisa", 16),
		text = "Predicted label is: "+self.prediction)

		 # button to close
		self.button_close = Button(main, text="Close", command=self.closeButton)
		#self.button.grid(row=0, column=0)
		self.button_close.place(x=0,y=0)

		# button to change to next image
		self.button_chooseImage = Button(main, text="Choose Image", command=self.chooseImage)
		""" You can change the place of the button giving
		different values to the self.screen_width and self.screen_height """
		#self.button.grid(row=4, column=10,columnspan=2, rowspan=2)
		self.button_chooseImage.place(x=(self.screen_width/2)-120,y=0)

		# button to change to predict image
		self.button_predict = Button(main, text="Predict", command=self.predict)
		#self.button.grid(row=4, column=11)
		""" You can change the place of the button giving
		different values to the self.screen_width and self.screen_height """
		self.button_predict.place(x=(self.screen_width/2),y=0)


	#----------------
	# BUTTONS
	#----------------
	def predict(self):
		'''Predicting the label of the image with keras model'''
		self.prediction = predict_class(self.filename)
		data_of_training = {'ID': len(self.df_results.index),'NAME': filename, 'PREDICTED LABEL': self.prediction}
		df_aux = pd.DataFrame(data=[data_of_training])
		self.df_results = self.df_results.append(df_aux, ignore_index=True)
		self.canvas.itemconfigure(self.prediction_text,
		text="Predicted label is: "+str(self.prediction))
	def chooseImage(self):
		'''Choose the image for predicting'''
		Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
		new_filename = fd.askopenfilename(parent=self.main) # show an "Open" dialog box and return the path to the selected file
		self.filename = new_filename
		self.image = ImageTk.PhotoImage(PIL.Image.open(new_filename))
		# change image
		self.filename_name_only = new_filename.split("/")
		self.canvas.itemconfigure(self.texto,
		text=self.filename_name_only[len(self.filename_name_only)-1])
		self.canvas.itemconfig(self.image_on_canvas, image = self.image)

	def closeButton(self):
		#Close button
		self.main.destroy()
		write_to_file(df_results,'results.csv')
		sys.exit()

	def toggle_fullscreen(self, main):
		self.state = not self.state  # Just toggling the boolean
		main.attributes("-fullscreen", self.state)
		return "break"

	def end_fullscreen(self,main):
		self.state = False
		main.attributes("-fullscreen", False)
		return "break"

""" MAIN PROGRAM """
name_of_file = "results.cvs"
# Trying to open the results file, if it doesn't exist create it
try:
	df_results = read_from_file(name_of_file)
except:
	# Creating the columns for the dataframe
	columns = ['ID','NAME','PREDICTED LABEL']
	# Creating the file for the dataframe
	create_file(columns,name_of_file)
	# Opening the experiments file
	df_results = read_from_file(name_of_file)

root = Tk()
root = Toplevel()
filename = fd.askopenfilename(parent=root) # show an "Open" dialog box and return the path to the selected file
MainWindow(root,filename,df_results)
root.mainloop()
