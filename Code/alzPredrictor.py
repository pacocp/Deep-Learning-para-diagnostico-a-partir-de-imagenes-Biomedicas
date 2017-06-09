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
from PIL import ImageTk, Image

class MainWindow():

    #----------------

    def __init__(self, main,filename):

        #attributes
        self.main = main
        self.filename = filename
        self.prediction = ""
        screen_width = main.winfo_screenwidth()
        screen_height = main.winfo_screenheight()
        self.filename_name_only = filename.split("/")
        #Fullscreen
        main.attributes("-fullscreen", True)

        # canvas for image
        self.canvas = Canvas(main, width=screen_width, height=screen_height)
        self.canvas.grid(row=screen_height, column=screen_width)
        self.state = False

        self.image = ImageTk.PhotoImage(PIL.Image.open(self.filename))
        # set first image on canvas
        """ You can change the place of the images giving
        different values to the screen_width and screen_height """
        self.image_on_canvas = self.canvas.create_image((screen_width/2)-100, (screen_height/2)-200, anchor = NW, image = self.image)
        self.texto = self.canvas.create_text((screen_width)-(screen_width/2),50,font=("Purisa", 16),
        text = self.filename_name_only[len(self.filename_name_only)-1])
        self.prediction_text = self.canvas.create_text((screen_width)-(screen_width/2),(screen_height/2)-300,font=("Purisa", 16),
        text = "Predicted label is: "+self.prediction)

         # button to close
        self.button_close = Button(main, text="Close", command=self.closeButton)
        #self.button.grid(row=0, column=0)
        self.button_close.place(x=0,y=0)

        # button to change to next image
        self.button_nextimage = Button(main, text="Choose Image", command=self.chooseImage)
        """ You can change the place of the button giving
        different values to the screen_width and screen_height """
        #self.button.grid(row=4, column=10,columnspan=2, rowspan=2)
        self.button_nextimage.place(x=(screen_width/2)-120,y=0)

        # button to change to previous image
        self.button_previous = Button(main, text="Predict", command=self.predict(filename))
        #self.button.grid(row=4, column=11)
        """ You can change the place of the button giving
        different values to the screen_width and screen_height """
        self.button_previous.place(x=(screen_width/2),y=0)


    #----------------
    # BUTTONS
    #----------------
    def predict(self,filename):
        '''Predicting the label of the image with keras model'''
    def chooseImage(self):
        '''Choose the image for predicting'''
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
        new_filename = fd.askopenfilename(parent=self.main) # show an "Open" dialog box and return the path to the selected file
        self.image = ImageTk.PhotoImage(PIL.Image.open(new_filename))
        # change image
        self.filename_name_only = new_filename.split("/")
        self.canvas.itemconfigure(self.texto,
        text=self.filename_name_only[len(self.filename_name_only)-1])
        self.canvas.itemconfig(self.image_on_canvas, image = self.image)

    def closeButton(self):
        #Close button
        self.main.destroy()
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
root = Tk()
root = Toplevel()
filename = fd.askopenfilename(parent=root) # show an "Open" dialog box and return the path to the selected file
MainWindow(root,filename)
root.mainloop()
