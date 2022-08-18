import os
import tkinter as tk
from matplotlib import pyplot as plt
from log.BaseLogger import LOG_LOCATION
from tkinter import filedialog
import torch
import inspect


class _setit:
    """Internal class. It wraps the command in the widget OptionMenu."""

    def __init__(self, var, value, callback=None):
        self.__value = value
        self.__var = var
        self.__callback = callback

    def __call__(self, *args):
        self.__var.set(self.__value)
        if self.__callback:
            self.__callback(self.__value, *args)

class LinkedDropDown(tk.OptionMenu):

    def __init__(self, master, variable, value, previous = None ,*args, **kwargs):
        tk.OptionMenu.__init__(window, var, value, *args, **kwargs)
        self.values = values
        self.previous = []

    def setNext(self, _next):

        self.next = _next

class UI():

    def __init__(self):

        self.LogDirectory = LOG_LOCATION + '/runs'

        self.BuildEnv()

        self.window.mainloop()

        self.OpendFiles = []


    def BuildEnv(self):
        # Build parent window

        self.window = tk.Tk()

        self.window.geometry('1080x720')

        self.window.title('Plot Results')

        self.SelectedFiles = tk.Frame(self.window)
        self.SelectedFiles.pack(side= tk.RIGHT)

        self.AddFileFrame = tk.Frame(self.SelectedFiles)
        self.AddFileFrame.pack(side = tk.BOTTOM)

        self.AddButton = tk.Button(self.AddFileFrame,text='Add File', command= self.AddFile)
        self.AddButton.pack()





    def AddFile(self):

        filename = filedialog.askopenfilename()

        if filename != '':

            suffix = os.path.splitext(filename)[-1]

            try:

                if suffix == '.pt':
                    file = torch.load(filename)
                    tensors = inspect.getmembers(file,lambda x: isinstance(x,torch.Tensor) or isinstance())

                    Scroll = tk.OptionMenu(self.SelectedFiles, value = '')


            except:
                pass


        1



