# drag the slider to modify the image.
#

from tkinter import *
from PIL import Image, ImageTk, ImageEnhance
import sys

#
# enhancer widget

class Enhance(Frame):
    def __init__(self, master, image, enhancer, value):
        Frame.__init__(self, master)

        # set up the image
        self.tkim = ImageTk.PhotoImage(image.mode, image.size)
        self.enhancer = enhancer(image)
        self.update("1.0") # normalize

    def update(self, value):
        self.value = eval(value)
        self.tkim.paste(self.enhancer.enhance(self.value))

#
# main

root = Tk()

im = Image.open(sys.argv[1])

im.thumbnail((200, 200))

Enhance(root, im, ImageEnhance.Color, 3.0)
Enhance(Toplevel(), im, "Sharpness", ImageEnhance.Sharpness, -2.0, 10.0).pack()
Enhance(Toplevel(), im, "Brightness", ImageEnhance.Brightness, -1.0, 3.0).pack()
Enhance(Toplevel(), im, "Contrast", ImageEnhance.Contrast, -1.0, 3.0).pack()


enhancer = ImageEnhance.Sharpness(image)
im = ImageEnhance.Sharpness(image).enhance(3)

root.mainloop()