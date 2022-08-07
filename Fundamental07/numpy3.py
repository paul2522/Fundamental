from PIL import Image, ImageColor
import os
import numpy as np

img_path = "newyork.jpg"
img = Image.open(img_path)
#print(os.getcwd())
#print(img_path)
#print(type(img))
#img.show()
imgArray = np.array(img)

print(imgArray)