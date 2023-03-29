# read img.jpg and print the intensity of the first row of the image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('img.jpg')
# normalize between 0 and 1
img = img / 255

print(img[0, :, 0])

def img_sample(img, x, row):
    """ x is a value between 0 and 1. return the nearest neighbour."""
    return img[row, int(x * img.shape[1]), 0]
print(img_sample(img, 0.5, 0))