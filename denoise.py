import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
 
# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread(args["image"])
image = cv2.resize(image,None,fx=0.1, fy=0.1, interpolation = cv2.INTER_CUBIC)


b,g,r = cv2.split(image)           # get b,g,r
rgb_img = cv2.merge([r,g,b])     # switch it to rgb

# Denoising
dst = cv2.fastNlMeansDenoising(image,None,7,7,21)
b,g,r = cv2.split(dst)           # get b,g,r
rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
gray = cv2.cvtColor(rgb_dst, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]



plt.subplot(211),plt.imshow(rgb_img)
plt.subplot(212),plt.imshow(thresh)
plt.show()