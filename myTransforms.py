'''
Has transforms 
Author: Manu Ramesh
'''
import numpy as np
import cv2
import os
import sys
from PIL import Image


class ToPillowImg(object):
	def __init__(self):
		pass

	def __call__(self, sample):

		out = Image.fromarray(sample)
		return out


#calling this as transform will alter clear image also, so alter the datset only
class RenAugmentImg(object):
	'''
	input and output images are numpy cv2 style
	takes in 3 channel image and returns a six channel image with 
	white balanced, contrast corrected and gamma corrected channels
	Inspired from Ren et al. 
	https://github.com/rwenqi/GFN-dehazing/blob/master/demo_test.m
	'''
	
	def __init__(self):
		pass

	def __call__(self, sample):
	
	    	hazyImg = sample.astype(np.float32)

		hazy_wb = RealGWbal((255*hazyimg).astype(np.uint8));
		hazy_wb = hazy_wb/255;
		hazy_cont = (2*(0.5+np.mean(hazyimg))).*(hazyimg-np.mean(hazyimg));
		hazy_gamma = hazyimg ** 2.5;

		outImg = np.concatenate((hazy_wb, hazy_cont, hazy_gamma, sample), axis=2)

		return outImg


	#white balanced image generation - from Ren - ported to python
	def getRealGWbal(self, img):
		'''
		reads image in bgr format - cv2 style
		'''

		img = img.astype(np.float)
		r=img[:,:,2] 
		g=img[:,:,1]
		b=img[:,:,0]

		h, w , ch = img.shape
		avgR = np.mean(r)
		avgG = np.mean(g)
		avgB = np.mean(b)

		avgRGB = np.array([avgR, avgG, avgB])
		grayValue = (avgR + avgG + abgB)/3
		scaleValue = grayValue/(avgRGB+0.001)
		R = scaleValue[0] * r
		G = scaleValue[1] * g
		B = scaleValue[2] * b

		for i in range(h):
			for j in range(w):
				if R[i,j] > 255:
					R[i,j] = 255
				if G[i,j] > 255:
					G[i,j] = 255
				if B[i,j] > 255:
					B[i,j] = 255
			
		newI = np.zeros(img.shape, np.float) #change to int later if you want to
		newI[:,:,0] = R
		newI[:,:,1] = G
		newI[:,:,2] = B

		return newI
