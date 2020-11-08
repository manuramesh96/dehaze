'''
Has transforms 
Author: Manu Ramesh
'''
import numpy as np
import cv2
import os
import sys
from PIL import Image
import torch


class ToPillowImg(object):
	def __init__(self):
		pass

	def __call__(self, sample):

		out = Image.fromarray(sample)
		return out


#calling this as transform will alter clear image also, so alter the datset only
class RenAugmentImg(object):
	'''
	takes in torch tensor
	input and output images are numpy cv2 style
	takes in 3 channel image and returns a six channel image with 
	white balanced, contrast corrected and gamma corrected channels
	Inspired from Ren et al. 
	https://github.com/rwenqi/GFN-dehazing/blob/master/demo_test.m
	'''
	
	def __init__(self, device):
		self.device = device
		pass

	def __call__(self, sample):


		#sample = np.array(sample)	
		#hazyimg = sample.astype(np.float32)

		#sample is now a tensor
		hazyimg = sample.double()

		hazy_wb = self.getRealGWbal((255*hazyimg).type(torch.uint8));
		hazy_wb = hazy_wb/255;
		hazy_cont = (2*(0.5+torch.mean(hazyimg)))*(hazyimg-torch.mean(hazyimg));
		hazy_gamma = hazyimg ** 2.5;

		#outImg = np.concatenate((hazy_wb, hazy_cont, hazy_gamma, sample), axis=2)

		return hazy_wb, hazy_cont, hazy_gamma, sample


	#white balanced image generation - from Ren - ported to python
	def getRealGWbal(self, img):
		'''
		reads image in rgb format
		'''

		#first index is batch index
		img = img.float()
		r=img[:,0,:,:] 
		g=img[:,1,:,:]
		b=img[:,2,:,:]

		print(f"getRealGWbal: image size = {img.size()}")

		bsize, ch, h, w = img.size()

		avgR = torch.mean(r)
		avgG = torch.mean(g)
		avgB = torch.mean(b)

		avgRGB = torch.tensor([avgR, avgG, avgB]).to(self.device)
		grayValue = (avgR + avgG + avgB)/3
		#scaleValue = grayValue/(avgRGB+0.001)
		print(f"In renAuger: device = {self.device}")
		scaleValue = grayValue.to(self.device)/(avgRGB+torch.tensor([0.001]).to(self.device))
		R = scaleValue[0] * r
		G = scaleValue[1] * g
		B = scaleValue[2] * b

		print(f"R size = {R.size()}")
		print(f"R type = {R.dtype}")
	
		'''	
		for bid in range(bsize):
			for i in range(h):
				for j in range(w):
					if R[i,j] > 255:
						R[i,j] = 255
					if G[i,j] > 255:
						G[i,j] = 255
					if B[i,j] > 255:
						B[i,j] = 255
		'''
		'''

		R = torch.where(R>255.0, torch.tensor([255]), R)
		G = torch.where(G>255.0, 255, G)
		B = torch.where(B>255.0, 255, B)
		'''
		#img = img*torch.tensor(scaleValue) #just don't pass batch size 3
		img[:,0,:,:] *= scaleValue[0]
		img[:,1,:,:] *= scaleValue[1]
		img[:,2,:,:] *= scaleValue[2]

		#img = torch.where(img>torch.tensor([255.0]), torch.tensor([255.0]), img)
		for bid in range(bsize):
			for cid in range(ch):
				for i in range(h):
					for j in range(w):
						if img[bid, cid, i,j] > 255:
							img[bid, cid, i, j] = 255
		
		'''	
		newI = torch.zeros_like(img).float() #change to int later if you want to
		newI[:,:,0] = R
		newI[:,:,1] = G
		newI[:,:,2] = B
		'''
		#newImg = torch.cat((R,G,B), dim=1) #along ch dim
		newImg = img

		return newImg.float()
