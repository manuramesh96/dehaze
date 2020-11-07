from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2

from myTransforms import RenAugmentImg()

class ITS_Dataset(Dataset):
	'''ITS Dataset'''

	def __init__(self, csv_file, root_dir, transform=None, renAugment=False):
                 """
                 Args:
                     csv_file (string): Path to the csv file with annotations.
                     root_dir (string): Directory with all the images.
                     transform (callable, optional): Optional transform to be applied
                         on a sample.
		
		     renAugment: if True set augments the hazy image with 3 more images according to Ren et al paper
                 """
                 self.its_frame = pd.read_csv(csv_file)
                 self.root_dir = root_dir
                 self.transform = transform

	def __len__(self):
		return len(self.its_frame)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
		    idx = idx.tolist()

		hazyImgPath = self.its_frame.iloc[idx,0]
		clearImgPath = self.its_frame.iloc[idx,1]

		#hazyImg = Image.open(hazyImgPath)
		#clearImg = Image.open(clearImgPath)

		
		hazyImg = cv2.imread(hazyImgPath)#, 0) #remove 0 later and convert to BGR2RGB
		clearImg = cv2.imread(clearImgPath)#, 0) #remove 0 later and convert BGR2RGB
		#hazyImg = cv2.cvtColor(hazyImg, cv2.COLOR_GRAY2RGB)
		#clearImg = cv2.cvtColor(clearImg, cv2.COLOR_GRAY2RGB)

		if renAugment == True:
			renAg = RenAugmentImg()
			hazyImg = renAg(hazyImg)

		#temp commenting for trial
		#hazyImg = Image.fromarray(hazyImg)
		#clearImg = Image.fromarray(clearImg)

		sample = {'hazyImg': hazyImg, 'clearImg': clearImg}

		if self.transform:
		    #sometimes you might want to transform the clear image also
		    #eg: for resizing applications
		    #you can add that feature here
		    #sample = {'hazyImg':self.transform(sample['hazyImg']), 'clearImg':sample['clearImg']}
		    sample = {'hazyImg':self.transform(sample['hazyImg']), 'clearImg':self.transform(sample['clearImg'])}

		return sample

	def transform(self, sample):
		'''write your resizing functions here'''

		return sample

