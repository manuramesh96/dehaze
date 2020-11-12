'''
creates annotation file for the ITS dataset
csv file has format
hazy image, its corresponding clear image
'''


import cv2
import numpy as np
import csv
import glob
import os
import sys
import pickle
import random

random.seed(0)

'''
../datasets/ITS/train/ITS_haze/
../datasets/ITS/train/ITS_clear/
'''

def create_main_dataset():
	#create dataset csv
	csvOutFile = open('../datasets/ITS_annotation.csv','w', newline='')
	outCSV = csv.writer(csvOutFile)
	outCSV.writerow(["hazy_path", "clear_path"])

	hazy_path = "../datasets/ITS/train/ITS_haze/"
	clear_path = "../datasets/ITS/train/ITS_clear/"

	hazyImages = glob.glob(hazy_path+'*.png')
	print('Number of hazy images = ', len(hazyImages))
	print('First sample of hazy image = ',hazyImages[0])

	for idx, hazyImgPath in enumerate(hazyImages):
		
		clearImgName = hazyImgPath.split('/')[-1].split('_')[0]+'.png'
		clearImgPath = clear_path + clearImgName

		row = [hazyImgPath, clearImgPath]
		outCSV.writerow(row)

	print('CSV file with hazy and clear image paths is created. Check ../dataset/ dir')

	csvOutFile.close()

def create_smaller_dataset():
	#from the main dataset pick only 5 percent of the total images in random order
	#and make a smaller dataset

	csvfile = open('../datasets/ITS_annotation.csv','r')
	readCSV = csv.reader(csvfile, delimiter = ',')

	csvOutFile = open('../datasets/ITS_subset_annotation.csv','w', newline='')
	outCSV = csv.writer(csvOutFile)

	#creating a separate test set also
	csvTestFile = open('../datasets/ITS_subset_test_annotation.csv','w', newline='')
	testCSV = csv.writer(csvTestFile)

	testCounter = 0
	maxTestImages = 100

	for idx, row in enumerate(readCSV):

		if idx == 0:
			outCSV.writerow(row)
			testCSV.writerow(row)
			continue

		p = random.uniform(0,1)
		if p <= 0.05:
			outCSV.writerow(row)

		else:
			#calling random functions again will alter random state
			#q = random.uniform(0,1)
			#q can be any number, but i need a more spread
			#p<=0.10 - didn't give much spread
			if p <= 0.052 and testCounter < maxTestImages: 
				testCSV.writerow(row)
				testCounter += 1

	print('CSV files with its subset created. Check ../dataset/ dir')

	csvfile.close()
	csvOutFile.close()
	csvTestFile.close()
	
if __name__ == "__main__":
	#create_main_dataset()
	create_smaller_dataset()
