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

'''
../datasets/ITS/train/ITS_haze/
../datasets/ITS/train/ITS_clear/
'''


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

