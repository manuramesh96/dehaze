'''
just writing some utility functions
'''
import numpy as np
import pickle
import glob
import os
import sys
import torch

def getVarOfPsnrLists():

	psnrListPaths = glob.glob("../final_outputs/*psnrList.p")
	print(f"Psnr list paths = {psnrListPaths}")
	for psnrListPath in psnrListPaths:

		modelName = psnrListPath.split('/')[-1].split('_psnrList.p')[0]
		psnrList = pickle.load(open(psnrListPath,'rb'))

		psnrList = np.array([x.item() for x in psnrList[:]])
		#print(f"Model {modelName} PSNR List = \n{psnrList}")

		psnrMean = np.mean(psnrList)
		psnrMedian  = np.median(psnrList)
		psnrVar  = np.var(psnrList)
		print(f"\nModel {modelName} PSNR Mean  = {psnrMean}")
		print(f"Model {modelName} PSNR Median  = {psnrMedian}")
		print(f"Model {modelName} PSNR Var   = {psnrVar}")
		print(f"Model {modelName} PSNR Max val = {np.max(psnrList)}\n")
		
		sortedIndices = np.argsort(psnrList)
		sortedArray = np.sort(psnrList)

		print(f"Model {modelName} best performance for images {sortedIndices[:-6:-1]}")
		print(f"Model {modelName} best performance values are {sortedArray[:-6:-1]}\n")
		print("*********************************************************")



def getVarOfSsimLists():

	ssimListPaths = glob.glob("../final_outputs/*ssimList.p")
	print(f"Ssim list paths = {ssimListPaths}")
	for ssimListPath in ssimListPaths:

		modelName = ssimListPath.split('/')[-1].split('_ssimList.p')[0]
		ssimList = pickle.load(open(ssimListPath,'rb'))

		ssimList = np.array([x.item() for x in ssimList[:]])
		#print(f"Model {modelName} SSIM List = \n{ssimList}")

		ssimMean = np.mean(ssimList)
		ssimMedian  = np.median(ssimList)
		ssimVar  = np.var(ssimList)
		print(f"\nModel {modelName} SSIM Mean  = {ssimMean}")
		print(f"Model {modelName} SSIM Median  = {ssimMedian}")
		print(f"Model {modelName} SSIM Var   = {ssimVar}")
		print(f"Model {modelName} SSIM Max val = {np.max(ssimList)}\n")
		
		sortedIndices = np.argsort(ssimList)
		sortedArray = np.sort(ssimList)

		print(f"Model {modelName} best performance for images {sortedIndices[:-6:-1]}")
		print(f"Model {modelName} best performance values are {sortedArray[:-6:-1]}\n")
		print("*********************************************************")

if __name__ == "__main__":

	getVarOfPsnrLists()
	getVarOfSsimLists()
