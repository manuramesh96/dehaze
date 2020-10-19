import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ITS_Dataset import ITS_Dataset
from my_models import Autoencoder2  
from my_models import Autoencoder3  

from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms, datasets
import pickle

import torch.nn as nn
import PIL
from PIL import Image
import cv2

'''
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
'''

transform = transforms.Compose([
	transforms.Resize((300,400)),
        transforms.ToTensor()
    ])

its_dataset = ITS_Dataset('../datasets/ITS_annotation.csv','../datasets/ITS/',transform=transform)
#its_dataset = ITS_Dataset('../datasets/ITS_annotation.csv','../datasets/ITS/')


def trial1_1():
	fig = plt.figure()
	print('Length of Cow Faces dataset = ', len(its_dataset))

	for i in range(len(its_dataset)):
		sample = its_dataset[i]

		print(i, sample['hazyImg'].shape, sample['clearImg'].shape)

		ax = plt.subplot(1, 4, i + 1)
		plt.tight_layout()
		ax.set_title('Sample #{}'.format(i))
		ax.axis('off')
		plt.imshow(sample['hazyImg'].permute(2,1,0))

		if i == 3:
			plt.show()
			fig.savefig('../outputs/temp_dataset_img.png')
			break

def trial1_2():
	''' trial with data loader'''
	dataloader = DataLoader(its_dataset, batch_size=4, shuffle=True, num_workers=4)
	for i_batch, sample_batched in enumerate(dataloader):
	
	    #hazyImg = sample_batched['hazyImg']
	    #hazyImg = transforms.ToTensor()(hazyImg).unsqueeze_(0)
	    #clearImg = sample_batched['clearImg']
	    #clearImg = transforms.ToTensor()(clearImg).unsqueeze_(0)
		
	    print(i_batch, sample_batched['hazyImg'].size(), sample_batched['clearImg'].size())
	    #print(i_batch, hazyImg.size(), clearImg.size())


def trial1_3():
	''' trial with data loader'''
	#https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets

	dataloader = DataLoader(face_dataset, batch_size=4, shuffle=True, num_workers=4)
	#for i_batch, sample_batched in enumerate(dataloader):
	#    #print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
	
	batch_size = 16
	test_split = .2
	shuffle_dataset = True
	random_seed= 42

	dataset_size = len(face_dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(test_split * dataset_size))
	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices) #randomly shuffles indices
	train_indices, test_indices = indices[split:], indices[:split]

	#creating (PT?) data samplers and loaders
	train_sampler = SubsetRandomSampler(train_indices)
	test_sampler = SubsetRandomSampler(test_indices)
	
	train_loader = DataLoader(face_dataset, batch_size=batch_size,sampler=train_sampler)
	test_loader  = DataLoader(face_dataset, batch_size=batch_size,sampler=test_sampler)

	#getting the model
	#vgg16 = models.vgg16() 

	for i_batch, sample_batched in enumerate(train_loader):
	    print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
	    #print(i_batch, sample_batched.size())
	    #print(i_batch, sample_batched['label'].size())

	for i_batch, sample_batched in enumerate(test_loader):
	    print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
	    #print('Test')


def get_loaders(batch_size_train, batch_size_test):
	''' makes and gets data loaders'''
	#https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets

	#for i_batch, sample_batched in enumerate(dataloader):
	#    #print(i_batch, sample_batched['image'].size(), sample_batched['label'].size())
	
	#batch_size = 64 #128 #16
	test_split = .2
	shuffle_dataset = True
	random_seed= 42

	dataset_size = len(its_dataset)
	indices = list(range(dataset_size))
	split = int(np.floor(test_split * dataset_size))
	if shuffle_dataset:
		np.random.seed(random_seed)
		np.random.shuffle(indices) #randomly shuffles indices
	train_indices, test_indices = indices[split:], indices[:split]

	#creating (PT?) data samplers and loaders
	train_sampler = SubsetRandomSampler(train_indices)
	test_sampler = SubsetRandomSampler(test_indices)
	
	train_loader = DataLoader(its_dataset, batch_size=batch_size_train,sampler=train_sampler, num_workers=16)
	test_loader  = DataLoader(its_dataset, batch_size=batch_size_test,sampler=test_sampler, num_workers=16)
	
	return train_loader, test_loader

	
def train(model, epochs, train_loader):
	model.to(device)
	model.train()
	#criterion = torch.nn.CrossEntropyLoss() 
	criterion = nn.MSELoss()

	learning_rate = 1e-3
	#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9) 
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

	
	for epoch in range(epochs):
		running_loss = 0.0
		for idx, batch in enumerate(train_loader):
			#print('Pass',end=' ')
			hazyImgs = batch['hazyImg'].to(device)
			clearImgs = batch['clearImg'].to(device)

			#print('Hazy img size = ', hazyImgs.size())
			#print('Clear img size = ', clearImgs.size())

			
			optimizer.zero_grad() #clearing all gradients
			outputs = model(hazyImgs)

			
			loss = criterion(outputs, clearImgs)
			loss.backward()
			optimizer.step()

			
			running_loss += loss.item()
			if idx%4 == 0:
				print(f'Epoch: {epoch}, Batch: {idx}, Running loss = {running_loss}\n')
				running_loss = 0
		print(' ')

	#saving states - see pytorch tutorial for how to load
	#https://pytorch.org/tutorials/beginner/saving_loading_models.html
	torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./states/ac2_{epochs}-epch_states.p')
	


def test(model, test_loader, device,  checkpoint_path):

	print(f"Evaluation: device = {device}")
	model.to(device)
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model_state_dict'])


	criterion = nn.MSELoss()
	avg_loss = 0 
	n_batches = 0

	print(f"Running model evaluation with states from {checkpoint_path}")
	
	with torch.no_grad():
		for idx, batch  in enumerate(test_loader):

			hazyImgs = batch['hazyImg'].to(device)
			clearImgs = batch['clearImg'].to(device)

			outputs = model(hazyImgs)
			loss = criterion(outputs, clearImgs)
	
			avg_loss += loss.item()
			n_batches = idx
			
			if idx%8 == 0:
				print(f'Current evaluation batch = {idx}, cumulative loss = {avg_loss}')

	n_samples = n_batches * test_loader.batch_size					
	avg_loss /= (n_samples)

	print(f"Average testing loss for approx {n_samples} test images = {avg_loss}")
	#approx because last batch need not have batch_size no of test images


def sample_outputs(model, test_loader, device, checkpoint_path):

	print(f"Outputs sampler: device = {device}")
	model.to(device)
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	
	n_images = 5 #save 5 images


	criterion = nn.MSELoss()
	avg_loss = 0 
	n_batches = 0

	print(f"Running output sampling with model states from {checkpoint_path}")
	
	with torch.no_grad():
		for idx, batch  in enumerate(test_loader):

			hazyImgs = batch['hazyImg'].to(device)
			clearImgs = batch['clearImg'].to(device)
			
			outputs = model(hazyImgs)
		
			outputs = outputs.cpu()
			clearImgs = clearImgs.cpu()
			#outputs = outputs.numpy()
	
			#print(f"Outputs size = {outputs.shape}")
			#shape = batch x ch x h x w
			
			for i in range(n_images):
				#out_img = np.zeros((outputs.shape[2], 2*outputs.shape[3]), np.uint8)
				pred_img = torch.squeeze(outputs[i]).permute(1,2,0)
				pred_img = pred_img.numpy()

				clearImg = torch.squeeze(clearImgs[i]).permute(1,2,0)
				clearImg = clearImg.numpy()
				
				print(f"pred image size = {pred_img.shape}")
				print(f"clear image size = {clearImg.shape}")

				big_img = np.concatenate((pred_img, clearImg), axis = 1)
				print(f"Big image shape = {big_img.shape}")

				#big_img =  Image.fromarray(np.uint8(big_img))
				#big_img =  Image.fromarray(np.uint8(big_img)*255)

				#big_img.save(f"../outputs/outImage_{i}.jpg")
				cv2.imwrite(f"../outputs/outImage_{i}.jpg", np.uint8(big_img*255))
			break

def make_model():
	#model  = Autoencoder3()
	model  = Autoencoder2()
	return model


if __name__ == "__main__":

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(f'cuda available = {torch.cuda.is_available()}, Device = {device}')

	model  = make_model()
	#print('Model = ', vars(model))
 
	epochs = 1
	batch_size_train = 256 #128 #16
	batch_size_test = 16 #256

	train_loader, test_loader = get_loaders(batch_size_train, batch_size_test)
	#print('Train loader = ',train_loader)

	sample = next(iter(test_loader))
	#print(sample)
	sample = torch.squeeze(sample['hazyImg'][0]).permute(2,1,0)
	sample = sample.numpy()
	print('sample = \n', sample)
	
	cv2.imwrite('../outputs/sample.jpg',np.uint8(sample*255))


	trial1_1()
	#trial1_2()
	#trial1_3()
	#train(model=model, epochs=epochs, train_loader=train_loader)

	#test(model=model, test_loader=test_loader, device=device,  checkpoint_path="./states/ac2_1-epch_states.p")

	sample_outputs(model, test_loader, device, checkpoint_path="./states/ac2_1-epch_states.p")

	pass
