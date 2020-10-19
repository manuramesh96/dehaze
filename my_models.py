import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms
from torchsummary import summary

class Autoencoder1(nn.Module):
	#https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
	'''
		too big to fit in memory
	'''

	def __init__(self):
		super(Autoencoder1, self).__init__()
		self.name = "Autoencoder1"

		self.encoder = nn.Sequential(
			nn.Linear(300*400*3,300*400),
			nn.Relu(True),
			nn.Linear(300*400,28*28),
			nn.Relu(True),
			nn.Linear(28 * 28, 128),
			nn.ReLU(True),
			nn.Linear(128, 64),
			nn.ReLU(True), 
			nn.Linear(64, 12),
			nn.ReLU(True), 
			nn.Linear(12, 3))

		self.decoder = nn.Sequential(
			nn.Linear(3, 12),
			nn.ReLU(True),
			nn.Linear(12, 64),
			nn.ReLU(True),
			nn.Linear(64, 128),
			nn.ReLU(True), 
			nn.Linear(128, 28 * 28),
			nn.Relu(True),
			nn.Linear(28*28, 300*400),
			nn.Relu(True),
			nn.Linear(300*400, 300*400*3),
			nn.Sigmoid())
			#nn.Tanh())

	def forward(self, x):
	    x = self.encoder(x)
	    x = self.decoder(x)
	    return x


class Autoencoder2(nn.Module):
	#https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
	def __init__(self):
		super(Autoencoder2, self).__init__()
		self.encoder = nn.Sequential(
		    nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
		    nn.ReLU(True),
		    nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
		    nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
		    nn.ReLU(True),
		    nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
		)
		self.decoder = nn.Sequential(
		    nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
		    nn.ReLU(True),
		    nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
		    nn.ReLU(True),
		    nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
		    nn.ReLU(True),
		    nn.Upsample((300,400)),
		    nn.Sigmoid() #nn.Tanh()
		)

	def forward(self, x):
	    x = self.encoder(x)
	    x = self.decoder(x)
	    return x

class Autoencoder3(nn.Module):
	
	def __init__(self):
		super(Autoencoder3, self).__init__()
		self.encoder = nn.Sequential(
		nn.Conv2d(3,16,5),nn.ReLU(True),
		nn.MaxPool2d(2,2),
		nn.Conv2d(16,8,5),nn.ReLU(True),
		nn.MaxPool2d(2,2)
		)
		self.decoder = nn.Sequential(
		nn.Conv2d(8,16,5),nn.ReLU(True),
		nn.Upsample((148,198)),
		nn.Conv2d(16,8,5),nn.ReLU(True),
		nn.Upsample((296,396)),
		nn.Conv2d(8,3,5),nn.ReLU(True),
		nn.Upsample((300,400)),
		nn.Sigmoid()
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x


if __name__ == "__main__":

	model = Autoencoder2()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	#print(ac1.encoder)
	summary(model, (3, 300, 400))
	#summary(model, (3, 28, 28))
