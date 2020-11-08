import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms
from torchsummary import summary
import copy
from torch.autograd import Variable

from myTransforms import ToPillowImg
from myTransforms import RenAugmentImg
from dehaze1113 import Dense_rain_cvprw3 as zhangAE


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

class Autoencoder4(nn.Module): #from ai homework
  '''
  15 epoch, avg test loss = ?
  from AI Homework
  '''
  def __init__(self):
    super(Autoencoder4, self).__init__()

    # encoder
    self.n_channels = 3 #number of input channels
    self.n_latent = 4

    self.conv1 = nn.Conv2d(self.n_channels, 5, kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(5, 10, kernel_size=2, stride=2)
    self.conv3 = nn.Conv2d(10, self.n_latent, kernel_size=2, stride=1)

    # decoder
    self.tran_conv1 = nn.ConvTranspose2d(self.n_latent, 10, kernel_size=2, stride=1)
    self.tran_conv2 = nn.ConvTranspose2d(10, 5, kernel_size=2, stride=2)
    self.tran_conv3 = nn.ConvTranspose2d(5, self.n_channels, kernel_size=2, stride=2)


  def forward(self, x):

    # encoding layers
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))

    # decoding layers
    x = F.relu(self.tran_conv1(x))
    x = F.relu(self.tran_conv2(x))
    x = torch.sigmoid(self.tran_conv3(x))
    
    return x


#mixes ZHang and Ren
class AutoEncoder5(nn.Module): #from ai homework
  '''
  only Zhang one epoch - loss around 2.00
  send in 256x256 images
  '''
  def __init__(self, device="cpu"):
    super(AutoEncoder5, self).__init__()
    
    self.device = device

    self.img_wb = []
    self.img_c = []
    self.img_g = []

    self.renAuger = RenAugmentImg(device=self.device)

    #fusion gates - coeffs
    self.wb_fgate = Variable(torch.tensor([1.0]), requires_grad=True)
    self.c_fgate  = Variable(torch.tensor([1.0]), requires_grad=True)
    self.g_fgate  = Variable(torch.tensor([1.0]), requires_grad=True)

    self.zae_wb  = zhangAE() #for white balanced image
    self.zae_c   = zhangAE() #for contrast enhanced image
    self.zae_g   = zhangAE() #for gamma corrected image
    
    #self.zae_rgb = zhangAE() #for the original image #not needed now

    #gating by linear layers- 
    #not enough memory - needs 288GB, nn.Linear(266*256*3*2, 256*256*3) - three such layers
    # fusion by coeffs
    #can try gating by conv layers later
    self.wb_gate = nn.Conv2d(6, 3, 3, stride=1, padding=1)
    self.c_gate  = nn.Conv2d(6, 3, 3, stride=1, padding=1)
    self.g_gate  = nn.Conv2d(6, 3, 3, stride=1, padding=1)
	
    self.relu = nn.ReLU()

  def forward(self, x):
	
    x_wb, x_c, x_g, x_rgb = self.renAuger(x)
    self.img_wb = copy.deepcopy(x_wb) #if this doesn't work, read values directly into self variables above

    x_wb = x_wb.float();
    x_c = x_c.float();
    x_g = x_g.float()


    self.img_c  = copy.deepcopy(x_c )
    self.img_g  = copy.deepcopy(x_g )

    x_wb = self.zae_wb(x_wb).to(self.device)		
    x_c = self.zae_wb(x_c).to(self.device)		
    x_g = self.zae_wb(x_g).to(self.device)	

    print(f"x_wb size = {x_wb.size()}")	

    #0 is batch dim, 1 is channel dim
    x_c  = torch.cat((self.img_c,  x_c ), dim=1)
    x_g  = torch.cat((self.img_g,  x_g ), dim=1)   
    x_wb = torch.cat((self.img_wb, x_wb), dim=1)
    
    print(f"x_wb size after conv = {x_wb.size()}")	

    x_wb = self.relu(self.wb_gate(x_wb))
    x_c  = self.relu(self.c_gate(x_c  ))
    x_g  = self.relu(self.g_gate(x_g  ))

    #out = (F.sigmoid(self.wb_fgate) * x_wb + F.sigmoid(self.c_fgate) * x_c + F.sigmoid(self.g_fgate) * x_g) / torch.tesnsor([3]).to(self.device)
    out = (F.sigmoid(self.wb_fgate.to(self.device)) * x_wb + F.sigmoid(self.c_fgate.to(self.device)) * x_c + F.sigmoid(self.g_fgate.to(self.device)) * x_g) / torch.tensor([3.0]).to(self.device)

    out = F.sigmoid(out)

    return out



#mixes ZHang and Ren 2
class AutoEncoder6(nn.Module): #from ai homework
  '''
  only Zhang one epoch - loss around 2.00
  send in 256x256 images
  '''
  def __init__(self, device="cpu"):
    super(AutoEncoder6, self).__init__()
    
    self.device = device

    self.img_wb = []
    self.img_c = []
    self.img_g = []

    self.renAuger = RenAugmentImg(device=self.device)

    #fusion gates - coeffs
    #self.wb_fgate = Variable(torch.tensor([1.0]), requires_grad=True)
    #self.c_fgate  = Variable(torch.tensor([1.0]), requires_grad=True)
    #self.g_fgate  = Variable(torch.tensor([1.0]), requires_grad=True)

    self.fusionGate = nn.Conv2d(9,3,1, stride=1) # padding = 0 by default

    self.zae_wb  = zhangAE() #for white balanced image
    self.zae_c   = zhangAE() #for contrast enhanced image
    self.zae_g   = zhangAE() #for gamma corrected image
    
    #self.zae_rgb = zhangAE() #for the original image #not needed now

    #gating by linear layers- 
    #not enough memory - needs 288GB, nn.Linear(266*256*3*2, 256*256*3) - three such layers
    # fusion by coeffs
    #can try gating by conv layers later
    self.wb_gate = nn.Conv2d(6, 3, 1, stride=1, padding=0)
    self.c_gate  = nn.Conv2d(6, 3, 1, stride=1, padding=0)
    self.g_gate  = nn.Conv2d(6, 3, 1, stride=1, padding=0)
	
    self.relu = nn.ReLU()

  def forward(self, x):
	
    x_wb, x_c, x_g, x_rgb = self.renAuger(x)
    self.img_wb = copy.deepcopy(x_wb) #if this doesn't work, read values directly into self variables above

    x_wb = x_wb.float();
    x_c = x_c.float();
    x_g = x_g.float()


    self.img_c  = copy.deepcopy(x_c )
    self.img_g  = copy.deepcopy(x_g )

    x_wb = self.zae_wb(x_wb).to(self.device)		
    x_c = self.zae_wb(x_c).to(self.device)		
    x_g = self.zae_wb(x_g).to(self.device)	

    print(f"x_wb size = {x_wb.size()}")	

    #0 is batch dim, 1 is channel dim
    x_c  = torch.cat((self.img_c,  x_c ), dim=1)
    x_g  = torch.cat((self.img_g,  x_g ), dim=1)   
    x_wb = torch.cat((self.img_wb, x_wb), dim=1)
    
    print(f"x_wb size after conv = {x_wb.size()}")	

    x_wb = self.relu(self.wb_gate(x_wb))
    x_c  = self.relu(self.c_gate(x_c  ))
    x_g  = self.relu(self.g_gate(x_g  ))

    #out = (F.sigmoid(self.wb_fgate.to(self.device)) * x_wb + F.sigmoid(self.c_fgate.to(self.device)) * x_c + F.sigmoid(self.g_fgate.to(self.device)) * x_g) / torch.tensor([3.0]).to(self.device)
    out = torch.cat((x_wb, x_c, x_g), dim=1)
    out = self.fusionGate(out)

    out = F.sigmoid(out)

    return out

    
if __name__ == "__main__":

	model = Autoencoder4()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	#print(ac1.encoder)
	summary(model, (3, 300, 400))
	#summary(model, (3, 28, 28))
