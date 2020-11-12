********************************************************************************
ECE 570 Artificial Intelligence - Course project
Fall 2020 
Purdue University

Author: Manu Ramesh
********************************************************************************

Dataset can be downloaded from:

https://sites.google.com/view/reside-dehaze-datasets/reside-v0
I have taken the Indoor Training Set (ITS)

Links of implementations of selected papers:

GAN Paper
Enhanced Pix2Pix Dehazing Network: Qu et al:
https://github.com/ErinChen1/EPDN

Autoencoder Papers:

Multi-scale Single Image Dehazing using Perceptual Pyramid Deep Network: Zhang et al:
https://github.com/hezhangsprinter/NTIRE-2018-Dehazing-Challenge

Gated Fusion Network for Single Image Dehazing: Ren et al:
https://github.com/rwenqi/GFN-dehazing
https://sites.google.com/site/renwenqi888/research/dehazing/gfn

--------------------------------------------------------------------------------

File descriptions

> annotate.py - 
	creates annotation file for the ITS dataset
	creates csv files with format
	hazy image, its corresponding clear image

	Run the functions >create_main_dataset
			  >create_smaller_dataset
	in the same order.

> my_models.py - 
	Has all models that I have used in experiments except 'Model 2: Zhang'.

> dehaze1113.py - 
	Taken from implementation of Zhang et al paper. 
	From this file, I use Dense_rain_cvprw3 model as my Model 2.

> myTransforms.py -
	Has some classes that can be used as transforms.
	I wans't using them as torch transforms though.
	I was calling these functions from within the model.
	These can be further optimized and called as torch vision transforms but make sure that only the hazy images are transformed while the clear images stay how they are.

> psnr.py - 
	a file implementing psnr and ssim. I am not using this. 
	I use implementation by Zhang et al in metrics.py.

> metrics.py - 
	Zhang et al's implementation of psnr.
	I use this implementation for all comparisons.

> ITS_Dataset.py - 
	Has the dataset class of Indoor Training Set.

> its_trial.py - 
	Has functions for creating data loaders, training, testing and sampling.
	Make sure you change the model in get_model() function
	Also change  modelName and number of epocs at the very end under the main block.
	So, you should change three parameters for changing models.

> its_trial2.py -
	Exactly same as its_trial.py, but runs Model 4 - Zhang Ren 2 on cpu, as gpu memory was insufficient.

> utils.py - 
	Has some utility functions to measure stats of evaluations.

------------------------------------------------------------------------------------


