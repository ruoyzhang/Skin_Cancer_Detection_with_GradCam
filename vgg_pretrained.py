import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models, transforms, utils
from PIL import Image
from tqdm import tqdm
from torch.optim import Adam


class vgg_preloaded(nn.Module):

	def __init__(self, num_class, use_cuda):
		super(vgg_preloaded, self).__init__()
		self.use_cuda = use_cuda
		self.num_class = num_class
		self.dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
		model = models.vgg16(pretrained=True)
		self.model = model.cuda() if self.use_cuda else model
		for param in self.model.features.parameters():
			param.require_grad = False
		num_features = self.model.classifier[6].in_features
		features = list(self.model.classifier.children())[:-1] # Remove last layer
		features.extend([nn.Linear(num_features, self.num_class)])
		self.model.classifier = nn.Sequential(*features)
		self.model.classifier.require_grad = True

	def forward(self, inp):
		return(self.model(inp))

#----------------------------------------------------
# Below is the data loader
#----------------------------------------------------


class MelaData(Dataset):
	"""MelaData dataset."""

	def __init__(self, data_dir, label_csv, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with labels.
			data_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied on a sample: use prep1
		"""

		self.data_dir = data_dir
		self.files = os.listdir(data_dir)

		labels = pd.read_csv(label_csv)
		dx_to_num = {'nv' : 0, 'mel': 1, 'bkl': 2, 'df': 3, 'akiec': 4, 'bcc': 5, 'vasc' : 6}        
		labels['label'] = labels['dx'].apply(lambda x: dx_to_num[x])
		self.labels = labels

		if transform is None:
			normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			transform = transforms.Compose([
				transforms.Resize(224),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
				])


		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		image_name_with_extension = self.files[idx]
		image = Image.open(data_dir + image_name_with_extension)
		image_name = image_name_with_extension.strip('.jpg')

		if self.transform:
			image = self.transform(image)    

		label = self.labels.loc[self.labels['image_id'] == image_name, 'label']
		label = np.array(label)
		label_t = torch.from_numpy(label)[0]
		return(image, label_t)


#----------------------------------------------------
# Below is the train function
#----------------------------------------------------



def train(data_dir, label_dir, save_dir, epoch, mb, num_class, num_workers = 1, use_cuda = False, conti = False, lr = 1e-3, save = True, name = None):
	# instantiate the vgg model
	model = vgg_preloaded(num_class, use_cuda)

	if name is None:
		name = 'model'

	# if dir does not exit, make it:
	if not os.path.isdir(save_dir):
		os.mkdir(save_dir)

	# define model path
	modelpath = os.path.join(save_dir, '{}.pt'.format(name))

	# do we wanna continue to train
	if os.path.isfile(modelpath) and conti:
		model.load_state_dict(torch.load(modelpath))
	if use_cuda:
		model = model.cuda()
	model.train()

	loss_train = np.zeros(epoch)
	acc_train = np.zeros(epoch)
	loss_fun = torch.nn.CrossEntropyLoss(reduction = 'sum')
	optim = Adam(model.parameters(), lr = lr)

	for epoch_num in range(1, epoch+1):
		running_loss = 0.0
		running_corrects = 0.0
		size = 0

		dataset = MelaData(data_dir = data_dir, label_csv = label_dir)
		dataloader = DataLoader(dataset, batch_size = mb, shuffle = True, num_workers = num_workers)

		pbar = tqdm(dataloader)
		pbar.set_description("[Epoch {}]".format(epoch_num))
		for inputs, labels in pbar:
			bs = labels.size(0)
			if use_cuda:
				inputs = inputs.cuda()
				labels = labels.cuda()
			output = model(inputs)
			_, preds = torch.max(output.data, 1)
			loss = loss_fun(output, labels)
			running_loss += loss
			running_corrects += preds.eq(labels.view_as(preds)).sum()
			optim.zero_grad()
			loss.backward()
			optim.step()
			size += bs

		epoch_loss = running_loss / size
		epoch_acc = running_corrects.item() / size
		loss_train[epoch_num-1] = epoch_loss
		acc_train[epoch_num-1] = epoch_acc
		print('Train - Loss: {:.4F} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

	if save:
		torch.save(model.state_dict(), os.path.join(save_dir, '{}.pt'.format(name)))
		torch.save(optim.state_dict(), os.path.join(save_dir, '{}.optim.pt'.format(name)))
	return(loss_train, acc_train)

#----------------------------------------------------
# Below is the eval function
#----------------------------------------------------

