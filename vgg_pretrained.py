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
import math


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
				transforms.ColorJitter(hue=.05, saturation=.05),
				transforms.RandomHorizontalFlip(),
				transforms.RandomRotation(360, resample=Image.BILINEAR),
				transforms.ToTensor(),
				normalize,
				])


		self.transform = transform

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		image_name_with_extension = self.files[idx]
		image = Image.open(self.data_dir + image_name_with_extension)
		image_name = image_name_with_extension.strip('.jpg')

		if self.transform:
			image = self.transform(image)    

		label = self.labels.loc[self.labels['image_id'] == image_name, 'label']
		label = np.array(label)
		label_t = torch.from_numpy(label)[0]
		return(image, label_t)

#----------------------------------------------------
# Below is the data splitter
#----------------------------------------------------


def train_val_test_split(dataset, train_split, val_split, test_split):
	"""
	Split data set into training, validation, and test sets.
	"""
	#if train_split + val_split + test_split != 1:
		#print('Incorrect split sizes')

	# Size of data set
	N = dataset.__len__()

	# Size of train set
	train_size = math.floor(train_split * N)

	# Size of validation set
	val_size = math.floor(val_split * N)

	# List of all data indices
	indices = list(range(N))

	# Random selection of indices for train set
	train_ids = np.random.choice(indices, size=train_size, replace=False)
	train_ids = list(train_ids)

	# Deletion of indices used for train set
	indices = list(set(indices) - set(train_ids))

	# Random selection of indices for validation set
	val_ids = np.random.choice(indices, size=val_size, replace=False)
	val_ids = list(val_ids)

	# Selecting remaining indices for test set
	test_ids = list(set(indices) - set(val_ids))

	# Creating subsets
	train_data = torch.utils.data.Subset(dataset, train_ids)
	val_data = torch.utils.data.Subset(dataset, val_ids)
	test_data = torch.utils.data.Subset(dataset, test_ids)
	return(train_data, val_data, test_data)





#----------------------------------------------------
# Below is the train function
#----------------------------------------------------



def train(data_dir, label_dir, save_dir, epoch, mb, num_class, num_workers = 1, use_cuda = False, conti = False, lr = 1e-3, final_save = True, save_freq = 0, name = None, train_prop = 0.7):
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
	dataset = MelaData(data_dir = data_dir, label_csv = label_dir)
	val_prop = 1 - train_prop
	train_data, val_data, test_data = train_val_test_split(dataset, train_prop, val_prop, 0.0)

	for epoch_num in range(1, epoch+1):
		running_loss = 0.0
		running_corrects = 0.0
		size = 0

		dataloader = DataLoader(train_data, batch_size = mb, shuffle = True, num_workers = num_workers)

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

		if not epoch_num%save_freq:
			mid_name = '{}_{}_loss_{}_acc_{}'.format(str(name), str(epoch_num), str(epoch_loss), str(epoch_acc))
			torch.save(model.state_dict(), os.path.join(save_dir, '{}.pt'.format(mid_name)))
			torch.save(optim.state_dict(), os.path.join(save_dir, '{}.optim.pt'.format(mid_name)))			

		loss_train[epoch_num-1] = epoch_loss
		acc_train[epoch_num-1] = epoch_acc
		print('Train - Loss: {:.4F} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

	if final_save:
		torch.save(model.state_dict(), os.path.join(save_dir, '{}.pt'.format(name)))
		torch.save(optim.state_dict(), os.path.join(save_dir, '{}.optim.pt'.format(name)))
	return(loss_train, acc_train, val_data)

#----------------------------------------------------
# Below is the eval function
#----------------------------------------------------
def test_model(model_dir, val_data, label_dir, batch_size, num_workers = 1, use_cuda = False):
	model = vgg_preloaded(7, use_cuda=use_cuda)
	model.load_state_dict(torch.load(model_dir))
	model = model.cuda() if use_cuda else model

	#dataset = MelaData(data_dir = data_dir, label_csv = label_dir)
	dataloader = DataLoader(val_data, batch_size = batch_size, shuffle = False, num_workers = num_workers)
	loss_fn = torch.nn.CrossEntropyLoss(reduction = 'sum')

	model.eval()
	predictions = [] #Store predictions in here
	class_list = [] #store ground truth here

	running_loss = 0.0
	running_corrects = 0
	count = 0

	pbar = tqdm(dataloader)
	pbar.set_description("[Epoch {}]".format('Validation'))
	for inputs,classes in pbar:
		if use_cuda:
			inputs = inputs.cuda()
			classes = classes.cuda()
		else:
			inputs = inputs
			classes = classes
		outputs = model(inputs)
		loss = loss_fn(outputs,classes) 
		_,preds = torch.max(outputs.data, 1)
		running_loss += loss.cpu().data.item()
		running_corrects += preds.eq(classes.view_as(preds)).sum()
		predictions += list(preds.cpu().data.numpy())
		if use_cuda:
			class_save = classes.cpu().data.numpy()
		else:
			class_save = classes.data.numpy()
		class_list.append(class_save)
		count +=1

	print('Loss: {:.4f} Acc: {:.4f}'.format(running_loss / len(val_data), running_corrects.data.item() / len(val_data)))
	return({'loss': running_loss / len(val_data), 'acc': running_corrects.data.item() / len(val_data), 'predictions': predictions, 'classes': class_list})