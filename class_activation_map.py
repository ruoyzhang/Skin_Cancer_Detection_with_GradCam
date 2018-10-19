#------------------------------------------------------------------------------------------------------
# the following code has been inspired and derived from the original codes of @jacobgil found here:
# https://github.com/jacobgil/pytorch-grad-cam
#------------------------------------------------------------------------------------------------------

import torch
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

class ExtractFeatures:
	"""
	Class for mannually passing the input image through the model
	the gradient at the identified layer will be registered and saved
	"""
	def __init__(self, model, target_layer):
		self.model = model
		self.target_layer = target_layer
		self.gradients = []

	def save_grad(self, grad):
		self.gradients.append(grad)

	def record_and_feed_forward(self, inp):
		# we want to reset the gradient every time we call the method
		self.gradients = []
		layer_output = None
		for name, layer in self.model.features._modules.items():
			# here we mannually pass the input through the NN until the identified layer
			inp = layer(inp)
			if name == self.target_layer:
				# here by registering the hook, we tag the layer and notify torch that we want to keep the gradient for this layer
				inp.register_hook(self.save_grad)
				# record the output at the identified layer, the output is the same as the input for the next layer (activation: final ReLU for VGG before classif)
				layer_output = inp
		# need to flatten before the fully connected layers in the classifier half
		inp = inp.view(inp.size(0), -1)
		# classify
		inp = self.model.classifier(inp)
		return(layer_output, inp)

class GradCam:
	"""
	Class for generating the CAM
	"""

	def __init__(self, model, target_layer, cuda = True):
		self.cuda = cuda
		self.model = model.cuda() if self.cuda else model
		# setting the model to the eval mode so no gradients are hurt during the making of this class
		self.model.eval()
		self.extractfeatures = ExtractFeatures(self.model, target_layer)

	def forward(self, img):
		return(self.model(img))

	def generate_CAM(self, img, class_code = None):
		img = img.cuda() if self.cuda else img
		feature, output = self.extractfeatures.record_and_feed_forward(img)

		if class_code is None:
			class_code = np.argmax(output.cpu().data.numpy())

		activation_value = output[0][class_code]

		# reset grad
		self.model.features.zero_grad()
		self.model.classifier.zero_grad()
		# backpro
		activation_value.backward()

		# we then obtain the gradient of the identified layer
		grads = self.extractfeatures.gradients[-1].cpu().data.numpy()
		grads = np.mean(grads, axis = (2,3))[0]

		target_layer = feature.cpu().data.numpy()[0] # in case the input target_layer is multiple
		cam = np.zeros(target_layer.shape[1 : ], dtype = np.float32)

		for i, w in enumerate(grads):
			cam += w * target_layer[i, :, :]
		
		# the ReLU activation taking advantage of python broadcasting
		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (600, 450))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return(cam)







