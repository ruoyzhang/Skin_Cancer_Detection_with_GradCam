import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randint
import pandas as pd


def select_images(labels):
	images = []
	cancer_types = list(labels.dx.unique())
	for c in cancer_types:
		c_data = labels[labels.dx == c]
		N = c_data.shape[0]
		image_loc = randint(0, N)
		image = c_data.iloc[image_loc].image_id
		image = image + '.jpg'
		images.append(image)
	return images

def plot_images(image_dir, images, cancer_types):
	fig = plt.figure(figsize=(26, 16))
	for i in range(0, len(cancer_types)):
		plt.subplot(2, 4, i + 1)
		plt.title(cancer_types[i])
		plt.axis('off')
		img = mpimg.imread(image_dir + images[i])
		imgplot = plt.imshow(img)

	plt.show()
