import os
from typing import List

import cv2
import keras
import numpy as np


class Dataset:
	"""CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

	Args:
		images_dir (str): path to images folder
		masks_dir (str): path to segmentation masks folder
		class_values (list): values of classes to extract from segmentation mask
		augmentation (albumentations.Compose): data transfromation pipeline
			(e.g. flip, scale, etc.)
		preprocessing (albumentations.Compose): data preprocessing
			(e.g. noralization, shape manipulation, etc.)
	"""
	CLASSES = ['background', 'covid', 'bacterial', 'viral']

	def __init__(self, images_dir, masks_dir, classes=None, augmentation=None, preprocessing=None):
		exts = ('.jpeg', '.jpg', '.png')
		self.ids = [name for name in os.listdir(images_dir) if name.endswith(exts)]
		self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
		self.masks_fps = [os.path.join(masks_dir, f"{image_id.rsplit('.', 1)[0]}.png") for image_id in self.ids]

		# convert str names to class values on masks
		self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

		self.augmentation = augmentation
		self.preprocessing = preprocessing

	def __getitem__(self, i):
		try:
			image = cv2.imread(self.images_fps[i])
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			mask = cv2.imread(self.masks_fps[i], 0)

			# extract certain classes from mask (e.g. cars)
			masks = [(mask == v) for v in self.class_values]
			mask = np.stack(masks, axis=-1).astype('float')

			# add background if mask is not binary
			if mask.shape[-1] != 1:
				background = 1 - mask.sum(axis=-1, keepdims=True)
				mask = np.concatenate((mask, background), axis=-1)

			# apply augmentations
			if self.augmentation:
				sample = self.augmentation(image=image, mask=mask)
				image, mask = sample['image'], sample['mask']

			# apply preprocessing
			if self.preprocessing:
				sample = self.preprocessing(image=image, mask=mask)
				image, mask = sample['image'], sample['mask']

			return image, mask
		except:
			print(f'ERRO AO LER IMG | i = {i}, imgs = {len(self.images_fps)}, masks = {len(self.masks_fps)}')
			print(f'img = {self.images_fps[i]}, masks = {self.masks_fps[i]}')
			print(f'img = {self.images_fps[i - 1]}, masks = {self.masks_fps[i - 1]}')

	def __len__(self):
		return len(self.ids)


class Dataloader(keras.utils.Sequence):
	"""Load data from dataset and form batches

	Args:
		dataset: instance of Dataset class for image loading and preprocessing.
		batch_size: Integet number of images in batch.
		shuffle: Boolean, if `True` shuffle image indexes each epoch.
	"""

	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.indexes = np.arange(len(dataset))
		self.on_epoch_end()

	def __getitem__(self, i):

		# collect batch data
		start = i * self.batch_size
		stop = (i + 1) * self.batch_size
		data = []
		for j in range(start, stop):
			data.append(self.dataset[j])

		# transpose list of lists
		batch = [np.stack(samples, axis=0) for samples in zip(*data)]
		return batch

	def __len__(self):
		"""Denotes the number of batches per epoch"""
		return len(self.indexes) // self.batch_size

	def on_epoch_end(self):
		"""Callback function to shuffle indexes each epoch"""
		if self.shuffle:
			self.indexes = np.random.permutation(self.indexes)


def build_dataloader(
		path_imgs: str, path_masks: str, classes: List[str],
		batch_size=1, shuffle=False
) -> Dataloader:
	dataset = Dataset(path_imgs, path_masks, classes=classes)
	return Dataloader(dataset, batch_size=batch_size, shuffle=shuffle)
