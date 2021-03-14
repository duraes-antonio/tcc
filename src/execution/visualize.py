import pathlib
from os import path

import keras
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from backbones import UNetBackbone, DeeplabBackbone
from cli import read_args_visualize
from data.dataset import prepare_datasets, build_dataset_name, build_data_for_visualization
from dataset_dataloader import Dataset
from deeplab import deeplabv3
from enums import Env, Network
from network.common import get_optimizer
from network.unet import get_preprocessing, Unet
from params import ParamsVisualization


def visualize(**images):
	n = len(images)
	plt.figure(figsize=(16, 5))
	for i, (name, image) in enumerate(images.items()):
		plt.subplot(1, n, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.title(' '.join(name.split('_')).title())
		plt.imshow(image)
	plt.show()


def denormalize(x):
	"""Scale image to range 0..1 for correct plot"""
	x_max = np.percentile(x, 98)
	x_min = np.percentile(x, 2)
	x = (x - x_min) / (x_max - x_min)
	x = x.clip(0, 1)
	return x


def plot_images(model: keras.models.Model, dataset: Dataset):
	font = {'family': 'normal', 'weight': 'normal', 'size': 8}
	matplotlib.rc('font', **font)
	n = 15
	ids = np.random.choice(np.arange(len(dataset)), size=n)

	for i in ids:
		image, gt_mask = dataset[i]
		image = np.expand_dims(image, axis=0)
		pr_mask = model.predict(image)
		visualize(
			image=denormalize(image.squeeze()),
			# gt_mask=gt_mask.squeeze(),
			background_mask=gt_mask[..., 0].squeeze(),
			covid_gt=gt_mask[..., 1].squeeze(),
			bacterial_gt=gt_mask[..., 2].squeeze(),
			viral_gt=gt_mask[..., 3].squeeze(),

			background_pr=pr_mask[..., 0].squeeze(),
			covid_pr=pr_mask[..., 1].squeeze(),
			bacterial_pr=pr_mask[..., 2].squeeze(),
			viral_pr=pr_mask[..., 3].squeeze(),
		)


def main():
	classes = ['background', 'covid', 'bacterial', 'viral']
	repository_name = 'tcc'
	path_where = pathlib.Path().absolute()
	path_root = str(path_where).split(repository_name)[0]
	path_datasets = path.join(path_root, 'datasets')
	args = read_args_visualize()
	params = ParamsVisualization(args.path_trained)

	# Baixar e extrair datasets
	prepare_datasets(path_datasets, args.size)
	backbone = UNetBackbone.vgg19_drop.value if args.network == Network.unet else DeeplabBackbone.mobile_net.value

	# Definir params
	if args.network == Network.unet:
		model = Unet(backbone, classes=len(classes) + 1, activation='softmax', dropout=params.dropout)

	else:
		model = deeplabv3(
			input_shape=(params.size, params.size, 3), classes=params.n_classes,
			backbone=backbone, dropout=params.dropout
		)

	# Compilar modelo
	optim = get_optimizer(params.opt, params.lr, params.clip_value)
	model.compile(optimizer=optim, loss=params.loss)

	# Gerar Dataloaders
	preprocess_fn = get_preprocessing(backbone) if args.network == Network.unet else None
	path_dataset = path.join(path_datasets, build_dataset_name(params))
	train_dataset = build_data_for_visualization(path_dataset, classes, Env.train, preprocess_fn)
	val_dataset = build_data_for_visualization(path_dataset, classes, Env.eval, preprocess_fn)
	test_dataset = build_data_for_visualization(path_dataset, classes, Env.test, preprocess_fn)

	# Avaliar modelo
	model.load_weights(args.path_trained)
	plot_images(model, test_dataset)

	return 0


main()
