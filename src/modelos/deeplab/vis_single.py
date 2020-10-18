from io import BytesIO

import numpy as np
import PIL.Image as PILImage
import tensorflow as tf
import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL.Image import Image
from typing import Tuple, Union


def create_pascal_label_colormap():
	"""Creates a label colormap used in PASCAL VOC segmentation benchmark.

	Returns:
		A Colormap for visualizing segmentation results.
	"""
	colormap = np.zeros((256, 3), dtype=int)
	ind = np.arange(256, dtype=int)

	for shift in reversed(range(8)):
		for channel in range(3):
			colormap[:, channel] |= ((ind >> channel) & 1) << shift
		ind >>= 3

	return colormap


def label_to_color_image(label):
	"""Adds color defined by the dataset colormap to the label.

	Args:
		label: A 2D array with integer type, storing the segmentation label.

	Returns:
		result: A 2D array with floating type. The element of the array
			is the color indexed by the corresponding element in the input label
			to the PASCAL color map.
	Raises:
		ValueError: If label is not of rank 2 or its value is larger
			than color map maximum entry.
	"""
	if label.ndim != 2:
		raise ValueError('Expect 2-D input label')

	colormap = create_pascal_label_colormap()

	if np.max(label) >= len(colormap):
		raise ValueError('label value too large.')

	return colormap[label]


def open_model_frozen(path_model_frozen: str) -> tf.Session:
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'
	graph = tf.Graph()
	graph_def = None

	if FROZEN_GRAPH_NAME not in path_model_frozen:
		raise ValueError(f"The file name is not within the accepted standard. Rename it to '{FROZEN_GRAPH_NAME}'.")

	with open(path_model_frozen, 'rb') as model_frozen_file:
		graph_def = tf.GraphDef.FromString(model_frozen_file.read())

	with graph.as_default():
		tf.import_graph_def(graph_def, name='')

	return tf.Session(graph=graph)


def run(
		img: Image, session: tf.Session, INPUT_SIZE=600,
		INPUT_TENSOR='ImageTensor:0', OUTPUT_TENSOR='SemanticPredictions:0'
) -> Tuple[Image, Union]:
	width, height = img.size
	resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
	target_size = (int(resize_ratio * width), int(resize_ratio * height))
	resized_image = img.convert('RGB').resize(target_size, PILImage.ANTIALIAS)
	batch_seg_map = session.run(
		OUTPUT_TENSOR,
		feed_dict={INPUT_TENSOR: [np.asarray(resized_image)]}
	)
	seg_map = batch_seg_map[0]
	print(seg_map)
	return resized_image, seg_map


class DeepLabModel(object):
	"""Class to load deeplab model and run inference."""

	INPUT_TENSOR_NAME = 'ImageTensor:0'
	OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
	FROZEN_GRAPH_NAME = 'frozen_inference_graph'

	def __init__(self, path_model_frozen: str):
		self.sess = open_model_frozen(path_model_frozen)

	def run(self, image: Image):
		return run(image, self.sess)


def vis_segmentation(image: Image, seg_map: Union):
	"""Visualizes input image, segmentation map and overlay view."""
	plt.figure(figsize=(15, 5))
	grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

	plt.subplot(grid_spec[0])
	plt.imshow(image)
	plt.axis('off')
	plt.title('input image')

	plt.subplot(grid_spec[1])
	seg_image = label_to_color_image(seg_map).astype(np.uint8)
	plt.imshow(seg_image)
	plt.axis('off')
	plt.title('segmentation map')

	plt.subplot(grid_spec[2])
	plt.imshow(image)
	plt.imshow(seg_image, alpha=0.7)
	plt.axis('off')
	plt.title('segmentation overlay')

	unique_labels = np.unique(seg_map)
	ax = plt.subplot(grid_spec[3])
	plt.imshow(
		FULL_COLOR_MAP[unique_labels].astype(np.uint8),
		interpolation='nearest'
	)
	ax.yaxis.tick_right()
	plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
	plt.xticks([], [])
	ax.tick_params(width=0.0)
	plt.grid('off')
	plt.show()


def run_visualization(url: str):
	"""Inferences DeepLab model and visualizes result."""
	try:
		f = urllib.request.urlopen(url)
		jpeg_str = f.read()
		original_im = Image.open(BytesIO(jpeg_str))
	except IOError:
		print(f'Cannot retrieve image. Please check url: {url}')
		return

	print(f'running deeplab on image {url}...')
	resized_im, seg_map = MODEL.run(original_im)
	vis_segmentation(resized_im, seg_map)


LABEL_NAMES = np.asarray(['background', 'bacterial', 'covid-19', 'virus', 'x'])
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL = DeepLabModel('')

SAMPLE_IMAGE = 'image1'  # @param ['image1', 'image2', 'image3']
IMAGE_URL = ''  # @param {type:"string"}
_SAMPLE_URL = ('https://github.com/tensorflow/models/blob/master/research/'
			   'deeplab/g3doc/img/%s.jpg?raw=true')

image_url = IMAGE_URL or _SAMPLE_URL % SAMPLE_IMAGE
run_visualization(image_url)
