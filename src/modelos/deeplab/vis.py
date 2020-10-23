# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from typing import Any, List

import numpy as np
import tensorflow as tf
from PIL.Image import Image
from six.moves import range
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import training as contrib_training

from deeplab import common
from deeplab import model
from deeplab.datasets import data_generator
from deeplab.utils import save_annotation

flags = tf.app.flags
FLAGS = flags.FLAGS


def define_initial_flags() -> Any:
	flags = tf.app.flags
	flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

	# Settings for log directories.
	flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')
	flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

	# Settings for visualizing the model.
	flags.DEFINE_integer('vis_batch_size', 1, 'The number of images in each batch during evaluation.')
	flags.DEFINE_list('vis_crop_size', '513,513', 'Crop size [height, width] for visualization.')
	flags.DEFINE_integer('eval_interval_secs', 60 * 5, 'How often (in seconds) to run evaluation.')

	# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
	# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
	# one could use different atrous_rates/output_stride during training/evaluation.
	flags.DEFINE_multi_integer('atrous_rates', None, 'Atrous rates for atrous spatial pyramid pooling.')
	flags.DEFINE_integer('output_stride', 16, 'The ratio of input to output spatial resolution.')

	# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
	flags.DEFINE_multi_float('eval_scales', [1.0], 'The scales to resize images for evaluation.')

	# Change to True for adding flipped images during test.
	flags.DEFINE_bool('add_flipped_images', False, 'Add flipped images for evaluation or not.')

	flags.DEFINE_integer('quantize_delay_step', -1,
						 'Steps to start quantized training. If < 0, will not quantize model.')

	# Dataset settings.
	flags.DEFINE_string('dataset', 'pascal_voc_seg', 'Name of the segmentation dataset.')
	flags.DEFINE_string('vis_split', 'val', 'Which split of the dataset used for visualizing results')
	flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')
	flags.DEFINE_enum('colormap_type', 'pqr', ['pascal', 'cityscapes', 'ade20k', 'pqr'], 'Visualization colormap type.')
	flags.DEFINE_boolean('also_save_raw_predictions', False, 'Also save raw predictions.')
	flags.DEFINE_integer(
		'max_number_of_iterations', 0,
		'Maximum number of visualization iterations. Will loop ''indefinitely upon nonpositive values.'
	)

	return flags


flags = define_initial_flags()

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
	"""Converts the predicted label for evaluation.

	There are cases where the training labels are not equal to the evaluation
	labels. This function is used to perform the conversion so that we could
	evaluate the results on the evaluation server.

	Args:
		prediction: Semantic segmentation prediction.
		train_id_to_eval_id: A list mapping from train id to evaluation id.

	Returns:
		Semantic segmentation prediction whose labels have been changed.
	"""
	converted_prediction = prediction.copy()
	for train_id, eval_id in enumerate(train_id_to_eval_id):
		converted_prediction[prediction == train_id] = eval_id
	return converted_prediction


def run(self, image: Image):
	"""Runs inference on a single image.

	Args:
		image: A PIL.Image object, raw input image.

	Returns:
		resized_image: RGB image resized from original input image.
		seg_map: Segmentation map of `resized_image`.
	"""
	width, height = image.size
	resized_image = image.convert('RGB')
	batch_seg_map = self.sess.run(
		self.OUTPUT_TENSOR_NAME,
		feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]}
	)
	seg_map = batch_seg_map[0]
	return resized_image, seg_map


def _process_batch(
		sess, original_images, semantic_predictions, image_names, image_heights,
		image_widths, image_id_offset, save_dir, raw_save_dir, train_id_to_eval_id=None
):
	"""Evaluates one single batch qualitatively.

	Args:
		sess: TensorFlow session.
		original_images: One batch of original images.
		semantic_predictions: One batch of semantic segmentation predictions.
		image_names: Image names.
		image_heights: Image heights.
		image_widths: Image widths.
		image_id_offset: Image id offset for indexing images.
		save_dir: The directory where the predictions will be saved.
		raw_save_dir: The directory where the raw predictions will be saved.
		train_id_to_eval_id: A list mapping from train id to eval id.
	"""
	run_return = sess.run([original_images, semantic_predictions, image_names, image_heights, image_widths])
	(original_images, semantic_predictions, image_names, image_heights, image_widths) = run_return
	num_image = semantic_predictions.shape[0]

	for i in range(num_image):
		image_height = np.squeeze(image_heights[i])
		image_width = np.squeeze(image_widths[i])
		original_image = np.squeeze(original_images[i])
		semantic_prediction = np.squeeze(semantic_predictions[i])
		predict = semantic_prediction[:image_height, :image_width]

		# Save image.
		save_annotation.save_annotation(original_image, save_dir, image_names[i], False)

		# Save prediction.
		save_annotation.save_annotation(predict, save_dir, image_names[i], True, colormap_type=FLAGS.colormap_type)

		if FLAGS.also_save_raw_predictions:
			image_filename = os.path.basename(image_names[i])

			if train_id_to_eval_id is not None:
				predict = _convert_train_id_to_eval_id(predict, train_id_to_eval_id)

			save_annotation.save_annotation(predict, raw_save_dir, image_filename, False)


# def main(unused_argv):
# 	tf.logging.set_verbosity(tf.logging.INFO)
#
# 	# Get dataset-dependent information.
# 	dataset = data_generator.Dataset(
# 		dataset_name=FLAGS.dataset,
# 		split_name=FLAGS.vis_split,
# 		dataset_dir=FLAGS.dataset_dir,
# 		batch_size=FLAGS.vis_batch_size,
# 		crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
# 		min_resize_value=FLAGS.min_resize_value,
# 		max_resize_value=FLAGS.max_resize_value,
# 		resize_factor=FLAGS.resize_factor,
# 		model_variant=FLAGS.model_variant,
# 		is_training=False,
# 		should_shuffle=False,
# 		should_repeat=False
# 	)
#
# 	train_id_to_eval_id = None
#
# 	# Prepare for visualization.
# 	tf.gfile.MakeDirs(FLAGS.vis_logdir)
#
# 	save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
# 	tf.gfile.MakeDirs(save_dir)
#
# 	raw_save_dir = os.path.join(FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
# 	tf.gfile.MakeDirs(raw_save_dir)
#
# 	tf.logging.info('Visualizing on %s set', FLAGS.vis_split)
#
# 	with tf.Graph().as_default():
# 		samples = dataset.get_one_shot_iterator().get_next()
#
# 		model_options = common.ModelOptions(
# 			outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
# 			crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
# 			atrous_rates=FLAGS.atrous_rates,
# 			output_stride=FLAGS.output_stride
# 		)
#
# 		if tuple(FLAGS.eval_scales) == (1.0,):
# 			tf.logging.info('Performing single-scale test.')
# 			predictions = model.predict_labels(
# 				samples[common.IMAGE],
# 				model_options=model_options,
# 				image_pyramid=FLAGS.image_pyramid
# 			)
# 		else:
# 			tf.logging.info('Performing multi-scale test.')
# 			if FLAGS.quantize_delay_step >= 0:
# 				raise ValueError('Quantize mode is not supported with multi-scale test.')
# 			predictions = model.predict_labels_multi_scale(
# 				samples[common.IMAGE],
# 				model_options=model_options,
# 				eval_scales=FLAGS.eval_scales,
# 				add_flipped_images=FLAGS.add_flipped_images
# 			)
# 		predictions = predictions[common.OUTPUT_TYPE]
#
# 		if FLAGS.min_resize_value and FLAGS.max_resize_value:
# 			# Only support batch_size = 1, since we assume the dimensions of original
# 			# image after tf.squeeze is [height, width, 3].
# 			assert FLAGS.vis_batch_size == 1
#
# 			# Reverse the resizing and padding operations performed in preprocessing.
# 			# First, we slice the valid regions (i.e., remove padded region) and then
# 			# we resize the predictions back.
# 			original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
# 			original_image_shape = tf.shape(original_image)
# 			predictions = tf.slice(
# 				predictions,
# 				[0, 0, 0],
# 				[1, original_image_shape[0], original_image_shape[1]]
# 			)
# 			resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]), tf.squeeze(samples[common.WIDTH])])
# 			predictions = tf.squeeze(
# 				tf.image.resize_images(
# 					tf.expand_dims(predictions, 3), resized_shape,
# 					method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True
# 				),
# 				3
# 			)
#
# 		tf.train.get_or_create_global_step()
# 		if FLAGS.quantize_delay_step >= 0:
# 			contrib_quantize.create_eval_graph()
#
# 		num_iteration = 0
# 		max_num_iteration = FLAGS.max_number_of_iterations
# 		checkpoints_iterator = contrib_training.checkpoints_iterator(
# 			FLAGS.checkpoint_dir, min_interval_secs=FLAGS.eval_interval_secs
# 		)
# 		for checkpoint_path in checkpoints_iterator:
# 			num_iteration += 1
# 			tf.logging.info(
# 				'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
# 			tf.logging.info('Visualizing with model %s', checkpoint_path)
#
# 			scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
# 			session_creator = tf.train.ChiefSessionCreator(
# 				scaffold=scaffold, master=FLAGS.master,
# 				checkpoint_filename_with_path=checkpoint_path
# 			)
# 			with tf.train.MonitoredSession(session_creator=session_creator, hooks=None) as sess:
# 				batch = 0
# 				image_id_offset = 0
#
# 				while not sess.should_stop():
# 					tf.logging.info('Visualizing batch %d', batch + 1)
# 					_process_batch(
# 						sess=sess,
# 						original_images=samples[common.ORIGINAL_IMAGE],
# 						semantic_predictions=predictions,
# 						image_names=samples[common.IMAGE_NAME],
# 						image_heights=samples[common.HEIGHT],
# 						image_widths=samples[common.WIDTH],
# 						image_id_offset=image_id_offset,
# 						save_dir=save_dir,
# 						raw_save_dir=raw_save_dir,
# 						train_id_to_eval_id=train_id_to_eval_id)
# 					image_id_offset += FLAGS.vis_batch_size
# 					batch += 1
#
# 			tf.logging.info(
# 				'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
# 			if 0 < max_num_iteration <= num_iteration:
# 				break

def _process_batch2(
		sess, original_images, semantic_predictions, image_names, image_heights,
		image_widths, train_id_to_eval_id=None
) -> List[Image]:
	"""Evaluates one single batch qualitatively.

	Args:
		sess: TensorFlow session.
		original_images: One batch of original images.
		semantic_predictions: One batch of semantic segmentation predictions.
		image_names: Image names.
		image_heights: Image heights.
		image_widths: Image widths.
		train_id_to_eval_id: A list mapping from train id to eval id.
	"""
	run_return = sess.run([original_images, semantic_predictions, image_names, image_heights, image_widths])
	(original_images, semantic_predictions, image_names, image_heights, image_widths) = run_return
	num_image = semantic_predictions.shape[0]

	for i in range(num_image):
		image_height = np.squeeze(image_heights[i])
		image_width = np.squeeze(image_widths[i])
		original_image = np.squeeze(original_images[i])
		semantic_prediction = np.squeeze(semantic_predictions[i])
		predict = semantic_prediction[:image_height, :image_width]

		img_orig = save_annotation.build_annotation(label=original_image, add_colormap=False)
		img_pred = save_annotation.build_annotation(label=predict, add_colormap=True, colormap_type=FLAGS.colormap_type)

		if FLAGS.also_save_raw_predictions:
			if train_id_to_eval_id is not None:
				predict = _convert_train_id_to_eval_id(predict, train_id_to_eval_id)

			img_pred_raw = save_annotation.build_annotation(label=predict, add_colormap=False)
			return [img_orig, img_pred, img_pred_raw]

		else:
			return [img_orig, img_pred]


def main(unused_argv):
	tf.logging.set_verbosity(tf.logging.INFO)

	# Get dataset-dependent information.
	dataset = data_generator.Dataset(
		dataset_name=FLAGS.dataset,
		split_name=FLAGS.vis_split,
		dataset_dir=FLAGS.dataset_dir,
		batch_size=FLAGS.vis_batch_size,
		crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
		min_resize_value=FLAGS.min_resize_value,
		max_resize_value=FLAGS.max_resize_value,
		resize_factor=FLAGS.resize_factor,
		model_variant=FLAGS.model_variant,
		is_training=False,
		should_shuffle=False,
		should_repeat=False
	)

	train_id_to_eval_id = None

	# Prepare for visualization.
	tf.gfile.MakeDirs(FLAGS.vis_logdir)

	save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
	tf.gfile.MakeDirs(save_dir)

	raw_save_dir = os.path.join(FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
	tf.gfile.MakeDirs(raw_save_dir)

	tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

	with tf.Graph().as_default():
		samples = dataset.get_one_shot_iterator().get_next()

		model_options = common.ModelOptions(
			outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
			crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
			atrous_rates=FLAGS.atrous_rates,
			output_stride=FLAGS.output_stride
		)

		if tuple(FLAGS.eval_scales) == (1.0,):
			tf.logging.info('Performing single-scale test.')
			predictions = model.predict_labels(
				samples[common.IMAGE],
				model_options=model_options,
				image_pyramid=FLAGS.image_pyramid
			)
		else:
			tf.logging.info('Performing multi-scale test.')
			if FLAGS.quantize_delay_step >= 0:
				raise ValueError('Quantize mode is not supported with multi-scale test.')
			predictions = model.predict_labels_multi_scale(
				samples[common.IMAGE],
				model_options=model_options,
				eval_scales=FLAGS.eval_scales,
				add_flipped_images=FLAGS.add_flipped_images
			)
		predictions = predictions[common.OUTPUT_TYPE]

		if FLAGS.min_resize_value and FLAGS.max_resize_value:
			# Only support batch_size = 1, since we assume the dimensions of original
			# image after tf.squeeze is [height, width, 3].
			assert FLAGS.vis_batch_size == 1

			# Reverse the resizing and padding operations performed in preprocessing.
			# First, we slice the valid regions (i.e., remove padded region) and then
			# we resize the predictions back.
			original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
			original_image_shape = tf.shape(original_image)
			predictions = tf.slice(
				predictions,
				[0, 0, 0],
				[1, original_image_shape[0], original_image_shape[1]]
			)
			resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]), tf.squeeze(samples[common.WIDTH])])
			predictions = tf.squeeze(
				tf.image.resize_images(
					tf.expand_dims(predictions, 3), resized_shape,
					method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True
				),
				3
			)

		tf.train.get_or_create_global_step()
		if FLAGS.quantize_delay_step >= 0:
			contrib_quantize.create_eval_graph()

		num_iteration = 0
		max_num_iteration = FLAGS.max_number_of_iterations
		checkpoints_iterator = contrib_training.checkpoints_iterator(
			FLAGS.checkpoint_dir, min_interval_secs=FLAGS.eval_interval_secs
		)
		for checkpoint_path in checkpoints_iterator:
			num_iteration += 1
			tf.logging.info(
				'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
			tf.logging.info('Visualizing with model %s', checkpoint_path)

			scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
			session_creator = tf.train.ChiefSessionCreator(
				scaffold=scaffold, master=FLAGS.master,
				checkpoint_filename_with_path=checkpoint_path
			)
			with tf.train.MonitoredSession(session_creator=session_creator, hooks=None) as sess:
				batch = 0
				image_id_offset = 0

				while not sess.should_stop():
					tf.logging.info('Visualizing batch %d', batch + 1)
					imagens = _process_batch(
						sess=sess,
						original_images=samples[common.ORIGINAL_IMAGE],
						semantic_predictions=predictions,
						image_names=samples[common.IMAGE_NAME],
						image_heights=samples[common.HEIGHT],
						image_widths=samples[common.WIDTH],
						image_id_offset=image_id_offset,
						train_id_to_eval_id=train_id_to_eval_id,
						save_dir=save_dir,
						raw_save_dir=raw_save_dir
					)
					image_id_offset += FLAGS.vis_batch_size
					batch += 1

			tf.logging.info('Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
			if 0 < max_num_iteration <= num_iteration:
				break


if __name__ == '__main__':
	flags.mark_flag_as_required('checkpoint_dir')
	flags.mark_flag_as_required('vis_logdir')
	flags.mark_flag_as_required('dataset_dir')
	tf.app.run()
