import gc
from os import path
from pathlib import Path
from typing import List, Dict, Union, Callable

import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from keras import Model
from tensorflow.keras.callbacks import Callback

from enums import Metrics, Optimizer, Network
from .deeplab import build_deeplab
from .params import DeeplabParams, UNetParams
from .unet.unet_wrapper import build_unet


def f1_score(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	recall = true_positives / (possible_positives + K.epsilon())
	f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
	return f1_val


class ModifiedMeanIOU(tf.keras.metrics.MeanIoU):
	def update_state(self, y_true, y_pred, sample_weight=None):
		return super().update_state(
			tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight
		)


class GarbageCollectorCallback(Callback):
	def on_epoch_end(self, epoch, logs=None):
		gc.collect()


def build_network(net: Network, config: Union[DeeplabParams, UNetParams]) -> Model:
	handlers: Dict[Network, Callable[[], Model]] = {
		Network.unet: build_unet(config),
		Network.deeplab: build_deeplab(config)
	}
	return handlers[net]()


def get_callbacks(path_save_model: str) -> List:
	Path(path.dirname(path_save_model)).mkdir(parents=True, exist_ok=True)
	path_with_ext = f"{path_save_model}{'' if path_save_model.endswith('.h5') else '.h5'}"
	return [
		keras.callbacks.ModelCheckpoint(
			path_with_ext, save_weights_only=True,
			save_best_only=True, mode='min'
		),
		keras.callbacks.ReduceLROnPlateau(),
		GarbageCollectorCallback()
	]


def get_metrics(n_classes: int) -> List:
	return [
		Metrics.accuracy.value,
		tf.keras.metrics.Recall(name=Metrics.recall.value),
		tf.keras.metrics.Precision(name=Metrics.precision.value),
		ModifiedMeanIOU(num_classes=n_classes + 1, name=Metrics.miou.value),
		f1_score
	]


def get_optimizer(opt: Optimizer, lr: float, clip_value: float) -> keras.optimizers.Optimizer:
	options = {
		Optimizer.adam: keras.optimizers.Adam(lr, clipnorm=clip_value),
		Optimizer.rmsprop: keras.optimizers.RMSprop(lr, clipnorm=clip_value)
	}
	return options[opt]
