import gc
from typing import List

import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


# taken from old keras source code
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


def get_callbacks(path_save_model: str) -> List[Callback]:
	path_with_ext = f"{path_save_model}{'.h5' if path_save_model.endswith('.h5') else ''}"
	return [
		keras.callbacks.ModelCheckpoint(
			path_with_ext, save_weights_only=True,
			save_best_only=True, mode='min'
		),
		keras.callbacks.ReduceLROnPlateau(),
		GarbageCollectorCallback()
	]


def get_metrics(n_classes: int) -> List[keras.metrics]:
	return [
		'accuracy',
		tf.keras.metrics.Recall(name='recall'),
		tf.keras.metrics.Precision(name='precision'),
		ModifiedMeanIOU(num_classes=n_classes + 1, name='miou'),
		f1_score
	]
