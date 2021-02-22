import os
import pathlib
import re
from datetime import datetime
from typing import Callable, Optional

import keras
import pandas as pd
import tensorflow as tf


def get_name(item) -> str:
	"""Função para converter um objeto em string de acordo com seu tipo (otimizador, função de perda, métrica, etc)"""
	output = ''
	if isinstance(item, str):
		output = item
	elif isinstance(item, keras.losses.Loss):
		output = item.name
	elif isinstance(item, keras.optimizers.Optimizer):
		output = re.search("'(.+)'", str(type(item))).group(1).rsplit('.', 1)[1]
	elif isinstance(item, tf.keras.metrics.Metric):
		output = item.name
	elif callable(item):
		output = item.__name__
	return output.lower()


def write_csv_metrics(model_history: dict, out_path: Optional[str] = None, filename: str = None):
	if out_path:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
	data = {
		'_'.join([term for term in key.split('_') if not term.isdigit()]): model_history[key]
		for key in model_history
	}
	data = {
		'epoch': [i + 1 for i in range(len(model_history['loss']))],
		**data
	}
	df = pd.DataFrame.from_dict(data)

	if filename:
		df.to_csv(pathlib.Path().absolute().joinpath(filename), sep=',', index=False)
	return df.to_csv(out_path, sep=',', index=False)


def write_csv_metrics_test(model_history: dict, out_path: Optional[str] = None, filename: str = None):
	if out_path:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
	df = pd.DataFrame(model_history, index=[0])

	if filename:
		df.to_csv(pathlib.Path().absolute().joinpath(filename), sep=',', index=False)
	return df.to_csv(out_path, sep=',', index=False)


def timer(fn_exec_task: Callable[[], None]) -> Callable[[], None]:
	def intercept():
		start = datetime.now()
		print("Start:", start)
		fn_exec_task()
		end = datetime.now()
		print("Duration:", end - start)

	return intercept
