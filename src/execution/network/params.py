from enum import Enum
from typing import Optional, List

import path

from enums import DatasetPartition, DatasetFormat, Optimizer
from network.backbones import DeeplabBackbone, UNetBackbone
from test_case.case import TestCase


class DatasetBasicParams:
	def __init__(self, partition: DatasetPartition, format: DatasetFormat, size=512):
		self.size = size
		self.partition = partition
		self.format = format


class NetworkParams(DatasetBasicParams):
	epochs = 35
	lr = 0.0001
	clip_value = 0.001
	batch = 4
	loss = 'categorical_crossentropy'
	train_shuffle = False
	backbone: Enum = ''
	classes = []

	def __init__(
			self, case: TestCase, classes: List[str],
			backbone: Optional[Enum] = None, size=512
	):
		super().__init__(case.partition, case.format, size)
		self.classes = classes
		self.n_classes = len(classes) + 1
		self.backbone = backbone
		self.batch = case.batch
		self.opt = case.opt
		self.dropout = case.dropout


class ParamsVisualization(DatasetBasicParams):
	'Example: 448x448; 702010; hist; vgg19_drop; rmsprop; batch-4; epochs-35_lr-0.0001; drop-0; clip-0.001; 102'

	def __init__(self, trained_model_path: str, sep='_'):
		params = path.basename(trained_model_path).split(sep)
		self.n_classes = 5
		self.size = int(params[0].split('x')[0])
		self.partition = DatasetPartition(params[1])
		self.format = DatasetFormat(params[2])
		self.backbone = DeeplabBackbone.mobile_net if DatasetFormat(
			params[3]) == DeeplabBackbone.mobile_net.value else UNetBackbone.vgg19_drop
		self.batch = int(params[-6].split('-')[-1])
		self.opt = Optimizer(params[-7])
		self.lr = float(params[-4].split('-')[-1])
		self.loss = NetworkParams.loss
		self.clip_value = float(params[-2].split('-')[-1])
		self.dropout = 0.2 if params[-3].endswith('-1') else 0
		super().__init__(self.partition, self.format, self.size)


class DeeplabParams(NetworkParams):
	def __init__(
			self, case: TestCase, classes: List[str],
			backbone=DeeplabBackbone.mobile_net, size=512
	):
		super().__init__(case, classes, backbone, size)
		self.os = 16


class UNetParams(NetworkParams):
	def __init__(self, case: TestCase, classes: List[str], size=512):
		backbone = UNetBackbone.vgg19_drop
		super().__init__(case, classes, backbone, size)
