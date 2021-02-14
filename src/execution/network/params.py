from enum import Enum
from typing import Dict, Optional, List

import keras.optimizers

from enums import Optimizer
from network.backbones import DeeplabBackbone, UNetBackbone
from test_case.case import TestCase


class NetworkParams:
	epochs = 40
	lr = 0.0001
	size = 512
	clip_value = 0.001
	batch = 4
	loss = 'categorical_crossentropy'
	train_shuffle = False
	backbone: Enum = ''
	classes = []

	def __init__(self, case: TestCase, classes: List[str], backbone: Optional[Enum] = None):
		self.classes = classes
		self.n_classes = len(classes) + 1
		self.backbone = backbone
		self.batch = case.batch

		opts: Dict[Optimizer, keras.optimizers.Optimizer] = {
			Optimizer.adam: keras.optimizers.Adam(lr=self.lr, clipnorm=self.clip_value),
			Optimizer.rmsprop: keras.optimizers.RMSprop(lr=self.lr, clipnorm=self.clip_value)
		}
		self.partition = case.partition
		self.format = case.format
		self.opt = opts[case.opt]
		self.dropout = case.dropout


class DeeplabParams(NetworkParams):
	def __init__(
			self, case: TestCase, classes: List[str],
			backbone=DeeplabBackbone.mobile_net
	):
		super().__init__(case, classes, backbone)
		self.os = 16


class UNetParams(NetworkParams):
	def __init__(self, case: TestCase, classes: List[str]):
		backbone = UNetBackbone.vgg19_drop if case.dropout > 0 else UNetBackbone.vgg19
		super().__init__(case, classes, backbone)
