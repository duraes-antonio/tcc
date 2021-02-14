from enum import Enum


class Network(Enum):
	deeplab = 'deeplab'
	unet = 'unet'


class DatasetPartition(Enum):
	train_70_eval_20_test_10 = '702010'
	train_80_eval_10_test_10 = '801010'


class DatasetFormat(Enum):
	equal_hist = 'hist'
	morp_transf = 'morf'


class Optimizer(Enum):
	adam = 'adam'
	rmsprop = 'rmsprop'


class Env(Enum):
	train = 'train'
	eval = 'eval'
	test = 'test'


class State(Enum):
	busy = 'busy'
	done = 'done'
	free = 'free'


class TestProgress(Enum):
	start = 'start'
	end = 'end'
