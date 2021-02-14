from enum import Enum


class DeeplabBackbone(Enum):
	mobile_net = 'mobilenetv2'
	xception = 'xception'


class UNetBackbone(Enum):
	vgg19 = 'vgg19'
	vgg19_drop = 'vgg19_drop'
