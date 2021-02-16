from typing import Callable

from keras import Model

from network.params import UNetParams
from unet.models.unet import Unet


def build_unet(params: UNetParams) -> Callable[[], Model]:
	def child():
		return Unet(
			params.backbone.value, classes=params.n_classes,
			activation='sigmoid' if params.n_classes == 1 else 'softmax',
			dropout=params.dropout
		)

	return child
