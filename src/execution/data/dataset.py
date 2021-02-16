from os import path
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional, Callable
from zipfile import ZipFile

import wget

from enums import DatasetPartition, DatasetFormat, Env
from network.dataset_dataloader import Dataloader, build_dataloader
from network.params import NetworkParams


class DatasetDownload:
	def __init__(
			self, prefix_name: str, partition: DatasetPartition,
			fmt: DatasetFormat, size: Union[Tuple[int, int], int],
			url_download: str
	):
		self.partition = partition
		self.format = fmt
		self.prefix = prefix_name
		self.size: Tuple[int, int] = [size, size] if isinstance(size, int) else size
		self.url = url_download

	def get_name(self) -> str:
		size = 'x'.join([str(s) for s in self.size])
		return '_'.join([self.prefix, size, self.partition.value, self.format.value])


def get_dataset_512x512(prefix='pneumonia') -> List[DatasetDownload]:
	size = 512
	return [
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.equal_hist, size,
			'https://uc3c4b0a0ca8ba49c2425747255f.dl.dropboxusercontent.com/cd/0/get/BJFjOq2ZHpyHx_drZuv1cZaVUByHCt2wkEOjO4K_QfQued1hsU0wFEBPnV-mZP227oTcVXx8jFSBGqul4W5EuLU_cbxX5XFzXlN77a0u5Tkl07tU_lLC9iYsaiGliAlPHQU/file#',
		),
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.morp_transf, size,
			'https://uc18668b9d15fbaeafa6ea7d32ea.dl.dropboxusercontent.com/cd/0/get/BJFcDCk03JJ9klUuJ2l3akhN8MXiimIAz9F3fPrfDkO-R95sMCm6NwGtugKsPS5iXNL3RnvaIIeBxofllRieT2DLoBAA-52na5MxyejeByD3ed5zYsjJ3eMiySotEcCQlek/file#',
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.equal_hist, size,
			'https://uc9a11a8951effd828167e1b04c9.dl.dropboxusercontent.com/cd/0/get/BJF1wGKgoxnLszmLOLUysOaI1AFNEMQZHAn2H_NTxL7xJd2RlPjwYXsoHsI72muTAflYIXU4HfNrE3E_wKPUVhagGvl2KfaJ9BdMzI6PfwqGQgFtEw4d5b07imufYTMooM8/file#'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.morp_transf, size,
			'https://ucf7d617260b33de44ece0136292.dl.dropboxusercontent.com/cd/0/get/BJFYypoXS3qP9e1v4lc9aIZyWuIBaHZ-DspqsUZRADoyQelPSJY-kVsWrAP8a_2a9VTUqFto1Mn8STNhq9WbjQC53tThQQOTNeqzsazobG8COxcHICs5yjRgO2z-mpjUQhc/file#'
		),
	]


def get_dataset_448x448(prefix='pneumonia') -> List[DatasetDownload]:
	size = 448
	return [
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.equal_hist, size,
			'https://ucbeaceceb81dd75a2cff03c4a3c.dl.dropboxusercontent.com/cd/0/get'
			'/BJBN2xAxTA6ecM2SOyEI1LphASmxljggHJa6rk_J58l7YT0kzG7K6OhN0ESRwMXn3rzWLu4lcHKsiI0MWDxLTgjCcgo86VANRV2ie5U3C2kvlhEFkvFge9XHDA1RBiuTpqs/file# '
		),
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.morp_transf, size,
			'https://ucaab2200f9cd2a25395a3f867ff.dl.dropboxusercontent.com/cd/0/get'
			'/BJCHeV1A8SvjPlq7DEcwYDziqQWXvkWYZUJ4ggEWNLfF6mXTQJRiLf2iiTLEcNyxr4u3JR9CVUvRMKN9qoFV3XikFHKAMTR2hQLJfeFDqzKv1LolKYLsYMMZ-AA2cIar_pA/file# '
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.equal_hist, size,
			'https://uc6c5cf8ef83708e40a6b350a6d7.dl.dropboxusercontent.com/cd/0/get/BJCKHQL-vVjf85ub0aSElsEq'
			'-XTsmr4aSVPk5cO7R70q0WHqtV9Whb12T9gkeLkgpNSNuQWViunlIILdwSWQzDJmroMPZlCp1XIWV9rL0mf6yOvma32bNNekqn'
			'-qYgr7rYo/file# '
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.morp_transf, size,
			'https://ucb4819599fb7269550aa1ca84ab.dl.dropboxusercontent.com/cd/0/get/BJDxOHnqMhSmfmTPRbfPm8UFHxjF91K'
			'-7ieP6UPBno3lhxVpcdVLQvndH8TfwIz-is20HRzPkHcrEyiWyAzN0KzQUUat8Vr6k8L9QXpaV-gMpM3IJCMhAiogQnynFas72SA'
			'/file# '
		),
	]


def prepare_datasets(path_to_save: str, size=448):
	size_dataset = {
		448: get_dataset_448x448(),
		512: get_dataset_512x512()
	}
	Path(path_to_save).mkdir(parents=True, exist_ok=True)

	for ds in size_dataset[size]:
		path_zip = path.join(path_to_save, ds.get_name() + '.zip')
		if not path.exists(path_zip):
			wget.download(ds.url, path_zip)

		if not path.exists(path_zip.rsplit('.zip')[0]):
			with ZipFile(path_zip, 'r') as ds_zipped:
				ds_zipped.extractall(path_to_save)


def build_data(
		path_ds: str, classes: List[str], env: Env, batch: int,
		fn_preprocessing=None
) -> Dataloader:
	prefix: Dict[Env, str] = {Env.eval: 'val', Env.test: 'test', Env.train: 'train'}
	path_imgs = path.join(path_ds, prefix[env])
	path_masks = path_imgs + '_gt'
	return build_dataloader(path_imgs, path_masks, classes, batch, fn_preproc=fn_preprocessing)


def build_dataset_name(params: NetworkParams) -> str:
	dataset_size = f'{params.size}x{params.size}'
	dataset_config = '_'.join([dataset_size, params.partition.value, params.format.value])
	return f'pneumonia_{dataset_config}'
