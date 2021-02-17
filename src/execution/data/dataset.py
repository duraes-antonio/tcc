from os import path
from pathlib import Path
from typing import Dict, List, Union, Tuple
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
			'http://getmega.net/download/file_418c59da60/pneumonia_512x512_702010_hist.zip',
		),
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.morp_transf, size,
			'http://getmega.net/download/file_93a3b7894d/pneumonia_512x512_702010_morf.zip',
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.equal_hist, size,
			'http://getmega.net/download/file_c786e77ef3/pneumonia_512x512_801010_hist.zip'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.morp_transf, size,
			'http://getmega.net/download/file_449e66eba9/pneumonia_512x512_801010_morf.zip'
		),
	]


def get_dataset_448x448(prefix='pneumonia') -> List[DatasetDownload]:
	size = 448
	return [
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.equal_hist, size,
			'http://getmega.net/download/file_8b9770b800/pneumonia_448x448_702010_hist.zip'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.morp_transf, size,
			'http://getmega.net/download/file_063895e2a7/pneumonia_448x448_702010_morf.zip'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.equal_hist, size,
			'http://getmega.net/download/file_54c1d1e43d/pneumonia_448x448_801010_hist.zip'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.morp_transf, size,
			'http://getmega.net/download/file_4dbf01e541/pneumonia_448x448_801010_morf.zip'
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
