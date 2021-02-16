from os import path
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
			'https://ucd3426c24b8cce15ceb52c94198.dl.dropboxusercontent.com/cd/0/get/BI_NWA2h_kjVAb'
			'-idrCqt4koRxdU0RHhY5oWO0zm6g3S4ZWf3msyCA4NPDLCH9Bj2nC-VrEqAX88xLfQY3o'
			'-_N0tm8ejVt6FssFfCeFipT_qJTP9jtrbL6xp2SeQXehK5hM/file#'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.morp_transf, size,
			'https://uc04f881e88340c7626bdcdad17d.dl.dropboxusercontent.com/cd/0/get'
			'/BI9m7MDX0T3Xxc6uSp_9f4lsPQ3TMGVybzUZ'
			'-iCp7zsDflryVEDuj2Z012Pa0ozNgC8fJsli5VqV_pC7PQ6CHmWgJRgy8FvfwdBoHtRYE2dboSFwcx-s6CYuB8GpQTT0JFs/file#',
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.equal_hist, size,
			'https://uc57012e2112cb117d519f1a3cfb.dl.dropboxusercontent.com/cd/0/get/BI8oumkJy5DXWrmZHpVkF-oFP81KF6wg'
			'-0CVPUGzsr3f8B7yXphMc0qdbxeyg5dSaP2RT-q7EIk2uQAOGURkDQBuGdHpvctLvpCyMAd4VhgwBTXbp6DDyiCSfCnD3puAAic/file'
			'#'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.morp_transf, size,
			'https://uc20c0e54e4d945c4dbc09aec0cf.dl.dropboxusercontent.com/cd/0/get/BI-k_Fs1D'
			'-yf2jJ8kAvAdbRCXMvFn3fW4z9VPmlcEhlocQggoKYlHZgLV4vaNLZo6dAhfciOyAlPMiHL3tP83PI0HY'
			'c6EEVF2OVhHFdVnUNrll23MjoCxqwtqG9CMv_wBH4/file#'
		),
	]


def get_dataset_448x448(prefix='pneumonia') -> List[DatasetDownload]:
	size = 448
	return [
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.equal_hist, size,
			'https://drive.google.com/u/0/uc?export=download&confirm=LezY&id=1_hPTB6p6bqkUNzN4QTXBW4U9W7txFDyL'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.morp_transf, size,
			'https://drive.google.com/u/0/uc?export=download&confirm=9JNO&id=16ydo48sN7RsN_mSjYPfLqANSZW079JWi'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.equal_hist, size,
			'https://drive.google.com/u/0/uc?export=download&confirm=-RoB&id=1gLQrItHappIg2oJTSe1B1DGxt0RtVZsR'
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.morp_transf, size,
			'https://drive.google.com/u/0/uc?export=download&confirm=5otE&id=1LirPpJf4z9WEhuV3zbHXmrnWkWgOa1HO'
		),
	]


def prepare_datasets(path_to_save: str, size=448):
	size_dataset = {
		448: get_dataset_448x448(),
		512: get_dataset_512x512()
	}

	for ds in size_dataset[size]:
		path_zip = path.join(path_to_save, ds.get_name() + '.zip')
		if not path.exists(path_zip):
			wget.download(ds.url, path_zip)

		if not path.exists(path_zip.rsplit('.zip')[0]):
			with ZipFile(path_zip, 'r') as ds_zipped:
				ds_zipped.extractall(path_to_save)


def build_data(path_ds: str, classes: List[str], env: Env, batch: int) -> Dataloader:
	prefix: Dict[Env, str] = {Env.eval: 'val', Env.test: 'test', Env.train: 'train'}
	path_imgs = path.join(path_ds, prefix[env])
	path_masks = path_imgs + '_gt'
	return build_dataloader(path_imgs, path_masks, classes, batch)


def build_dataset_name(params: NetworkParams) -> str:
	dataset_size = f'{params.size}x{params.size}'
	dataset_config = '_'.join([dataset_size, params.partition.value, params.format.value])
	return f'pneumonia_{dataset_config}'
