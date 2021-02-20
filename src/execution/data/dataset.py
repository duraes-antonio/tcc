from os import path, remove
from pathlib import Path
from typing import List, Union, Tuple, Dict
from zipfile import ZipFile

from .dropbox_wrapper import DropboxWrapper
from enums import DatasetPartition, DatasetFormat, Env
from network.dataset_dataloader import build_dataloader
from network.params import NetworkParams


class DatasetDownload:
	def __init__(
			self, prefix_name: str, partition: DatasetPartition,
			fmt: DatasetFormat, size: Union[Tuple[int, int], int],
	):
		self.partition = partition
		self.format = fmt
		self.prefix = prefix_name
		self.size: Tuple[int, int] = [size, size] if isinstance(size, int) else size

	def get_name(self) -> str:
		size = 'x'.join([str(s) for s in self.size])
		return '_'.join([self.prefix, size, self.partition.value, self.format.value])


def get_dataset_512x512(prefix='pneumonia') -> List[DatasetDownload]:
	size = 512
	return [
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.equal_hist, size,
		),
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.morp_transf, size
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.equal_hist, size
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.morp_transf, size
		),
	]


def get_dataset_448x448(prefix='pneumonia') -> List[DatasetDownload]:
	size = 448
	return [
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.equal_hist, size
		),
		DatasetDownload(
			prefix, DatasetPartition.train_70_eval_20_test_10,
			DatasetFormat.morp_transf, size
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.equal_hist, size
		),
		DatasetDownload(
			prefix, DatasetPartition.train_80_eval_10_test_10,
			DatasetFormat.morp_transf, size
		),
	]


def prepare_datasets(path_to_save: str, size=448):
	size_dataset = {
		448: get_dataset_448x448(),
		512: get_dataset_512x512()
	}
	size_token = {
		448: 'Bibn2C5INDYAAAAAAAAAAZgMq5Kv8bpeRNL5NfW1BmYjZzmgZr4FF6nEtp41g2HO',
		512: 'AlVAPTglrDsAAAAAAAAAAdRnFpr2gkwzjzddx6bFUEFMG_5nqqdhB15AZTA2W073'
	}
	Path(path_to_save).mkdir(parents=True, exist_ok=True)

	token = size_token[size]
	dbx = DropboxWrapper(token)
	for ds in size_dataset[size]:
		filename = ds.get_name() + '.zip'
		path_zip = path.join(path_to_save, filename)
		if not path.exists(path_zip):
			dbx.download_dataset(filename, path_zip)

		if not path.exists(path_zip.rsplit('.zip')[0]):
			with ZipFile(path_zip, 'r') as ds_zipped:
				ds_zipped.extractall(path_to_save)
				remove(path_zip)


def build_data(
		path_ds: str, classes: List[str], env: Env, batch: int,
		fn_preprocessing=None
):
	prefix: Dict[Env, str] = {Env.eval: 'val', Env.test: 'test', Env.train: 'train'}
	path_imgs = path.join(path_ds, prefix[env])
	path_masks = path_imgs + '_gt'
	return build_dataloader(path_imgs, path_masks, classes, batch, fn_preproc=fn_preprocessing)


def build_dataset_name(params: NetworkParams) -> str:
	dataset_size = f'{params.size}x{params.size}'
	dataset_config = '_'.join([dataset_size, params.partition.value, params.format.value])
	return f'pneumonia_{dataset_config}'

