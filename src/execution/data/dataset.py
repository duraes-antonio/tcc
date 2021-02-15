from os import path
from typing import Dict, List
from zipfile import ZipFile

import wget

from enums import DatasetPartition, DatasetFormat, Env
from network.dataset_dataloader import Dataloader, build_dataloader
from network.params import NetworkParams


def prepare_datasets(path_to_save: str, size=448):
	prefix = f'pneumonia_{size}x{size}'
	dataset_url = {
		f'{prefix}_{DatasetPartition.train_70_eval_20_test_10.value}_{DatasetFormat.equal_hist.value}.zip':
			'https://ucd3426c24b8cce15ceb52c94198.dl.dropboxusercontent.com/cd/0/get/BI_NWA2h_kjVAb'
			'-idrCqt4koRxdU0RHhY5oWO0zm6g3S4ZWf3msyCA4NPDLCH9Bj2nC-VrEqAX88xLfQY3o'
			'-_N0tm8ejVt6FssFfCeFipT_qJTP9jtrbL6xp2SeQXehK5hM/file#',
		f'{prefix}_{DatasetPartition.train_70_eval_20_test_10.value}_{DatasetFormat.morp_transf.value}.zip':
			'https://uc04f881e88340c7626bdcdad17d.dl.dropboxusercontent.com/cd/0/get'
			'/BI9m7MDX0T3Xxc6uSp_9f4lsPQ3TMGVybzUZ'
			'-iCp7zsDflryVEDuj2Z012Pa0ozNgC8fJsli5VqV_pC7PQ6CHmWgJRgy8FvfwdBoHtRYE2dboSFwcx-s6CYuB8GpQTT0JFs/file#',
		f'{prefix}_{DatasetPartition.train_80_eval_10_test_10.value}_{DatasetFormat.equal_hist.value}.zip':
			'https://uc57012e2112cb117d519f1a3cfb.dl.dropboxusercontent.com/cd/0/get/BI8oumkJy5DXWrmZHpVkF-oFP81KF6wg'
			'-0CVPUGzsr3f8B7yXphMc0qdbxeyg5dSaP2RT-q7EIk2uQAOGURkDQBuGdHpvctLvpCyMAd4VhgwBTXbp6DDyiCSfCnD3puAAic/file'
			'#',
		f'{prefix}_{DatasetPartition.train_80_eval_10_test_10.value}_{DatasetFormat.morp_transf.value}.zip':
			'https://uc20c0e54e4d945c4dbc09aec0cf.dl.dropboxusercontent.com/cd/0/get/BI-k_Fs1D'
			'-yf2jJ8kAvAdbRCXMvFn3fW4z9VPmlcEhlocQggoKYlHZgLV4vaNLZo6dAhfciOyAlPMiHL3tP83PI0HY'
			'c6EEVF2OVhHFdVnUNrll23MjoCxqwtqG9CMv_wBH4/file# '
	}

	for ds in dataset_url:
		path_zip = path.join(path_to_save, ds)
		if not path.exists(path_zip):
			wget.download(dataset_url[ds], path_zip)

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
