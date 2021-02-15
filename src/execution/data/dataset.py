from zipfile import ZipFile

from os import path
import wget

from enums import DatasetPartition, DatasetFormat


def prepare_datasets(path_to_save: str, size=448):
	prefix = f'pneumonia_{size}x{size}'
	dataset_url = {
		f'{prefix}_{DatasetPartition.train_70_eval_20_test_10.value}_{DatasetFormat.equal_hist.value}.zip':
			'https://uc069a24fe6d276313bc6cc43664.dl.dropboxusercontent.com/cd/0/get/BI'
			'-0_xutkKmibSbrYkr9JDhQIkv5Dv8lEaXufhHIfkDmher5KYXpLD5px53emTjJXkUtWsa48KaIHBlrzc-Deg0daNPXBbgoAqqS5HNX'
			'-khStpJHV4WDyp9qdRBFyxGWw8M/file#',
		f'{prefix}_{DatasetPartition.train_70_eval_20_test_10.value}_{DatasetFormat.morp_transf.value}.zip':
			'https://uce87d062178e8b2cd8f36f44681.dl.dropboxusercontent.com/cd/0/get'
			'/BI9cvsigYCwfrZURqSKWEz4yw81Dpmin4Zj9ZbU1w7Wej1UG7DMQWTU'
			'-uZdW0IiC9JaTK3lAJcGuKZWidMwRopB6KcZcdc5q5BelY2zW7vQhd9wDtizEmJCBe4JstlduAp8/file#',
		f'{prefix}_{DatasetPartition.train_80_eval_10_test_10.value}_{DatasetFormat.morp_transf.value}.zip':
			'https://uc7b92abf9423211a0b111f210c6.dl.dropboxusercontent.com/cd/0/get/BI'
			'-79VbVD1P2DqRPTzxhamVSh2zlQZxwv5R5QIqKqbPDgOUW4_KXkIqLfCe9Y6uEZNP2sXmZ'
			'-4gDcI4sRExf8UrH1oCSotPxOzlbIFQLWn0qj5ToZDS_pFC7scI7RDvFTHU/file#',
		f'{prefix}_{DatasetPartition.train_80_eval_10_test_10.value}_{DatasetFormat.equal_hist.value}.zip':
			'https://uc1ab11456ee7303f60c74b49039.dl.dropboxusercontent.com/cd/0/get'
			'/BI9KKd3H6vAT4aoyd2FMJ4VpVwfiJ9Jia3vApuOif9RS_PJ6uYdPOt7C4Q8vUs7wypnWKZl3'
			'_Dkc6WtAGEXPj2Hy7ZUpbALymSauTiFcjMgBIlT0ovs7tGoCbw9YIV6sSt4/file# '
	}

	for ds in dataset_url:
		path_zip = path.join(path_to_save, ds)
		if not path.exists(path_zip):
			wget.download(dataset_url[ds], path_zip)

		if not path.exists(path_zip.rsplit('.zip')[0]):
			with ZipFile(path_zip, 'r') as ds_zipped:
				ds_zipped.extractall(path_to_save)
