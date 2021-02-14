from zipfile import ZipFile

from os import path
import requests


def prepare_datasets(path_to_save: str):
	dataset_url = {
		'pneumonia_512x512_702010_hist.zip': 'https://www.dropbox.com/sh/rmqrdj0o7ca9fin/AAB8jGCH5cHR5TUn1ZtQ7_Hfa?dl=0'
	}

	for ds in dataset_url:
		path_zip = path.join(path_to_save, ds)
		if path.exists(path_zip) or path.exists(path_zip.rsplit('.zip')[0]):
			continue

		response = requests.get(dataset_url[ds])
		# path_zip = path.join(path_to_save, u.rsplit(sep, 1)[-1].rsplit('_zip')[0] + '.zip')

		with open(path_zip, 'wb') as f:
			f.write(response.content)

		with ZipFile(path_zip, 'r') as ds_zipped:
			ds_zipped.extractall()
