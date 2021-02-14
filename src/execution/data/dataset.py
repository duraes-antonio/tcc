from zipfile import ZipFile

from os import path
import wget


def prepare_datasets(path_to_save: str):
	dataset_url = {
		'pneumonia_512x512_702010_hist.zip': 'https://uceb6126a8ab4e10298013f6fd06.dl.dropboxusercontent.com/cd/0/get/BI-ktltMCsdGn14dLhE0HZH861Jsgzh5Ym5gTVS9M-4yxx-HGW33tdtpNCzxSwBOnX0-gfuCOCo5AiWTXQgi2ZYJNQ_KflRrdibCM5NSlkK7LE6T8rePKFRAR2bwSr4NblI/file#'
	}

	for ds in dataset_url:
		path_zip = path.join(path_to_save, ds)
		if not path.exists(path_zip):
			wget.download(dataset_url[ds], path_zip)

		if not path.exists(path_zip.rsplit('.zip')[0]):
			with ZipFile(path_zip, 'r') as ds_zipped:
				ds_zipped.extractall()
