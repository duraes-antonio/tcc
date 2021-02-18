import dropbox


class DropboxWrapper:

	def __init__(self, token):
		self.dbx = dropbox.Dropbox(token)

	def download_dataset(self, name: str, path_save: str):
		with open(path_save, 'wb') as f:
			metadata, res = self.dbx.files_download(path=f"/datasets/{name}")
			f.write(res.content)
