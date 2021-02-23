import argparse


class ArgsCLI:
	def __init__(self, size: int, gh_token: str, credentials_path: str):
		self.size = size
		self.gh_token = gh_token
		self.credentials_path = credentials_path


def read_args() -> ArgsCLI:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--credentials', '-c', type=str, default=None, metavar='credentials',
		required=True, help="Path to the Google Drive service credentials JSON file",
	)
	parser.add_argument(
		'--ghtoken', '-t', metavar='ghtoken', required=True, type=str,
		help="Github token for committing and reading the repository",
	)
	parser.add_argument(
		'--size', '-s', choices=[448, 512], default=448, metavar='size',
		required=False, type=int,
		help="Width and height of the dataset images to be used [default: 448]",
	)

	_args = parser.parse_args()
	return ArgsCLI(_args.size, _args.ghtoken, _args.credentials)
