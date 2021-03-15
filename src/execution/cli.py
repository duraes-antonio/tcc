import argparse


class ArgsCLI:
	def __init__(self, size: int, gh_token: str, credentials_path: str):
		self.size = size
		self.gh_token = gh_token
		self.credentials_path = credentials_path


class ArgsVisualize:
	def __init__(self, network: str, size: int, partition: str, path_trained: str):
		self.size = size
		self.partition = partition
		self.network = network
		self.path_trained = path_trained


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


def read_args_visualize() -> ArgsVisualize:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--network', '-n', type=str, default=None, metavar='network',
		required=True, help="Name of network",
	)
	parser.add_argument(
		'--size', '-s', choices=[448, 512], default=448, metavar='size',
		required=False, type=int,
		help="Width and height of the dataset images to be used [default: 448]",
	)
	parser.add_argument(
		'--partition', '-p', type=str, default=None, metavar='partition',
		required=True, help="Dataset partition",
	)
	parser.add_argument(
		'--trained', '-t', type=str, default=None, metavar='trained',
		required=True, help="Path for trained model (h5 format)",
	)
	_args = parser.parse_args()
	return ArgsVisualize(_args.size, _args.size, _args.partition, _args.trained)
