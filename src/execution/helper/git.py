from time import sleep

from github import Github
from github.GithubException import UnknownObjectException
from requests.exceptions import ReadTimeout, ConnectTimeout, Timeout
from enums import Env
from network.params import NetworkParams
from .helpers import get_name


class Git:
	def __init__(self, username: str, repository_name: str, token: str):
		self.user = username
		self.repo_name = repository_name
		self.token = token
		self.start_session()

	def start_session(self):
		self.gh = Github(self.token)
		self.repository = self.gh.get_repo(f'{self.user}/{self.repo_name}')

	def create_file(self, file_repo_path: str, content: str, commit_msg: str):

		def get_content():
			return self.repository.get_contents(file_repo_path)

		def get_content_manager():
			retries_left = 3
			contents = None

			while not contents and retries_left > 0:
				try:
					self.start_session()
					contents = get_content()
				except (ReadTimeout, ConnectTimeout, Timeout):
					sleep(20)
					retries_left -= 1
					if retries_left < 0:
						raise
				except UnknownObjectException:
					contents = None
				finally:
					if contents:
						return contents

		contents = get_content_manager()

		if contents and contents.content:
			self.repository.update_file(file_repo_path, commit_msg, content, contents.sha)
		else:
			self.repository.create_file(file_repo_path, commit_msg, content)

	def build_commit_msg(self, params: NetworkParams, env: Env) -> str:
		fragments = [
			f'{params.size}x{params.size}',
			f'{params.partition.name}',
			f'{params.format.name}',
			params.backbone.value,
			get_name(params.opt.value),
			f'batch {params.batch}',
			f'epochs {params.epochs}',
			f'lr {params.lr}',
			f'dropout {1 if params.dropout > 0 else 0}',
		]
		return f"OUT: [{env.name.upper()}] {', '.join(fragments)}"
