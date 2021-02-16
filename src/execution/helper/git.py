from github import Github
from github.GithubException import UnknownObjectException

from enums import Env
from network.params import NetworkParams
from .helpers import get_name


class Git:
	def __init__(self, username: str, repository_name: str, token: str):
		self.user = username
		self.gh = Github(token)
		self.repository = self.gh.get_repo(f'{username}/{repository_name}')

	def create_file(self, file_repo_path: str, content: str, commit_msg: str):
		try:
			contents = self.repository.get_contents(file_repo_path)
		except UnknownObjectException:
			contents = None

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
			get_name(params.opt),
			f'batch {params.batch}',
			f'epochs {params.epochs}',
			f'lr {params.lr}',
			f'dropout {1 if params.dropout > 0 else 0}',
		]
		return f"OUT: [{env.name.upper()}] {', '.join(fragments)}"
