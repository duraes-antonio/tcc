import os

from enums import Env
from network.params import NetworkParams
from .helpers import get_name


class Git:

	def __init__(self, username: str, user_email: str, repository_name: str, token: str):
		self.user = username
		self.user_email = user_email
		self.repo_name = repository_name
		self.token = token

	def config_user(self):
		os.system(f"""git config --global user.email "{self.user_email}" """)
		os.system(f"""git config --global user.name "{self.user}" """)

	def add_commit_item(self, item_local_path: str, commit_msg: str):
		os.system(f"""git add {item_local_path}""")
		os.system(f"""git commit -m "{commit_msg}" """)

	def pull_changes(self):
		os.system("git pull")
		os.system("git pull origin master")

	def push(self):
		os.system(f"git push https://{self.user}:{self.token}@github.com/{self.user}/{self.repo_name}.git")

	def save_item(self, file_repo_path: str, commit_msg: str):
		self.config_user()
		self.add_commit_item(file_repo_path, commit_msg)
		self.pull_changes()
		self.push()

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
