from datetime import datetime
from typing import List, Union

from gspread import Worksheet
from pandas import DataFrame, Series

from enums import State, Network, DatasetPartition, DatasetFormat, Optimizer, Env


class TestCase:
	__date_format__ = '%d/%m/%Y %H:%M:%S'
	__cols__ = [
		'id', 'net', 'batch', 'partition', 'opt', 'dropout', 'format',
		'train_start', 'train_end', 'train_state', 'eval_start', 'eval_end', 'eval_state'
	]

	def __init__(self, s: Series):
		self.id = int(s['id'])
		self.net: Network = Network(s['net'])
		self.batch = int(s['batch'])
		self.partition: DatasetPartition = DatasetPartition(str(s['partition']))

		self.opt: Optimizer = Optimizer(s['opt'])
		self.dropout = float(s['dropout'])
		self.format: DatasetFormat = DatasetFormat(s['format'])

		self.train_start: datetime = self.parse_data(s['train_start'], self.__date_format__)
		self.train_end: datetime = self.parse_data(s['train_end'], self.__date_format__)
		self.train_state = State(s['train_state'])

		self.train_start: datetime = self.parse_data(s['eval_start'], self.__date_format__)
		self.train_end: datetime = self.parse_data(s['eval_end'], self.__date_format__)
		self.eval_state = State(s['eval_state'])

	@staticmethod
	def parse_data(data: str, fmt: str) -> datetime:
		return datetime.strptime(data, fmt) if data else None

	def update_state(self, ws: Worksheet, s: State, env: Env):
		ws.update_cell(self.id + 1, self.__cols__.index(f'{env.name}_state') + 1, s.name)

	def done(self, ws: Worksheet, env: Env):
		ws.update_cell(self.id + 1, self.__cols__.index(f'{env.name}_end') + 1, str(datetime.now()))
		self.update_state(ws, State.done, env)

	def free(self, ws: Worksheet, env: Env):
		self.update_state(ws, State.free, env)

	def busy(self, ws: Worksheet, env: Env):
		ws.update_cell(self.id + 1, self.__cols__.index(f'{env.name}_start') + 1, str(datetime.now()))
		self.update_state(ws, State.busy, env)


class TestCaseManager:
	cases: List[TestCase]

	def __init__(self, cases: Union[List[TestCase], DataFrame, Worksheet]):
		if isinstance(cases, List):
			self.cases = cases
		elif isinstance(cases, DataFrame):
			self.cases = self.parse_dataframe(cases)
		elif isinstance(cases, Worksheet):
			self.cases = self.parse_worksheet(cases)

	@staticmethod
	def parse_dataframe(data: DataFrame) -> List[TestCase]:
		return [TestCase(row) for index, row in data.iterrows()]

	@staticmethod
	def parse_worksheet(data: Worksheet) -> List[TestCase]:
		return TestCaseManager.parse_dataframe(DataFrame.from_dict(data.get_all_records()))

	@staticmethod
	def first_case(cases: List[TestCase], env=Env.train, s=State.free) -> TestCase:
		def case_free(c: TestCase) -> bool:
			return (c.train_state if env == Env.train else c.eval_state) == s

		cases_free = filter(case_free, sorted(cases, key=lambda c: c.train_state.name.lower()))
		cases_free = list(cases_free)
		return cases_free[0] if len(cases_free) > 0 else None

	def first_case_free(self, env=Env.train) -> TestCase:
		return self.first_case(self.cases, env, State.free)
