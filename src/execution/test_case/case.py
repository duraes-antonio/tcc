from datetime import datetime
from time import sleep
from typing import List, Optional, Dict
from requests.exceptions import ReadTimeout
from gspread import Worksheet
from pandas import DataFrame, Series

from enums import State, Network, DatasetPartition, DatasetFormat, Optimizer, Env, TestProgress, Metrics


class TestCase:
	date_fmt = '%d/%m/%Y %H:%M:%S'

	def __init__(self, s: Series):
		self.columns = [k for k in dict(s)]
		self.id = int(s['id'])
		self.net: Network = Network(s['net'])
		self.batch = int(s['batch'])
		self.partition: DatasetPartition = DatasetPartition(str(s['partition']))

		self.opt: Optimizer = Optimizer(s['opt'])
		self.dropout = float(s['dropout'])
		self.format: DatasetFormat = DatasetFormat(s['format'])

		env = Env.train
		self.train_start: datetime = self.parse_data(s[f'{env.name}_{TestProgress.start.name}'], self.date_fmt)
		self.train_end: datetime = self.parse_data(s[f'{env.name}_{TestProgress.start.name}'], self.date_fmt)
		self.train_state = State(s[f'{env.name}_state'])

		env = Env.test
		self.test_start: datetime = self.parse_data(s[f'{env.name}_{TestProgress.start.name}'], self.date_fmt)
		self.test_end: datetime = self.parse_data(s[f'{env.name}_{TestProgress.start.name}'], self.date_fmt)
		self.test_state = State(s[f'{env.name}_state'])

	@staticmethod
	def parse_data(data: str, fmt: str) -> datetime:
		return datetime.strptime(data, fmt) if data else None

	def update_date(self, ws: Worksheet, env: Env, prog: TestProgress, date: Optional[datetime] = None):
		ws.update_cell(
			self.id + 1, self.columns.index(f'{env.name}_{prog.name}') + 1,
			date.strftime(self.date_fmt) if date else ''
		)

	def update_state(self, ws: Worksheet, s: State, env: Env):
		ws.update_cell(self.id + 1, self.columns.index(f'{env.name}_state') + 1, s.name)

	def done(
			self, ws: Worksheet, env: Env,
			results: Optional[Dict[str, int]] = None,
			prefix_metrics: Optional[str] = None
	):
		self.update_state(ws, State.done, env)

		if env != Env.eval:
			self.update_date(ws, env, TestProgress.end, datetime.now())

		metrics = [
			Metrics.loss, Metrics.accuracy, Metrics.f1_score,
			Metrics.miou, Metrics.precision, Metrics.recall
		]

		cells = []

		prefix_col = '' if env == Env.train else f'{env.value}_'
		for m in metrics:
			cell = ws.cell(self.id + 1, self.columns.index(prefix_col + m.value) + 1)
			value = results[(prefix_metrics or '') + m.value]
			cell.value = "{:.6f}".format(value).replace('.', ',')
			cells.append(cell)

		if env != env.test:
			for m in metrics:
				name_key_best_val = f'best_{prefix_col}{m.value}'
				cell = ws.cell(self.id + 1, self.columns.index(name_key_best_val) + 1)
				value = results['best_' + prefix_metrics + m.value]
				cell.value = "{:.6f}".format(value).replace('.', ',')
				cells.append(cell)
		ws.update_cells(cells)

	def free(self, ws: Worksheet, env: Env, prog: Optional[TestProgress] = None):
		self.update_state(ws, State.free, env)
		self.update_date(ws, env, prog or TestProgress.end, None)

	def busy(self, ws: Worksheet, env: Env):
		self.update_state(ws, State.busy, env)
		self.update_date(ws, env, TestProgress.start, datetime.now())


class TestCaseManager:

	def __init__(self, ws: Worksheet):
		self.worksheet = ws

	@staticmethod
	def parse_dataframe(data: DataFrame) -> List[TestCase]:
		return [TestCase(row) for index, row in data.iterrows()]

	@staticmethod
	def parse_worksheet(data: Worksheet) -> List[TestCase]:
		retries_left = 3
		records = None

		def get_content_sheet():
			return data.get_all_records()

		while not records and retries_left > 0:
			try:
				records = get_content_sheet()
			except ReadTimeout:
				sleep(20)
				retries_left -= 1
				if retries_left < 0:
					raise
			finally:
				if records:
					return TestCaseManager.parse_dataframe(DataFrame.from_dict(records))

	@staticmethod
	def first_case(cases: List[TestCase], env=Env.train, s=State.free) -> TestCase:
		def case_free(c: TestCase) -> bool:
			return (c.train_state if env == Env.train else c.test_state) == s

		cases_free = filter(case_free, sorted(cases, key=lambda c: c.train_state.name.lower()))
		cases_free = list(cases_free)
		return cases_free[0] if len(cases_free) > 0 else None

	def first_case_free(self, env=Env.train) -> TestCase:
		cases = self.parse_worksheet(self.worksheet)
		return self.first_case(cases, env, State.free)
