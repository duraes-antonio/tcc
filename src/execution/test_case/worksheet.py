from typing import Optional

import gspread
from gspread import Worksheet

from case import TestCase
from enums import DatasetPartition, DatasetFormat, Optimizer, Metrics


def load_worksheet(
		ss_name='tcc', path_json_credent='credentials.json',
		ws_name: Optional[str] = None
) -> Worksheet:
	scopes = [
		"https://spreadsheets.google.com/feeds",
		'https://www.googleapis.com/auth/spreadsheets',
		"https://www.googleapis.com/auth/drive.file",
		"https://www.googleapis.com/auth/drive"
	]
	gc = gspread.service_account(path_json_credent, scopes)
	spreadsheet = gc.open(ss_name)
	return spreadsheet.worksheet(ws_name) if ws_name else spreadsheet.sheet1


def update_results(
		ws: Worksheet, case: TestCase, results: dict, row_start=6,
		prefix_metrics=''
):
	index_last_col = 17
	index_cell = index_last_col

	if case.partition == DatasetPartition.train_70_eval_20_test_10:
		index_cell -= 8

	if case.format == DatasetFormat.equal_hist:
		index_cell -= 4

	if case.opt == Optimizer.adam:
		index_cell -= 2

	if case.dropout > 0:
		index_cell -= 1

	metrics = [
		Metrics.loss, Metrics.accuracy, Metrics.f1_score, Metrics.miou,
		Metrics.precision, Metrics.recall
	]

	cells = ws.range(f'B{row_start}:B{row_start + 5}')
	for i in range(len(cells)):
		cells[i].value = results[prefix_metrics + metrics[i].value]
	ws.update_cells(cells)
