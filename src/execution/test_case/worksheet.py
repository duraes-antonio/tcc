import gspread
from gspread import Worksheet


def load_worksheet(name='tcc', name_json_credent='credentials.json') -> Worksheet:
	scopes = [
		"https://spreadsheets.google.com/feeds",
		'https://www.googleapis.com/auth/spreadsheets',
		"https://www.googleapis.com/auth/drive.file",
		"https://www.googleapis.com/auth/drive"
	]
	gc = gspread.service_account(name_json_credent, scopes)
	return gc.open(name).sheet1
