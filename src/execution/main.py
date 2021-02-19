import gc
import pathlib
from pandas import DataFrame
from os import path
from typing import Dict, Union

import tensorflow as tf
from gspread import Worksheet

from cli import read_args
from data.dataset import prepare_datasets, build_dataset_name, build_data
from enums import Env, Network, TestProgress
from helper.git import Git
from helper.helpers import get_name, write_csv_metrics, write_csv_metrics_test
from network.common import get_metrics, get_callbacks, build_network, get_optimizer
from network.params import UNetParams, DeeplabParams, NetworkParams
from network.unet import get_preprocessing
from test_case.case import TestCaseManager, TestCase
from test_case.worksheet import load_worksheet


def tf_gpu_allow_growth():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)


# Define que a alocação de memória da GPU ocorrerá sob demanda e não de uma vez
tf_gpu_allow_growth()

import keras
from keras.models import Model


def build_trained_model_name(params: NetworkParams) -> str:
	fragments = [
		f'{params.size}x{params.size}',
		f'{params.partition.value}',
		f'{params.format.value}',
		get_name(params.backbone.value),
		get_name(params.opt),
		f'batch-{params.batch}',
		f'epochs-{params.epochs}',
		f'lr-{params.lr}',
		f'drop-{1 if params.dropout > 0 else 0}',
		f'clip-{params.clip_value}'
	]
	return '_'.join(fragments)


def mark_done_and_commit_results(
		case: TestCase, ws: Worksheet, path_file: str, env: Env,
		gh: Git, params: NetworkParams, res: Dict[str, Union[int, list]]
):
	commit_msg = gh.build_commit_msg(params, env)

	if env == Env.test:
		file = write_csv_metrics_test(res)

	else:
		file = write_csv_metrics(res)

	gh.create_file(path_file, file, commit_msg)

	if env == Env.test:
		case.done(ws, env, res)
	else:
		last_epoch_results = {k: res[k][-1] for k in res}
		case.done(ws, env.train, last_epoch_results)
		case.done(ws, env.eval, last_epoch_results, 'val_')


def print_params(params: NetworkParams, opt: keras.optimizers.Optimizer):
	divider = '- - - - -'
	print('\n\n\n', divider, 'CONFIG', divider)
	print(DataFrame.from_dict(params.__dict__))
	print('\n\n', divider, 'OPTIMIZER CONFIG', divider)
	print(DataFrame.from_dict({**opt.get_config(), 'lr': opt.lr}))
	print(' '.join([divider, divider, divider]))
	print('\n\n\n')


def main():
	classes = ['background', 'covid', 'bacterial', 'viral']
	repository_name = 'tcc'
	path_where = pathlib.Path().absolute()
	path_root = str(path_where).split(repository_name)[0]
	path_datasets = path.join(path_root, 'datasets')

	args = read_args()
	path_gsheets_cred = args.credentials_path
	ws = load_worksheet('tcc', path_gsheets_cred, 'cases')
	test_manager = TestCaseManager(ws)
	case = test_manager.first_case_free()

	# Baixar e extrair datasets
	prepare_datasets(path_datasets, args.size)
	path_current = path.join(path_root, repository_name)
	path_results = path.join('results', case.net.value, case.partition.value)
	gh = Git('duraes-antonio', repository_name, args.gh_token)

	while case is not None:
		current_env = Env.train

		try:
			# Marcar caso como ocupado
			case.busy(ws, current_env)

			# Definir params
			if case.net == Network.unet:
				params = UNetParams(case, classes, args.size)
			else:
				params = DeeplabParams(case, classes, size=args.size)

			# Instanciar modelo
			model: Model = build_network(case.net, params)

			# Compilar modelo
			metrics = get_metrics(len(classes))
			optim = get_optimizer(params.opt, params.lr, params.clip_value)
			model.compile(optimizer=optim, loss=params.loss, metrics=metrics)

			# Gerar Dataloaders
			preprocess_fn = get_preprocessing(params.backbone.value) if case.net == Network.unet else None
			path_dataset = path.join(path_datasets, build_dataset_name(params))
			train_dataloader = build_data(path_dataset, classes, Env.train, params.batch, preprocess_fn)
			val_dataloader = build_data(path_dataset, classes, Env.eval, 1, preprocess_fn)
			test_dataloader = build_data(path_dataset, classes, Env.test, 1, preprocess_fn)

			trained_model_name = build_trained_model_name(params)
			path_trained_model = path.join(path_current, 'trained')
			callbacks = get_callbacks(path.join(path_trained_model, trained_model_name))
			print_params(params, optim)

			# Treinar modelo
			history = model.fit_generator(
				generator=train_dataloader, validation_data=val_dataloader,
				epochs=params.epochs, callbacks=callbacks, workers=4
			)
			mark_done_and_commit_results(
				case, ws, path.join(path_results, f'{trained_model_name}.csv'),
				current_env, gh, params, history.history
			)

			# Avaliar modelo
			current_env = Env.test
			case.busy(ws, current_env)
			model.load_weights(path.join(path_trained_model, f'{trained_model_name}.h5'))
			scores = model.evaluate_generator(test_dataloader)

			scores_dict = {get_name(m): v for m, v in zip(['loss', *metrics], scores)}
			mark_done_and_commit_results(
				case, ws, path.join(path_results, f'{trained_model_name}_test.csv'),
				current_env, gh, params, scores_dict
			)

			del model
			del train_dataloader
			del val_dataloader
			del test_dataloader
			gc.collect()
			case = test_manager.first_case_free()

		except:
			case.free(ws, current_env, TestProgress.start)
			raise

	return 0


main()
