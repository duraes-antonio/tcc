from os import path
from typing import Dict, Union, Callable, List

from keras.models import Model

from data.dataset import prepare_datasets
from enums import Env, Network
from helper.git import Git
from helper.helpers import get_name, write_csv_metrics, write_csv_metrics_test, timer
from network.callbacks_metrics import get_metrics, get_callbacks
from network.dataset_dataloader import Dataloader, build_dataloader
from network.deeplab import deeplabv3
from network.params import UNetParams, DeeplabParams, NetworkParams
from test_case.case import TestCaseManager
from test_case.worksheet import load_worksheet


def build_deeplab(params: DeeplabParams) -> Callable[[], Model]:
	def child():
		return deeplabv3(
			input_shape=(params.size, params.size, 3),
			classes=len(params.classes), backbone=params.backbone.name,
			OS=params.os, dropout=params.dropout
		)

	return child


# TODO: Continuar/Incluir UNET
def build_network(
		net: Network, config: Union[DeeplabParams, UNetParams]
) -> Model:
	handlers: Dict[Network, Callable[[], Model]] = {
		Network.unet: None,
		Network.deeplab: build_deeplab(config)
	}
	return handlers[net]()


def build_data(path_ds: str, classes: List[str], env: Env, batch: int) -> Dataloader:
	prefix: Dict[Env, str] = {Env.eval: 'val', Env.test: 'test', Env.train: 'train'}
	path_imgs = path.join(path_ds, prefix[env])
	path_masks = path_imgs + '_gt'
	return build_dataloader(path_imgs, path_masks, classes, batch)


def build_dataset_name(params: NetworkParams) -> str:
	dataset_size = f'{params.size}x{params.size}'
	dataset_config = '_'.join([dataset_size, params.partition.name, params.format.name])
	return f'pneumonia_{dataset_config}'


def build_trained_model_name(params: NetworkParams) -> str:
	fragments = [
		f'{params.size}x{params.size}',
		f'{params.partition.name}',
		f'{params.format.name}',
		get_name(params.backbone),
		get_name(params.opt),
		f'batch-{params.batch}',
		f'epochs-{params.epochs}',
		f'lr-{params.lr}',
		f'drop-{1 if params.dropout > 0 else 0}',
		f'clip-{params.clip_value}'
	]
	return '_'.join(fragments)


def main():
	classes = ['background', 'covid', 'bacterial', 'viral']

	# Obter dados sobre o caso de teste disponível (ainda não executado)
	path_root = '/home/acduraes/content'
	path_where = path.join(path_root, 'tcc', 'src', 'execution')
	ws = load_worksheet(path_json_credent=path.join(path_where, 'test_case', 'credentials.json'))
	test_manager = TestCaseManager(ws)
	case = test_manager.first_case_free()

	# Baixar e extrair datasets
	prepare_datasets(path_root)

	path_current = path.join(path_root, 'tcc')
	path_results = path.join('results', case.net.name, case.partition.name)
	gh = Git('duraes-antonio', 'tcc')

	try:
		# Marcar caso como ocupado
		case.busy(ws, Env.train)

		# Definir params
		params = UNetParams(case, classes) if case.net == Network.unet else DeeplabParams(case, classes)

		# Instanciar modelo
		model: Model = build_network(case.net, params)

		# Compilar modelo
		metrics = get_metrics(len(classes))
		model.compile(case.opt.name, params.loss, metrics=metrics)

		# Gerar Dataloaders
		ds_name = build_dataset_name(params)
		train_dataloader = build_data(ds_name, classes, Env.train, params.batch)
		val_dataloader = build_data(ds_name, classes, Env.eval, params.batch)
		test_dataloader = build_data(ds_name, classes, Env.test, 1)

		trained_model_name = build_trained_model_name(params)
		path_trained_model = path.join(path_current, 'trained')
		callbacks = get_callbacks(path.join(path_trained_model, trained_model_name))

		# Treinar modelo
		@timer
		def train_model():
			history = model.fit_generator(
				generator=train_dataloader, validation_data=val_dataloader,
				epochs=params.epochs, callbacks=callbacks, workers=8
			)

			# Commitar resultados
			log = write_csv_metrics(history.history)
			commit_msg = gh.build_commit_msg(params, Env.train)
			gh.create_file(path.join(path_results, f'{trained_model_name}.csv'), log, commit_msg)
			case.done(ws, Env.train)

		# Avalair modelo
		@timer
		def eval_model():
			model.load_weights(path.join(path_trained_model, f'{trained_model_name}.h5'))
			scores = model.evaluate_generator(test_dataloader)

			# Commitar resultados
			scores_dict = {get_name(m): v for m, v in zip(['loss', *metrics], scores)}
			log_test = write_csv_metrics_test(scores_dict)
			commit_msg = gh.build_commit_msg(params, Env.test)
			gh.create_file(path.join(path_results, f'{trained_model_name}_test.csv'), log_test, commit_msg)
			case.done(ws, Env.eval)

		eval_model()

	except:
		case.free(ws, Env.train)

	return 0


main()
