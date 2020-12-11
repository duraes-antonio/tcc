import argparse
import datetime
from typing import Iterable

import arquivo_util as au


def boolean(valor: str) -> bool:
	if isinstance(valor, bool):
		return valor

	if valor.lower() in ('yes', 'true', 't', 'y', '1'):
		return True

	elif valor.lower() in ('no', 'false', 'f', 'n', '0'):
		return False

	else:
		raise argparse.ArgumentTypeError('Um valor lógico é esperado.')


class CLIArgs:

	def __init__(
			self, dir_imagens: str, dir_anotacoes: str,
			converter_cor_indice=True, particionar=False,
			porcent_treino=.7, porcent_val=.2, porcent_teste=.1,
			tamanho: Iterable[int] = (512, 512), equalizar_histograma=False,
			transformacao_morfologica=False,
			imagem_formato='jpeg', anotacao_formato='png'
	):
		self.transformacao_morfologica = transformacao_morfologica
		self.equalizar_histograma = equalizar_histograma
		self.dir_imagens = dir_imagens
		self.dir_anotacoes = dir_anotacoes
		self.particionar = particionar
		self.porcent_treino = porcent_treino
		self.porcent_val = porcent_val
		self.porcent_teste = porcent_teste
		self.cor_indice = converter_cor_indice
		self.tamanho = tamanho
		self.imagem_formato = imagem_formato
		self.anotacao_formato = anotacao_formato


def ler_args() -> CLIArgs:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--download', '-d', type=boolean, default=False, metavar='download', required=False,
		help="Especifica se as imagens serão baixadas [default: False]"
	)
	parser.add_argument(
		'--formato_imagem', '-fi', type=str, default='jpeg', metavar='formato_imagem', required=False,
		help="Formato da imagem (não anotação) [default: 'jpeg']",
	)
	parser.add_argument(
		'--formato_anotacao', '-fa', type=str, default='png', metavar='formato_anotacao', required=False,
		help="Formato da imagem de anotação (máscara) [default: 'png']",
	)
	parser.add_argument(
		'--cor_indice', '-ci', type=boolean, default=False, metavar='cor_indice', required=False,
		help='Especifica se as cores RGB serão convertidas em índices das cores. '
			 'Exemplo: Cor 0, 1, 2, para quando houver 3 classes. [default: True]',
	)
	parser.add_argument(
		'--dir_imagens', '-di', type=au.dir_path, default=None, metavar='dir_imagens', required=False,
		help="Diretório onde estão as imagens a serem preparadas [default: '../dataset/JPEGImages']",
	)
	parser.add_argument(
		'--dir_anotacoes', '-da', type=au.dir_path, default=None,
		metavar='dir_anotacoes', required=False,
		help="Diretório onde estão as imagens de anotações"
			 " (máscaras) a serem preparadas [default: '../dataset/SegmentationClass']",
	)
	parser.add_argument(
		'--tamanho', '-t', nargs='+', default=['512', '512'], metavar='tamanho', required=False,
		help="Altura e largura em pixels que as imagens serão salvas [default: [512 512]]",
	)
	parser.add_argument(
		'--porcent_treino', '-ptreino', type=float, default=0.7, metavar='porcent_treino', required=False,
		help='Porcentagem de dados para treino',
	)
	parser.add_argument(
		'--porcent_val', '-pval', type=float, default=0.2, metavar='porcent_val', required=False,
		help='Porcentagem de dados para validação',
	)
	parser.add_argument(
		'--porcent_teste', '-pteste', type=float, default=0.1, metavar='porcent_teste', required=False,
		help='Porcentagem de dados para teste',
	)
	parser.add_argument(
		'--particionar', '-p', type=boolean, default=False, metavar='particionar', required=False,
		help='Especifica se os dados serão particionados em treino, validação e teste. [default: False]',
	)
	parser.add_argument(
		'--equal_histograma', '-eh', type=boolean, default=False, metavar='equalizar', required=False,
		help='Define se as imagens serão pré-processadas (histograma equalizado). [default: False]',
	)
	parser.add_argument(
		'--transf_morfologica', '-tm', type=boolean, default=False, metavar='trans_morf', required=False,
		help='Define se transformações morfológicas serão usadas nas imagens [default: False]',
	)
	_args = parser.parse_args()
	args_obj = CLIArgs(
		dir_imagens=_args.dir_imagens, dir_anotacoes=_args.dir_anotacoes,
		converter_cor_indice=_args.cor_indice, particionar=_args.particionar,
		porcent_treino=_args.porcent_treino, porcent_val=_args.porcent_val,
		porcent_teste=_args.porcent_teste,
		tamanho=[int(t) for t in _args.tamanho] if _args.tamanho else None,
		equalizar_histograma=_args.equalizar, transformacao_morfologica=_args.trans_morf,
		imagem_formato=_args.formato_imagem, anotacao_formato=_args.formato_anotacao
	)
	return args_obj


def print_barra_prog(
		iteracao: int, total: float, prefix='Progresso', suffix='Finalizado',
		decimals=1, comprimento=100, fill='█', print_char_end='\r'
):
	prog = 100 * (iteracao / max(float(total), 1))
	percent = ("{0:." + str(decimals) + "f}").format(prog)
	qtd_atingida = int(comprimento * iteracao // max(total, 1))
	bar = fill * qtd_atingida + '-' * (comprimento - qtd_atingida)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_char_end)

	if iteracao == total:
		print()


def print_msg(msg: str):
	print(f"--> {msg} [{datetime.datetime.now()}]")
