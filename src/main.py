import argparse
import asyncio
import datetime
import shutil
from os import path, remove
from pathlib import Path
from random import uniform
from typing import Dict, List, Coroutine, Optional, Tuple, Iterable, Set

from PIL import Image as PILImage
from PIL.Image import Image

import util.arquivo_util as au
import util.imagem_util as iu
from datasets.dataset_util import atualizar_imagens


def gerar_lista_arquivos(
		dir_treino: str, dir_val: str, dir_saida: str = None,
		porcent_treino=0.7, porcent_val=0.2, porcent_teste=0.1
):
	def criar_arq_lista_nomes(nome_arq_saida: str, nomes_arq: List[str]):
		Path(dir_saida).mkdir(parents=True, exist_ok=True)

		with open(nome_arq_saida, 'w') as arq:
			ult_indice = len(nomes_arq) - 1
			arq.writelines([nome + '\n' if i != ult_indice else nome for i, nome in enumerate(nomes_arq)])

	dir_saida = dir_saida or path.join(path.dirname(dir_treino), 'ImageSets')
	dic: Dict[str, Set[str]] = au.gerar_lista_validacao_treino(
		dir_treino, dir_val, restringir_somente_treino=True
	)
	arqs_nome = dic['treino']

	arq_nome_bacteria = sorted({nome for nome in arqs_nome if nome.endswith('bacterial')})
	arq_nome_covid = sorted({nome for nome in arqs_nome if nome.endswith('covid')})
	arq_nome_virus = sorted({nome for nome in arqs_nome if nome.endswith('viral')})
	fatia_nome_porcent = {'train': porcent_treino, 'val': porcent_val, 'test': porcent_teste}
	porcent_inic = 0

	for nome_subset in fatia_nome_porcent:
		porcent_fim = porcent_inic + fatia_nome_porcent[nome_subset]
		arqs_bacteria = arq_nome_bacteria[
						int(len(arq_nome_bacteria) * porcent_inic):int(len(arq_nome_bacteria) * porcent_fim)]
		arqs_covid = arq_nome_covid[int(len(arq_nome_covid) * porcent_inic):int(len(arq_nome_covid) * porcent_fim)]
		arqs_virus = arq_nome_virus[int(len(arq_nome_virus) * porcent_inic):int(len(arq_nome_virus) * porcent_fim)]
		arqs_nome_salvar = [*arqs_bacteria, *arqs_covid, *arqs_virus]
		criar_arq_lista_nomes(f'{dir_saida}/{nome_subset}.txt', arqs_nome_salvar)
		porcent_inic += fatia_nome_porcent[nome_subset]


def gerar_anot_img_adicional(
		img_path: str, anot_path: str, dir_saida: Optional[str] = None,
		alt_larg: Optional[Tuple[int, int]] = None, qtd=20
):
	def salvar_img(arq_path: str, img: Image, indice: int):
		path_ext = arq_path.rsplit('.', 1)
		mascara = 'mask' in arq_path
		novo_path = f'{path_ext[0]}_{indice}.{path_ext[1]}'

		if mascara:
			novo_path = f"{path_ext[0].replace('_mask', '')}_{i}_mask.{path_ext[1]}"

		with open(novo_path, 'wb') as arq:
			arq.write(iu.converter_img_para_bytes(img, novo_path.rsplit('.', 1)[1]))

	dir_saida_imagem = path.dirname(img_path) if dir_saida is None else path.join(dir_saida, 'imagens')
	dir_saida_anot = path.dirname(anot_path) if dir_saida is None else path.join(dir_saida, 'anotacoes')

	Path(dir_saida_imagem).mkdir(parents=True, exist_ok=True)
	Path(dir_saida_anot).mkdir(parents=True, exist_ok=True)

	for i in range(1, qtd + 1):
		grau_rotacao = uniform(-25, 25)
		porcent_zoom = uniform(0, 20)
		porcent_ruido = uniform(0, 10)

		img = iu.modificar_img(img_path, grau_rotacao, porcent_zoom, porcent_ruido, True, alt_larg)
		salvar_img(path.join(dir_saida_imagem, path.basename(img_path)), img, i)

		anot = iu.modificar_img(anot_path, grau_rotacao, porcent_zoom, alt_larg=alt_larg)
		salvar_img(path.join(dir_saida_anot, path.basename(anot_path)), anot, i)

	# Salvar imagens originais
	path_imgs_orig_dir_saida = {
		img_path: dir_saida_imagem, anot_path: dir_saida_anot}

	for img_path in path_imgs_orig_dir_saida:
		with open(path.join(path_imgs_orig_dir_saida[img_path], path.basename(img_path)), 'wb') as arq:
			img = PILImage.open(img_path)
			arq.write(iu.converter_img_para_bytes(img, img_path.rsplit('.', 1)[1]))


def padronizar_imgs(
		arquivos_path: Iterable[str],
		cores_indice: Optional[Dict[Tuple[int, int, int], Tuple[int, int, int]]] = None,
		tamanho: Optional[Tuple[int, int]] = None,
		formato: Optional[str] = None
):
	l = len(list(arquivos_path))
	print_barra_prog(0, l, length=50)

	for i, arq_path in enumerate(arquivos_path):
		img = PILImage.open(arq_path).convert('RGB')
		formato = formato or ('png' if arq_path.endswith('.png') else 'jpeg')

		if (cores_indice):
			img = iu.substituir_cores_por_indice(img, cores_indice)

		if (tamanho):
			img = iu.redimensionar(img, tuple(tamanho))

		novo_arq_path = arq_path.rsplit('.', 1)[0] + f'.{formato.lower()}'

		with open(novo_arq_path, 'wb') as saida:
			saida.write(iu.converter_img_para_bytes(img, formato))

		if (novo_arq_path != arq_path):
			remove(arq_path)

		print_barra_prog(i + 1, l, length=50)


def boolean(valor: str) -> bool:
	if isinstance(valor, bool):
		return valor

	if valor.lower() in ('yes', 'true', 't', 'y', '1'):
		return True

	elif valor.lower() in ('no', 'false', 'f', 'n', '0'):
		return False

	else:
		raise argparse.ArgumentTypeError('Um valor lógico é esperado.')


def ler_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--download', '-d', type=boolean, default=False, metavar='download', required=False,
		help="Especifica se as imagens serão baixadas [default: False]"
	)
	parser.add_argument(
		'--gerar', '-gn', type=int, default=0, metavar='gerar', required=False,
		help='Quantidade de dados que serão gerados (Data augmentation)'
			 ' a partir de cada imagem / anotação [default: 0]',
	)
	parser.add_argument(
		'--gerar_diretorio', '-gd', type=str, default='',
		metavar='gerar_diretorio', required=False,
		help="Diretório de saída das imagens geradas "
			 "(por padrão será criado um diretório 'dados_gerados') [default: False]",
	)
	parser.add_argument(
		'--sufixo', '-sm', type=str, default='_mask', metavar='suxifo', required=False,
		help="Sufixo das imagens de anotação (Exemplo: Para uma"
			 "img 'img12.jpeg' e anotação 'img12_mask.png', o sufixo é '_mask') [default: '_mask']",
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
			 'Exemplo: Cor 0, 1, 2, para quando houver 3 classes. [default: False]',
	)
	parser.add_argument(
		'--dir_imagens', '-di', type=au.dir_path, default=None,
		metavar='dir_imagens', required=False,
		help="Diretório onde estão as imagens a serem preparadas [default: '../dataset/JPEGImages']",
	)
	parser.add_argument(
		'--dir_anotacoes', '-da', type=au.dir_path, default=None,
		metavar='dir_anotacoes', required=False,
		help="Diretório onde estão as imagens de anotações"
			 " (máscaras) a serem preparadas [default: '../dataset/SegmentationClass']",
	)
	parser.add_argument(
		'--tamanho', '-t', nargs='+', default=None, metavar='tamanho', required=False,
		help="Altura e largura em pixels que as imagens"
			 " serão salvas [default: None]",
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

	return parser.parse_args()


def print_msg(msg: str):
	print(f"--> {msg} [{datetime.datetime.now()}]")


def print_barra_prog(
		iteration, total, prefix='Progresso', suffix='Finalizado',
		decimals=1, length=100, fill='█', print_char_end='\r'
):
	prog = 100 * (iteration / max(float(total), 1))
	percent = ("{0:." + str(decimals) + "f}").format(prog)
	filled_len = int(length * iteration // max(total, 1))
	bar = fill * filled_len + '-' * (length - filled_len)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_char_end)

	if iteration == total:
		print()


def gerar_dados(
		dir_anot: str, dir_img: str, dir_saida: str, qtd: int, anot_sufixo='_mask'
) -> int:
	causa_cor = {
		'covid': (0, 192, 128), 'bacterial': (128, 192, 128), 'viral': (128, 64, 128)
	}
	anotacoes_path = au.obter_path_arquivos(dir_anot)
	imgs_path = au.obter_path_arquivos(dir_img)

	anotacoes_nome_path: Dict[str, str] = {
		path.basename(anot_path).rsplit('.', 1)[0].replace(anot_sufixo, ''): anot_path
		for anot_path in anotacoes_path
	}
	imagens_nome_path: Dict[str, str] = {
		path.basename(img_path).rsplit('.', 1)[0]: img_path
		for img_path in imgs_path
	}
	total_gerado = 0

	for arq_nome in anotacoes_nome_path:
		img_covid = iu.contem_cor(
			PILImage.open(anotacoes_nome_path[arq_nome]),
			causa_cor['covid']
		)
		qtd_ = qtd if not img_covid else 0
		total_gerado += qtd_
		gerar_anot_img_adicional(
			imagens_nome_path[arq_nome], anotacoes_nome_path[arq_nome],
			dir_saida=dir_saida, qtd=qtd_
		)
	return total_gerado


def main():
	args = ler_args()
	dir_saida_img = args.dir_imagens
	dir_saida_anot = args.dir_anotacoes

	porc_treino = args.porcent_treino
	porc_val = args.porcent_val
	porc_teste = args.porcent_teste

	img_fmt = args.formato_imagem
	anot_fmt = args.formato_anotacao
	tamanho = [int(t) for t in args.tamanho] if args.tamanho else None

	if (args.download):
		tarefas: Coroutine = atualizar_imagens(
			dir_saida_img, dir_saida_anot, False, virus=True,
			bacteria=True, covid=True, altura_largura=tamanho
		)
		print_msg('Download: Iniciado!')
		asyncio.get_event_loop().run_until_complete(tarefas)
		print_msg('Download: Finalizado!')

	cor_indice = {
		# covid
		(0, 192, 128): (1, 1, 1),
		# bacterial
		(128, 192, 128): (2, 2, 2),
		# viral
		(128, 64, 128): (3, 3, 3)
	}
	cor_indice_reverso = {
		# covid
		(1, 1, 1): (0, 192, 128),
		# bacterial
		(2, 2, 2): (128, 192, 128),
		# viral
		(3, 3, 3): (128, 64, 128)
	}

	if args.gerar and args.gerar > 0:
		print_msg('Geração de dados: Iniciada!')
		qtd_gerada = gerar_dados(
			dir_saida_anot, dir_saida_img, args.gerar_diretorio,
			args.gerar, args.sufixo
		)
		print_msg(f'Geração de dados: Finalizada! ({qtd_gerada} anotações e {qtd_gerada} imagens)')

	if (args.cor_indice or args.tamanho):
		msg_conversao_cor = 'Conversão de cor' if args.cor_indice else ''
		msg_tamanho = 'Padronização de tamanho' if args.tamanho else ''
		msgs = ' e '.join([msg for msg in [msg_conversao_cor, msg_tamanho] if msg.strip()])
		print_msg(f'{msgs}: Iniciada!')

		cores_indice = cor_indice if args.cor_indice else None
		anots_dir = path.join(args.gerar_diretorio, 'anotacoes') if args.gerar_diretorio else dir_saida_anot
		imgs_dir = path.join(args.gerar_diretorio, 'imagens') if args.gerar_diretorio else dir_saida_img

		anotacoes_path = au.obter_path_arquivos(anots_dir)
		imagens_path = au.obter_path_arquivos(imgs_dir)

		padronizar_imgs(anotacoes_path, cores_indice, tamanho, 'png')
		padronizar_imgs(imagens_path, tamanho=tamanho, formato='jpeg')
		print_msg(f'{msgs}: Finalizada!')

	gerar_lista_arquivos(
		dir_saida_anot, dir_saida_img, porcent_treino=porc_treino,
		porcent_val=porc_val, porcent_teste=porc_teste
	)
	return 0


def organizar_dados(
		imgs_path: str, anots_path: str, img_ext='jpeg', anot_ext='png'
):
	arqs_sets_path = '../dataset/custom/ImageSets'
	nomes_arq_sets = ['test', 'train', 'val']

	for nome_set in nomes_arq_sets:
		Path(path.join(imgs_path, nome_set)).mkdir()
		Path(path.join(anots_path, f'{nome_set}_gt')).mkdir()

	def copiar_img_anot(nome_set: str, nome_arq: str):
		img_nome = f'{nome_arq}.{img_ext}'
		anot_nome = f'{nome_arq}.{anot_ext}'
		shutil.copyfile(
			path.join(imgs_path, img_nome),
			path.join(f'{imgs_path}/{nome_set}', img_nome)
		)
		shutil.copyfile(
			path.join(anots_path, anot_nome),
			path.join(f'{anots_path}/{nome_set}_gt', anot_nome)
		)

	for nome_arq_set in nomes_arq_sets:
		with open(path.join(arqs_sets_path, f'{nome_arq_set}.txt'), 'r') as arq_set:
			linhas_nome_arq = arq_set.readlines()
			qtd_arqs = len(linhas_nome_arq)
			print(f'---> Copiando e organizando - Dataset {nome_arq_set}')
			print_barra_prog(0, qtd_arqs, length=50)

			for i, arq_nome in enumerate(linhas_nome_arq):
				copiar_img_anot(nome_arq_set, arq_nome.strip())
				print_barra_prog(i + 1, qtd_arqs, length=50)


main()

# organizar_dados(
# 	'../dataset/custom/JPEGImages',
# 	'../dataset/custom/SegmentationClassIndex',
# )
