import argparse
import asyncio
import datetime

from os import path
from pathlib import Path
from random import shuffle, uniform
from typing import Dict, List, Coroutine, Optional, Tuple, Iterable
from PIL import Image as PILImage
from PIL.Image import Image

import util.arquivo_util as au
import util.imagem_util as iu
from util.dataset_util import atualizar_imagens


def gerar_lista_arquivos(
		dir_treino: str, dir_val: str, dir_saida='../dataset/ImageSets',
		porcent_treino=0.7, aleatorio=True
):
	dic: Dict[str, set] = au.gerar_lista_validacao_treino(
		dir_treino, dir_val, restringir_somente_treino=True
	)

	def criar_arq_lista_nomes(nome_arq_saida: str, nomes_arq: List[str]):
		Path(dir_saida).mkdir(parents=True, exist_ok=True)

		with open(nome_arq_saida, 'w') as arq:
			ult_indice = len(nomes_arq) - 1
			arq.writelines([nome + '\n' if i != ult_indice else nome for i, nome in enumerate(nomes_arq)])

	treino_ord = sorted(dic['treino'])

	if aleatorio:
		shuffle(treino_ord)

	treino_fatia = sorted(treino_ord[:int(len(treino_ord) * porcent_treino)])

	validacao_ord = sorted(set(treino_ord) - set(treino_fatia))

	criar_arq_lista_nomes(f'{dir_saida}/train.txt', treino_fatia)
	criar_arq_lista_nomes(f'{dir_saida}/val.txt', validacao_ord)
	criar_arq_lista_nomes(f'{dir_saida}/trainval.txt', treino_fatia + validacao_ord)


def gerar_dados_adicionais(
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
		porcent_ruido = uniform(0, 5)

		img = iu.modificar_img(img_path, grau_rotacao, porcent_zoom, porcent_ruido, True, alt_larg)
		salvar_img(path.join(dir_saida_imagem, path.basename(img_path)), img, i)

		anot = iu.modificar_img(anot_path, grau_rotacao, porcent_zoom, alt_larg=alt_larg)
		salvar_img(path.join(dir_saida_anot, path.basename(anot_path)), anot, i)

	# Salvar imagens originais
	path_imgs_orig_dir_saida = {img_path: dir_saida_imagem, anot_path: dir_saida_anot}

	for img_path in path_imgs_orig_dir_saida:
		with open(path.join(path_imgs_orig_dir_saida[img_path], path.basename(img_path)), 'wb') as arq:
			img = PILImage.open(img_path)
			arq.write(iu.converter_img_para_bytes(img, img_path.rsplit('.', 1)[1]))


def padronizar_imgs(
		arquivos_path: Iterable[str],
		cores_indice: Dict[Tuple[int, int, int], int],
		tamanho: Optional[Tuple[int, int]]
):
	for arq_path in arquivos_path:
		img = PILImage.open(arq_path)

		if (cores_indice):
			img = iu.substituir_cores_por_indice(img, cores_indice)

		if (tamanho):
			img = iu.redimensionar(img, tuple(tamanho))

		with open(arq_path.replace('.png', '.jpeg'), 'wb') as saida:
			saida.write(iu.converter_img_para_bytes(img, 'jpeg'))


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
		'--gerar_diretorio', '-gd', type=str, default='../dataset/dados_gerados',
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
		'--dir_imagens', '-di', type=au.dir_path, default='../dataset/JPEGImages',
		metavar='dir_imagens', required=True,
		help="Diretório onde estão as imagens a serem preparadas [default: '../dataset/JPEGImages']",
	)
	parser.add_argument(
		'--dir_anotacoes', '-da', type=au.dir_path, default='../dataset/SegmentationClass',
		metavar='dir_anotacoes', required=True,
		help="Diretório onde estão as imagens de anotações"
			 " (máscaras) a serem preparadas [default: '../dataset/SegmentationClass']",
	)
	parser.add_argument(
		'--tamanho', '-t', nargs='+', default=None, metavar='tamanho', required=False,
		help="Altura e largura em pixels que as imagens"
			 " serão salvas [default: None]",
	)

	return parser.parse_args()


def print_msg(msg: str):
	print(f"--> {msg} [{datetime.datetime.now()}]")


def main():
	args = ler_args()
	dir_saida_img = args.dir_imagens
	dir_saida_anot = args.dir_anotacoes

	if (args.download):
		tarefas: Coroutine = atualizar_imagens(
			dir_saida_img, dir_saida_anot, False, virus=True,
			bacteria=True, covid=True
		)
		print_msg('Download: Iniciado!')
		asyncio.get_event_loop().run_until_complete(tarefas)
		print_msg('Download: Finalizado!')

	causa_cor = {
		'covid': (0, 192, 128),
		'bacterial': (128, 192, 128),
		'viral': (128, 64, 128)
	}

	img_fmt = args.formato_imagem
	anot_fmt = args.formato_anotacao
	anot_sufixo = args.sufixo

	if args.gerar and args.gerar > 0:
		print_msg('Geração de dados: Iniciada!')
		arquivos_path = au.obter_path_arquivos(dir_saida_anot)

		for arq_path in arquivos_path:
			img_nome = path.basename(arq_path).replace(anot_sufixo, '').replace(anot_fmt, img_fmt)

			img_covid = iu.contem_cor(PILImage.open(arq_path), causa_cor['covid'])

			gerar_dados_adicionais(
				path.join(dir_saida_img, img_nome), arq_path,
				dir_saida=args.gerar_diretorio, qtd=args.gerar if not img_covid else 0
			)
		print_msg(f'Geração de dados: Finalizada! ({len(arquivos_path)} anotações e {len(arquivos_path)} imagens)')

	cor_indice = {
		# covid
		(0, 192, 128): 1,
		# bacterial
		(128, 192, 128): 2,
		# viral
		(128, 64, 128): 3
	}
	# cor_indice = {cor_indice[cor]: cor for cor in cor_indice}

	if (args.cor_indice or args.tamanho):
		arquivos_path = {}
		msg_conversao_cor = 'Conversão de cor' if args.cor_indice else ''
		msg_tamanho = 'Padronização de tamanho' if args.cor_indice else ''
		msgs = ' e '.join([msg for msg in [msg_conversao_cor, msg_tamanho] if msg.strip()])
		print_msg(f'{msgs}: Iniciada!')

		if (args.gerar_diretorio and args.gerar > 0):
			arquivos_path = au.obter_path_arquivos(args.gerar_diretorio, (anot_fmt))

		else:
			imagens_path = au.obter_path_arquivos(dir_saida_img)
			anotacoes_path = au.obter_path_arquivos(dir_saida_anot)
			arquivos_path = imagens_path.union(anotacoes_path)

		tamanho = [int(t) for t in args.tamanho] if args.tamanho else None
		converter_cor_indice = cor_indice if args.cor_indice else None
		padronizar_imgs(arquivos_path, converter_cor_indice, tamanho)
		print_msg(f'{msgs}: Finalizada!')

	# gerar_lista_arquivos(dir_saida_anot, dir_saida_img)
	return 0


# Baixar arquivos
# Gerar dados adicionais
# Converter cores para indices
# Renomear arquivos

main()
