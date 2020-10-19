import asyncio

import numpy as np
import glob
import io
from pathlib import Path
from random import randint, uniform
from typing import Dict, List, Coroutine
from PIL import Image as PILImage
from PIL.Image import Image

import imagem_transform as it
from arquivo_util import gerar_lista_validacao_treino
from imagem_util import atualizar_imagens


def gerar_lista_arquivos(dir_treino: str, dir_val: str, dir_saida='../dataset/ImageSets'):
	dic: Dict[str, set] = gerar_lista_validacao_treino(dir_treino, dir_val, restringir_somente_treino=True)

	def criar_arq_lista_nomes(nome_arq_saida: str, nomes_arq: List[str]):
		Path(dir_saida).mkdir(parents=True, exist_ok=True)

		with open(nome_arq_saida, 'w') as arq:
			ult_indice = len(nomes_arq) - 1
			arq.writelines([nome + '\n' if i != ult_indice else nome for i, nome in enumerate(nomes_arq)])

	treino_ord = sorted(dic['treino'])
	validacao_ord = sorted(dic['validacao'])
	criar_arq_lista_nomes(f'{dir_saida}/train.txt', treino_ord)
	criar_arq_lista_nomes(f'{dir_saida}/val.txt', validacao_ord)
	criar_arq_lista_nomes(f'{dir_saida}/trainval.txt', treino_ord + validacao_ord)


def gerar_dados_adicionais(img_treino_path: str, img_path: str):
	def modificar_img(
			img_url: str, grau: float, gauss: int, zoom: float, chance_ruido=0
	) -> Image:
		img: Image = PILImage.open(img_url)
		img = it.rotacionar(img, grau)
		img = it.gauss_blur(img, gauss)
		img = it.zoom(img, zoom)
		# img = it.ruido(img, True, chance_ruido)
		return img

	def salvar_img(arq_path: str, img: Image, i: int):
		path_ext = arq_path.rsplit('.', 1)
		mascara = 'mask' in arq_path
		novo_path = f'{path_ext[0]}_{i}.{path_ext[1]}'

		if (mascara):
			novo_path = f"{path_ext[0].replace('_mask', '')}_{i}_mask.{path_ext[1]}"

		img_bytes = io.BytesIO()
		pixels = np.array(img)
		img_array = PILImage.fromarray(pixels)
		img_array.save(img_bytes, path_ext[1])

		with open(novo_path, 'wb') as arq:
			arq.write(img_bytes.getvalue())

	for i in range(1, 11):
		grau_rotacao = uniform(-15, 15)
		gauss_fator = randint(0, 2)
		porcent_zoom = uniform(0, 15)
		porcent_ruido = uniform(0, 15)

		img = modificar_img(img_path, grau_rotacao, gauss_fator, porcent_zoom, porcent_ruido)
		salvar_img(img_path, img, i)

		img_treino = modificar_img(img_treino_path, grau_rotacao, gauss_fator, porcent_zoom)
		salvar_img(img_treino_path, img_treino, i)


def obter_path_arquivos(dir: str, formatos=('jpg', 'jpeg', 'png')) -> List[str]:
	return [
		nome_arq
		for arquivos_por_ext in [glob.glob(f'{dir}/**/*.{ext}', recursive=True) for ext in formatos]
		for nome_arq in arquivos_por_ext
	]


def main():
	pasta_img = 'JPEGImages'
	pasta_anot = 'SegmentationClassRaw'
	dir_saida_img = f'../dataset/{pasta_img}'
	dir_saida_anot = f'../dataset/{pasta_anot}'

	# tarefas: Coroutine = atualizar_imagens(
	# 	dir_saida_img, dir_saida_anot, False, virus=True,
	# 	bacteria=True, covid=True
	# )
	# asyncio.get_event_loop().run_until_complete(tarefas)

	arquivos_path = obter_path_arquivos(f'{dir_saida_anot}')

	for arq_path in arquivos_path:
		gerar_dados_adicionais(
			arq_path,
			arq_path.replace('_mask.png', '.jpeg').replace(pasta_anot, pasta_img)
		)
	gerar_lista_arquivos(dir_saida_anot, dir_saida_img)
	return 0


main()
