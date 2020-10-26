import io
from random import random
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image as PILImage
from PIL import ImageFilter
from PIL.Image import Image
from PIL.ImageOps import grayscale


def contem_cor(imagem: Image, cor: Tuple[int, int, int]) -> bool:
	pixels = np.array(imagem)
	# return cor in np.all(imagem)
	return len(pixels[np.all(pixels == cor, axis=-1)]) > 0


def converter_img_para_bytes(img: Image, formato: str) -> bytes:
	pixels = np.array(img)
	img_array = PILImage.fromarray(pixels)
	img_bytes = io.BytesIO()
	img_array.save(img_bytes, format=formato)
	return img_bytes.getvalue()


def gauss_blur(img: Image, fator=1) -> Image:
	return img.filter(ImageFilter.GaussianBlur(fator))


def modificar_img(
		img_path: str, grau_rotacao: float, zoom_pct: float, chance_ruido=0, img_cinza=False,
		alt_larg: Optional[Tuple[int, int]] = None
) -> Image:
	img: Image = PILImage.open(img_path) if not alt_larg else redimensionar(PILImage.open(img_path), alt_larg)
	return ruido(zoom(rotacionar(img, grau_rotacao), zoom_pct), img_cinza, chance_ruido)


def redimensionar(img: Image, altura_largura=(350, 350)) -> Image:
	return img.resize(altura_largura)


def rotacionar(img: Image, angulo: float) -> Image:
	return img.rotate(angulo)


def ruido(img: Image, img_cinza=False, porcentagem=10):
	img_copia = img.copy()
	pixels = img_copia.load()
	larg, alt = img_copia.size
	min_probabil = 1 - porcentagem / 100

	# 0 = ruído branco, 1 = ruído preto
	cor_ruido = 1 if img_cinza else 0

	for x in range(larg):
		for y in range(alt):
			if random() > min_probabil:
				pixels[x, y] = cor_ruido
	return img_copia


def substituir_cores_por_indice(
		imagem: Image, cores_indice: Dict[Tuple[int, int, int], int]
) -> Image:
	pixels = np.array(imagem)

	for cor in cores_indice:
		pixels[np.all(pixels == cor, axis=-1)] = cores_indice[cor]

	img_cores_indexadas: Image = PILImage.fromarray(pixels)
	return img_cores_indexadas


def tons_cinza(img: Image) -> Image:
	return grayscale(img)


def zoom(img: Image, porcentagem: float = 10) -> Image:
	"""
	Aplica uma taxa de zoom ao centro da imagem de entrada
	:param img: Objeto contendo imagem interpretada pelo Pillow (PIL)
	:param porcentagem: Por padrão aplicará 10% de zoom
	:return: Imagem de mesmas dimensões, porém com zoom
	"""
	larg_alt = img.width, img.height
	perc = porcentagem / 100
	arr = np.array(larg_alt) * perc
	arr2 = np.array(larg_alt) * (1 - perc)
	box = np.concatenate([arr, arr2])
	return img.resize(larg_alt, box=tuple(box))

# def translacionar(img: Image, esq=0, baixo=0, dir=0, cima=0):
# 	image = img.copy()
# 	shift = 20
# 	# Shift the image 5 pixels
# 	width, height = image.size
# 	shifted_image = Image.new('RGB', (width + shift, height))
# 	shifted_image.paste(image, (shift, 0))
# 	shifted_image.resize(img.size, box=(0, 0, width, height))
