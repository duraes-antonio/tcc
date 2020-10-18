from random import random
from PIL import ImageFilter
from PIL.Image import Image
from PIL.ImageOps import grayscale
import numpy as np


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


def rotacionar(img: Image, angulo: float) -> Image:
	return img.rotate(angulo)


def zoom(img: Image, porcentagem=10) -> Image:
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


def gauss_blur(img: Image, fator=1) -> Image:
	return img.filter(ImageFilter.GaussianBlur(fator))


def tons_cinza(img: Image) -> Image:
	return grayscale(img)

# def translacionar(img: Image, esq=0, baixo=0, dir=0, cima=0):
# 	image = img.copy()
# 	shift = 20
# 	# Shift the image 5 pixels
# 	width, height = image.size
# 	shifted_image = Image.new('RGB', (width + shift, height))
# 	shifted_image.paste(image, (shift, 0))
# 	shifted_image.resize(img.size, box=(0, 0, width, height))
