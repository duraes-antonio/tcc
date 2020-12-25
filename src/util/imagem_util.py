import io
from random import random
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageFilter
from PIL.Image import Image
from PIL.ImageOps import grayscale


def aplicar_transf_morf(img: Image, kernel_tamanho=(32, 32)):
	kernel = np.ones(kernel_tamanho, np.uint8)
	th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
	bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
	return cv2.subtract(cv2.add(img, th), bh)


def contem_cor(imagem: Image, cor: Tuple[int, int, int]) -> bool:
	pixels = np.array(imagem)
	return len(pixels[np.all(pixels == cor, axis=-1)]) > 0


def converter_img_para_bytes(img: Image, formato: str) -> bytes:
	pixels = np.array(img)
	img_array = PILImage.fromarray(pixels)
	img_bytes = io.BytesIO()
	img_array.save(img_bytes, format=formato)
	return img_bytes.getvalue()


def gauss_blur(img: Image, fator=1) -> Image:
	return img.filter(ImageFilter.GaussianBlur(fator))


def imagepil_para_opencv_hist_equal(img: Image):
	numpy_image = np.array(img.convert('L'))
	img_open_cv = cv2.equalizeHist(np.array(numpy_image))
	return img_open_cv


def imagepil_para_opencv(img: Image):
	# numpy_image = np.array(img.convert('L'))
	numpy_image = np.array(img.convert('RGB'))
	img_open_cv = np.array(numpy_image)
	return img_open_cv


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


def substituir_cor(
		img_bytes: io.BytesIO, cor_atual_rgb: Tuple[int, int, int],
		cor_nova_rgb: Tuple[int, int, int], restante_preto=True,
		formato_saida='jpeg'
) -> bytes:
	img_original: Image = PILImage.open(img_bytes).convert('RGB')
	pixels = np.array(img_original)

	if (restante_preto):
		pixels[np.all(pixels != cor_atual_rgb, axis=-1)] = 0

	pixels[np.all(pixels == cor_atual_rgb, axis=-1)] = cor_nova_rgb
	return converter_img_para_bytes(PILImage.fromarray(pixels), formato_saida)


def substituir_cores_por_indice(
		imagem: Image, cores_indice: Dict[Tuple[int, int, int], Tuple[int, int, int]]
) -> Image:
	img_arr = np.array(imagem, dtype=np.uint8)
	pixels = np.zeros(shape=img_arr.shape, dtype=np.uint8)

	for i, cor in enumerate(cores_indice):
		pixels[np.all(img_arr == cor, axis=-1)] = i

	img_cores_indexadas: Image = PILImage.fromarray((pixels).astype(np.uint8))
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
