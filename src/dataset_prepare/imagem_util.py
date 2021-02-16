import io
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL.Image import Image


def aplicar_transf_morf(img: Image, kernel_tamanho=(32, 32)):
	kernel = np.ones(kernel_tamanho, np.uint8)
	th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
	bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
	return cv2.subtract(cv2.add(img, th), bh)


def converter_img_para_bytes(img: Image, formato: str) -> bytes:
	pixels = np.array(img)
	img_array = PILImage.fromarray(pixels)
	img_bytes = io.BytesIO()
	img_array.save(img_bytes, format=formato)
	return img_bytes.getvalue()


def imagepil_para_opencv_hist_equal(img: Image):
	numpy_image = np.array(img.convert('L'))
	img_open_cv = cv2.equalizeHist(np.array(numpy_image))
	return img_open_cv


def imagepil_para_opencv(img: Image):
	# numpy_image = np.array(img.convert('L'))
	numpy_image = np.array(img.convert('RGB'))
	img_open_cv = np.array(numpy_image)
	return img_open_cv


def redimensionar(img: Image, altura_largura=(350, 350)) -> Image:
	return img.resize(altura_largura)


def substituir_cores_por_indice(
		imagem: Image, cores_indice: Dict[Tuple[int, int, int], Tuple[int, int, int]]
) -> Image:
	img_arr = np.array(imagem, dtype=np.uint8)
	pixels = np.zeros(shape=img_arr.shape, dtype=np.uint8)

	for i, cor in enumerate(cores_indice):
		pixels[np.all(img_arr == cor, axis=-1)] = i

	img_cores_indexadas: Image = PILImage.fromarray((pixels).astype(np.uint8))
	return img_cores_indexadas
