import asyncio
import glob
import io
import warnings
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Coroutine, Iterable, Set, Optional, Tuple, Union
from PIL import Image
from bs4 import BeautifulSoup
from pandas import DataFrame

from arquivo_util import baixar_arquivos

warnings.filterwarnings('ignore', 'This pattern has match groups')


async def baixar_imagens(
		urls_download: List[str], dir_saida='../../dataset', formatos=('jpeg', 'jpg', 'png'),
		formato_saida='jpeg', label_cor: int = None,
		altura_largura=(600, 600)
) -> None:
	def preparar_img(img_bytes: io.BytesIO, altura_largura=altura_largura) -> bytes:
		img_original_pb: Image = Image.open(img_bytes).convert('LA').convert('RGB')
		img_redim: Image = img_original_pb.resize(altura_largura)
		img_bytes = io.BytesIO()
		pixels = np.array(img_redim)
		cor_branca = (255, 255, 255)

		if (label_cor is not None):
			pixels[np.all(pixels == cor_branca, axis=-1)] = label_cor

		img_array = Image.fromarray(pixels)
		img_array.save(img_bytes, formato_saida)
		return img_bytes.getvalue()

	return await baixar_arquivos(urls_download, dir_saida, formatos, preparar_img, formato_saida)


def extrair_nome_anotacoes(url: str, classe_css: str) -> List[str]:
	# Capture o html cru da página da url recebida
	html_content = requests.get(url).content

	soup = BeautifulSoup(html_content, 'html.parser')

	# Filtre somente os elementos com a classe CSS desejada e retorne seu texto
	links = list(soup.findAll('a', attrs={'class': classe_css}))
	return [link.text for link in links]


def filtrar_por_causas(
		df: DataFrame, col_causa='finding', bacteria=False, covid=False, virus=False, fungo=False
) -> DataFrame:
	virus_regex = '(.*viral)+((?!covid).)*$'
	opcoes: dict = {'bacteria': bacteria, 'covid': covid, virus_regex: virus, 'fungal': fungo}

	# Concatenação usada para filtrar por nenhum ou múltiplas opções
	condicao = '|'.join([chave for chave in opcoes if opcoes[chave]])
	return df[df[col_causa].str.contains(condicao, regex=True, case=False)]


def relacionar_anotacao_com_causa_pd(
		df_meta: DataFrame, arqs_anotacoes_nome: List[str], causas: Iterable[str],
		col_causa='finding', col_arq='filename', formatos=('jpeg', 'jpg', 'png'),
		sufixo_anotacao='_mask.png'
) -> Dict[str, List[str]]:
	df_arq_causa = df_meta[[col_arq, col_causa]].copy(deep=True)

	# Extraia apenas o nome do arquivo de anotação, sem extensão e sufixo
	anotacoes_nome: Set[str] = {arq_nome.split(sufixo_anotacao)[0] for arq_nome in arqs_anotacoes_nome}
	anotacoes_nomes = map(lambda nome: [f'{nome}.{fmt}' for fmt in formatos], anotacoes_nome)
	anot_arqs_nome = [nome for nomes in anotacoes_nomes for nome in nomes]

	df_arq_causa: DataFrame = df_arq_causa[df_arq_causa[col_arq].isin(anot_arqs_nome)]

	# Para cada causa, abrevie os valores (Exemplo: 'Pneumonia/Viral/COVID-19' por 'viral')
	for causa in causas:
		regex = fr'(^.*{causa}.*$)'
		df_arq_causa[col_causa] = df_arq_causa[col_causa].str.replace(regex, causa, regex=True, case=False)

	def fn_converter_nome_mascara(arq_nome: str, sufixo=sufixo_anotacao):
		return arq_nome.rsplit('.', 1)[0] + sufixo

	df_arq_causa[col_arq] = df_arq_causa[col_arq].transform(fn_converter_nome_mascara)

	return {
		causa: df_arq_causa.loc[df_arq_causa[col_causa] == causa, col_arq].to_list()
		for causa in causas
	}


async def baixar_salvar_imagens(
		df_meta: DataFrame, causas: Iterable[str], dir_saida: str, agrupar_por_causa=False,
		altura_largura=(600, 600)
) -> Coroutine:
	url_imagens_github = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/'
	corotinas: List[Coroutine] = []

	for causa in causas:
		df_part = filtrar_por_causas(
			df_meta,
			bacteria='bact' in causa,
			fungo='fung' in causa,
			covid='covid' in causa,
			virus='vir' in causa
		)

		# Concatene a url de download COM o nome de cada arquivo, para baixá-los
		download_urls = [f'{url_imagens_github}/{arq_nome}' for arq_nome in df_part['filename'].to_list()]
		corotinas.append(
			baixar_imagens(
				download_urls, f"{dir_saida}/{causa if agrupar_por_causa else ''}",
				altura_largura=altura_largura
			)
		)

	return await asyncio.gather(*corotinas)


async def baixar_salvar_anot(
		df_imgs: DataFrame, causas: Iterable[str], dir_saida: str, agrupar_por_causa=False,
		altura_largura=(600, 600), sufixo_anotacao='.png'
) -> Coroutine:
	url_anotacoes_gh = 'https://github.com/ieee8023/covid-chestxray-dataset/tree/master/annotations/lungVAE-masks'

	# Extraia o nome dos arquivos de imagens anotadas
	arq_nomes = extrair_nome_anotacoes(url_anotacoes_gh, 'js-navigation-open link-gray-dark')

	url_anotacoes_down = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/annotations/lungVAE-masks'

	causa_anot_nome: Dict[str, List[str]] = relacionar_anotacao_com_causa_pd(
		df_imgs, arq_nomes, causas
	)
	tarefas: List[Coroutine] = []
	# viral (128, 64, 128): 1,
	# covid (0, 192, 128): 2,
	# bacterial (128, 192, 128): 3

	cores_label = {'viral': 1, 'covid': 2, 'bacterial': 3}

	# Concatene a url de download COM o nome de cada arquivo, para baixá-los
	for causa in causa_anot_nome:
		urls_download = [f'{url_anotacoes_down}/{arq_nome}' for arq_nome in causa_anot_nome[causa]]
		tarefa = baixar_imagens(
			urls_download, f"{dir_saida}/{causa if agrupar_por_causa else ''}",
			formato_saida='png', label_cor=cores_label[causa], altura_largura=altura_largura
		)
		tarefas.append(tarefa)

	return await asyncio.gather(*tarefas)


def imgs_nao_existem(
		dir_img: str, imgs_nome: List[str], dir_treino: str, imgs_treino_nome: List[str],
		img_ext='jpeg', treino_ext='png'
) -> List[str]:
	imgs = glob.glob(f'{dir_img}/**/*.{img_ext}', recursive=True)
	imgs_treino = glob.glob(f'{dir_treino}/**/*.{treino_ext}', recursive=True)

	def img_existe(lista_imgs: List[str], img_nome: str) -> bool:
		return any(img_nome in caminho for caminho in lista_imgs)

	imgs_existentes = {nome for nome in imgs_nome if not img_existe(imgs, nome)}
	imgs_treino_existentes = {nome for nome in imgs_treino_nome if not img_existe(imgs_treino, nome)}

	print(len(imgs_existentes))
	print(len(imgs_treino_existentes))

	return list(imgs_existentes.union(imgs_treino_existentes))


async def atualizar_imagens(
		dir_saida_img: Optional[str], dir_saida_anot: Optional[str],
		agrupar_por_causa=False, bacteria=False, covid=False, virus=False,
		altura_largura=(600, 600)
) -> Coroutine:
	causas = ('covid', 'viral', 'bacterial')

	# Baixar planilha de metadados das IMAGENS
	url_metadados = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/metadata.csv'
	df_imgs = filtrar_por_causas(pd.read_csv(url_metadados), bacteria=bacteria, covid=covid, virus=virus)

	tarefas: List[Coroutine] = []
	# nomes_imgs = df_imgs['filename'].to_list()

	# df_imgs = df_imgs[df_imgs['filename'].isin(imgs_nao_existem(dir_saida_img, nomes_imgs, dir_saida_anot, nomes_imgs))]

	if (dir_saida_img):
		tarefas.append(baixar_salvar_imagens(df_imgs, causas, dir_saida_img, agrupar_por_causa, altura_largura))

	if (dir_saida_anot):
		tarefas.append(baixar_salvar_anot(df_imgs, causas, dir_saida_anot, agrupar_por_causa, altura_largura))

	return await asyncio.gather(*tarefas)
