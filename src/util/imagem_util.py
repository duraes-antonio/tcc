import asyncio
import io
import warnings
from typing import List, Dict, Coroutine, Iterable, Set, Optional, Tuple

import pandas as pd
import numpy as np
import requests
from PIL import Image
from bs4 import BeautifulSoup
from pandas import DataFrame

from arquivo_util import baixar_arquivos

warnings.filterwarnings('ignore', 'This pattern has match groups')


async def baixar_imagens(
		urls_download: List[str], dir_saida='../../dataset', formatos=('jpeg', 'jpg', 'png'),
		formato_saida='jpeg', label_cor: Optional[Tuple[int, int, int]] = None,
		altura=600, largura=600
) -> None:
	def preparar_img(img_bytes: io.BytesIO, altura=altura, largura=largura) -> bytes:
		img_original_pb: Image = Image.open(img_bytes).convert('LA').convert('RGB')
		img_redim: Image = img_original_pb.resize((altura, largura))
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

	def fn_converter_nome_masc(arq_nome: str, sufixo=sufixo_anotacao):
		return arq_nome.rsplit('.', 1)[0] + sufixo

	df_arq_causa[col_arq] = df_arq_causa[col_arq].transform(fn_converter_nome_masc)

	return {
		causa: df_arq_causa.loc[df_arq_causa[col_causa] == causa, col_arq].to_list()
		for causa in causas
	}


async def baixar_salvar_imagens(
		df_meta: DataFrame, causas: Iterable[str], dir_saida: str, agrupar_por_causa=False
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
		corotinas.append(baixar_imagens(download_urls, f"{dir_saida}/{causa if agrupar_por_causa else ''}"))

	return await asyncio.gather(*corotinas)


async def baixar_salvar_anot(
		df_imgs: DataFrame, causas: Iterable[str], dir_saida: str, agrupar_por_causa=False
) -> Coroutine:
	# Extraia o nome dos arquivos de imagens anotadas
	url_anotacoes_gh = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/annotations/lungVAE-masks/'
	arq_nomes = extrair_nome_anotacoes(url_anotacoes_gh, 'js-navigation-open link-gray-dark')

	causa_anot_nome: Dict[str, List[str]] = relacionar_anotacao_com_causa_pd(df_imgs, arq_nomes, causas)
	tarefas: List[Coroutine] = []
	cores_label = {'viral': (128, 64, 128), 'covid': (0, 192, 128), 'bacterial': (128, 192, 128)}

	# Concatene a url de download COM o nome de cada arquivo, para baixá-los
	for causa in causa_anot_nome:
		urls_download = [f'{url_anotacoes_gh}/{arq_nome}' for arq_nome in causa_anot_nome[causa]]
		tarefa = baixar_imagens(
			urls_download, f"{dir_saida}/{causa if agrupar_por_causa else ''}",
			formato_saida='png', label_cor=cores_label[causa]
		)
		tarefas.append(tarefa)

	return await asyncio.gather(*tarefas)


async def atualizar_imagens(
		dir_saida_img: Optional[str], dir_saida_anot: Optional[str],
		agrupar_por_causa=False, bacteria=False, covid=False, virus=False
) -> Coroutine:
	causas = ('covid', 'viral', 'bacterial')

	# Baixar planilha de metadados das IMAGENS
	url_metadados = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/metadata.csv'
	df_imgs = filtrar_por_causas(
		pd.read_csv(url_metadados), bacteria=bacteria, covid=covid, virus=virus
	)

	tarefas: List[Coroutine] = []

	if (dir_saida_img):
		tarefas.append(baixar_salvar_imagens(df_imgs, causas, dir_saida_img, agrupar_por_causa))

	if (dir_saida_anot):
		tarefas.append(baixar_salvar_anot(df_imgs, causas, dir_saida_anot, agrupar_por_causa))

	return await asyncio.gather(*tarefas)
