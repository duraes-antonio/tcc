from os import path
from typing import Set, Tuple, Optional, List, Coroutine

import requests
from bs4 import BeautifulSoup

import util.arquivo_util as au


def gerar_url_github_raw(usuario: str, repo: str, dir='') -> str:
	return f'https://raw.githubusercontent.com/{usuario}/{repo}/master/{dir}'


def gerar_url_github_repo(username: str, repositorio: str) -> str:
	return f'https://github.com/{username}/{repositorio}/tree/master'


def extrair_nome_arquivos_github(
		url: str, classe_css='js-navigation-open link-gray-dark',
		extensoes: Optional[Tuple[str]] = None
) -> List[str]:
	# Capture o html cru da página da url recebida
	html_content = requests.get(url).content
	soup = BeautifulSoup(html_content, 'html.parser')

	# Filtre os itens com a classe e extensão desejada e retorne seu texto
	links = list(soup.findAll('a', attrs={'class': classe_css}))
	return [
		link.text for link in links
		if extensoes is None or link.text.endswith(extensoes)
	]


async def baixar_arquivos(
		url_github_dir: str, dir_salvar: str, extensoes=('png', 'jpeg', 'jpg'),
):
	# v7labs, covid-19-xray-dataset, tree, master, annotations/all-images-semantic-masks
	url_trechos = url_github_dir.split('github.com/')[1].split('/', 4)
	nome_anotacoes = extrair_nome_arquivos_github(url_github_dir, extensoes=extensoes)
	url_raw_dir = gerar_url_github_raw(url_trechos[0], url_trechos[1], url_trechos[-1])
	urls_raw_anots = [f'{url_raw_dir}/{anot}' for anot in nome_anotacoes]
	return await au.baixar_arquivos(urls_raw_anots, dir_salvar)
