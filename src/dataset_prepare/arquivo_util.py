import argparse
import asyncio
import glob
import io
import warnings
from os import path, replace
from pathlib import Path
from shutil import copyfile
from typing import List, Optional, Callable, Tuple, Iterable, Set, Dict

from aiohttp import ClientSession

warnings.filterwarnings('ignore', 'This pattern has match groups')


async def baixar_arquivo(url_download: str, sessao: ClientSession) -> bytes:
	"""
	Dada uma sessão aiohttp e uma url, faz download e retorna o arquivo baixado
	:param url_download: Url do arquivo a ser baixado
	:param sessao: Sessão da biblioteca aiohttp
	:return: Bytes do arquivo baixado
	"""
	async with sessao.get(url_download, timeout=2000) as resposta:
		if resposta.status != 200:
			print(resposta)
		return await resposta.read()


async def baixar_arquivos(
		urls_download: List[str], dir_saida: str,
		formatos: Optional[Tuple[str]] = None,
		fn_map: Optional[Callable[[io.BytesIO], bytes]] = None,
		fmt_saida: Optional[str] = None,
		verbose=True
) -> None:
	# Monte o nome do diretório e o crie (se não existir)
	Path(dir_saida).mkdir(parents=True, exist_ok=True)

	# Filtre somente os nomes de arquivos com as extensões aceitas
	url_download_tipos_filtrados = [
		url for url in urls_download
		if formatos is None or url.lower().endswith(formatos)
	]

	imgs_nome = [path.basename(url_arq) for url_arq in url_download_tipos_filtrados]
	paths_saida = [f'{dir_saida}/{nome}' for _, nome in enumerate(imgs_nome)]

	semaforo = asyncio.Semaphore(8)

	async with semaforo:
		async with ClientSession(trust_env=True, timeout=2000) as sessaoHttp:
			lista_params = [(url_download_tipos_filtrados[i], sessaoHttp) for i in range(len(imgs_nome))]
			qtd_imagens = len(url_download_tipos_filtrados)
			downloads_concluidos = 0
			arquivos = await asyncio.gather(*[baixar_arquivo(*params) for params in lista_params])

			for i, arq in enumerate(arquivos):
				path_saida = paths_saida[i] if fmt_saida is None else f"{paths_saida[i].rsplit('.', 1)[0]}.{fmt_saida}"

				with open(path_saida.replace('_mask', ''), 'wb') as handle:
					handle.write(arq if fn_map is None else fn_map(io.BytesIO(arq)))
				downloads_concluidos += 1
	return


def gerar_lista_validacao_treino(
		dir_arq_treino: str, dir_arq_val: str, formatos=('jpeg', 'jpg', 'png'),
		restringir_somente_treino=True
) -> Dict[str, Set[str]]:
	"""
	Gera um dicionário com os nomes dos arquivos de treino, validação
	:param dir_arq_treino: Caminho do diretório de arquivos de treino
	:param dir_arq_val: Caminho do diretório de arquivos de validação
	:param formatos: Formatos de arquivos a serem buscados
	:param restringir_somente_treino: Retorna somente os arquivos de validação com um correspondente na pasta de treino
	:return: Iterável com chaves ('treino', 'validacao') e seus respectivos nomes de arquivos
	"""

	def obter_nome_arquivos(dir: str, formatos: Iterable[str], remover_ext=True) -> Set[str]:
		return {
			path.basename(nome_arq.rsplit('.', 1)[0] if remover_ext else nome_arq)
			for arquivos_por_ext in [glob.glob(f'{dir}/**/*.{ext}', recursive=True) for ext in formatos]
			for nome_arq in arquivos_por_ext
		}

	treino_arqs: Set[str] = obter_nome_arquivos(dir_arq_treino, formatos)
	val_arqs: Set[str] = obter_nome_arquivos(dir_arq_val, formatos)

	if restringir_somente_treino:
		val_arqs = {
			arq_treino.replace('_mask', '') for arq_treino in treino_arqs
			if arq_treino.replace('_mask', '') in val_arqs
		}
	return {'treino': treino_arqs, 'validacao': val_arqs}


def obter_path_arquivos(dir: str, formatos=('jpg', 'jpeg', 'png')) -> Set[str]:
	"""
	Busca o caminho completo de todos arquivos das extensões e diretório de entrada
	:param dir: Diretório inicial os arquivos serão procurados
	:param formatos: Extensões aceitas para os arquivos buscados
	:return: Lista de caminhos de todos arquivos do diretório que contenham as extensões
	"""
	return {
		nome_arq
		for arquivos_por_ext in [
			glob.glob(f'{dir}/**/*.{ext}', recursive=True)
			for ext in formatos
		]
		for nome_arq in arquivos_por_ext
	}


def renomear_arquivo(arq_path: str, novo_nome: str) -> None:
	"""
	Renomeia o arquivo do path de entrada
	:param arq_path: Path para o arquivo a ser renomeado
	:param novo_nome: Novo nome ou path com novo nome do arquivo
	:param arq_path:
	:param novo_nome:
	"""
	novo_path = path.join(path.dirname(arq_path), path.basename(novo_nome))
	replace(arq_path, novo_path)


def replace_nome_arquivo(dir_path: str, str_substituir: str, str_nova: str) -> None:
	arqs_nome = obter_path_arquivos(dir_path, ('csv',))

	for arq in arqs_nome:
		nome_novo = path.basename(arq).replace(str_substituir, str_nova)
		renomear_arquivo(arq, path.join(path.dirname(arq), nome_novo))


def extair_nome_arq(path_arq: str) -> str:
	"""
	Extrai somente o nome (sem extensão) do arquivo do path de entrada
	:param path_arq: Path para o arquivo
	:return: Nome do arquivo, sem extensão
	"""
	return path.basename(path_arq).rsplit('.', 1)[0]


def filtrar_arquivos(
		arq_selecionados: List[str], path_procura: str, path_saida: str
):
	# Monte o nome do diretório e o crie (se não existir)
	Path(path_saida).mkdir(parents=True, exist_ok=True)

	# Extraia apenas o nome (sem extensão) do arquivo dos caminhos de entrada
	arqs_selec_nome: Set[str] = {extair_nome_arq(arq_path) for arq_path in arq_selecionados}

	arqs_alvos_path = set(glob.glob(f'{path_procura}/*.*'))

	for arq_alvo_path in arqs_alvos_path:
		if extair_nome_arq(arq_alvo_path) in arqs_selec_nome:
			copyfile(arq_alvo_path, path.join(path_saida, path.basename(arq_alvo_path)))


def dir_path(caminho: str) -> str:
	if path.isdir(caminho):
		return caminho

	elif not path.exists(caminho):
		Path(caminho).mkdir(parents=True, exist_ok=True)
		return caminho
	# raise argparse.ArgumentTypeError(f"Erro: o caminho '{path}' não existe")

	elif path.isfile(caminho):
		raise argparse.ArgumentTypeError(f"Erro: o caminho '{path}' pertence a um arquivo e não diretório")

# replace_nome_arquivo('../resultados/unet', '19_drop', '19')
