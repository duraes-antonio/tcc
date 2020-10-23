import asyncio
import glob
import io
import warnings
from os import path
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Iterable, Set, Dict

from aiohttp import ClientSession

warnings.filterwarnings('ignore', 'This pattern has match groups')


async def baixar_arquivo(url_download: str, sessao: ClientSession) -> bytes:
	async with sessao.get(url_download, timeout=400) as resposta:
		if resposta.status != 200:
			print(resposta)
		return await resposta.read()


async def baixar_arquivos(
		urls_download: List[str], dir_saida: str, formatos: Tuple[str],
		fn_map: Optional[Callable[[io.BytesIO], bytes]], fmt_saida: Optional[str]
) -> None:
	# Monte o nome do diretório e o crie (se não existir)
	Path(dir_saida).mkdir(parents=True, exist_ok=True)

	# Filtre somente os nomes de arquivos com as extensões aceitas
	url_download_tipos_filtrados = [url for url in urls_download if url.lower().endswith(formatos)]

	imgs_nome = [path.basename(url_arq) for url_arq in url_download_tipos_filtrados]
	paths_saida = [f'{dir_saida}/{nome}' for _, nome in enumerate(imgs_nome)]

	semaforo = asyncio.Semaphore(1)

	async with semaforo:
		async with ClientSession(trust_env=True, timeout=400) as sessaoHttp:
			lista_params = [(url_download_tipos_filtrados[i], sessaoHttp) for i in range(len(imgs_nome))]
			arquivos = await asyncio.gather(*[baixar_arquivo(*params) for params in lista_params])

			for i, arq in enumerate(arquivos):
				path_saida = paths_saida[i] if fmt_saida is None else f"{paths_saida[i].rsplit('.', 1)[0]}.{fmt_saida}"
				with open(path_saida.replace('_mask', ''), 'wb') as handle:
					handle.write(arq if fn_map is None else fn_map(io.BytesIO(arq)))
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
