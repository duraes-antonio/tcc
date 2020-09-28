import asyncio
from pathlib import Path
from typing import List, Dict

import pandas as pd
from pandas import DataFrame

from helpers import baixar_imagens_remotas


def filtrar_por_causas(df: DataFrame, col='finding', bacteria=False, virus=False, fungo=False) -> DataFrame:
	opcoes: dict = {'Bacteria': bacteria, 'Viral': virus, 'Fungal': fungo}

	# Concatenação usada para filtrar por nenhum ou múltiplas opções
	condicao = '|'.join([chave for chave in opcoes if opcoes[chave]])
	return df[df[col].str.contains(condicao)]


def filtrar_por_causa(df: DataFrame, causa: str, col='finding') -> DataFrame:
	return df[df[col].str.contains(causa)]


def capturar_causas(df: DataFrame, col='finding', bacteria=False, virus=False, fungo=False) -> List[str]:
	return sorted(filtrar_por_causas(df, col, bacteria, virus, fungo)[col].unique())


async def atualizar_imagens(
		df_metadados: DataFrame, col_causa='finding', col_img_nome='filename',
		fungo=False, bacteria=False, virus=False
):
	url_github = 'https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/'

	opcoes_causas: Dict[str, bool] = {'Bacterial': bacteria, 'Viral': virus, 'Fungal': fungo}
	co_rotinas = []
	formatos_aceitos = ('png', 'jpg', 'jpeg')

	for causa in opcoes_causas:

		if not opcoes_causas[causa]:
			continue

		# filtre as linhas pela causa atual
		dados_filtrados: DataFrame = filtrar_por_causa(df_metadados, causa, col_causa)

		# obtenha somente o nome das imagens
		imgs_nome = [img_nome for img_nome in dados_filtrados[col_img_nome].unique()
					 if img_nome.lower().endswith(formatos_aceitos)]

		# Monte o nome do diretório e o crie (se não existir)
		dir_nome = f'../dataset/{causa.lower()}'
		Path(dir_nome).mkdir(parents=True, exist_ok=True)

		paths_saida = [dir_nome for i in range(len(imgs_nome))]
		urls_down = [f'{url_github}/{nome}' for nome in imgs_nome]

		co_rotinas.append(baixar_imagens_remotas(imgs_nome, paths_saida, urls_down))

	await asyncio.gather(*co_rotinas)


def main():
	# Abra o arquivo com os metadados
	df = pd.read_csv('../dataset/metadata.csv', delimiter=',')

	# co_rotina = atualizar_imagens(df, fungo=True, bacteria=True, virus=True)
	# asyncio.get_event_loop().run_until_complete(co_rotina)
	return 0


main()
