import asyncio
import json
from io import BytesIO
from os import path, remove
from typing import Dict, Set, Optional, List, Tuple, Coroutine

import compartilhado.constantes as const
import util.arquivo_util as au
import util.imagem_util as iu


async def baixar_anotacoes(
		ids: List[int], dir_salvar: str, cor: Tuple[int, int, int],
		formato='png'
):
	github_repo_link = 'https://raw.githubusercontent.com/v7labs/covid-19-xray-dataset'
	url_github = f'{github_repo_link}/master/annotations/all-images-semantic-masks'
	urls_download = [f'{url_github}/{id}.png' for id in ids]

	def substituir_cor_branca(img: BytesIO) -> bytes:
		return iu.substituir_cor(img, (255, 255, 255), cor, True, formato)

	await au.baixar_arquivos(urls_download, dir_salvar, fn_map=substituir_cor_branca)
	return


async def baixar_imagens(urls_download: List[str], dir_salvar: str):
	await au.baixar_arquivos(urls_download, dir_salvar)


def extrair_id_url_json(
		path_json_causa: str, labels_ignorar: Optional[Set[int]] = None
) -> Dict[int, str]:
	with open(path_json_causa, 'r') as json_file:
		imgs_dados = [
			img['dataset_image']['image'] for img in json.load(json_file)['items']
			if labels_ignorar is None
			   or not any(label in set(img['labels']) for label in labels_ignorar)
		]
		return {img_dados['id']: img_dados['url'] for img_dados in imgs_dados}


def extrair_id_nome_json(
		path_json_causa: str, labels_ignorar: Optional[Set[int]] = None
) -> Dict[int, str]:
	with open(path_json_causa, 'r') as json_file:
		imgs_dados = [
			img['dataset_image']['image'] for img in json.load(json_file)['items']
			if labels_ignorar is None
			   or not any(label in set(img['labels']) for label in labels_ignorar)
		]
		return {img_dados['id']: path.basename(img_dados['key']).rsplit('.', 1)[0] for img_dados in imgs_dados}


def extrair_ids_arquivos_json(
		path_json_causa: str, labels_ignorar: Optional[Set[int]] = None
) -> Set[int]:
	with open(path_json_causa, 'r') as json_file:
		imgs_dados = [
			img['dataset_image']['image'] for img in json.load(json_file)['items']
			if labels_ignorar is None
			   or not any(label in set(img['labels']) for label in labels_ignorar)
		]
		return {img_dados['id'] for img_dados in imgs_dados}


def renomear_imgs_id(dir_base='../../dataset/v7labs/'):
	anot_bacteria_cor = extrair_id_nome_json('datasets/v7labs/json_causas/bacteria.json')
	anot_covid_cor = extrair_id_nome_json('datasets/v7labs/json_causas/covid.json')
	anot_virus_cor = extrair_id_nome_json('datasets/v7labs/json_causas/virus.json', {8190})

	for dic in [anot_bacteria_cor, anot_covid_cor, anot_virus_cor]:
		for id in dic:
			try:
				extensao = dic[id].rsplit('.', 1)[1]
				au.renomear_arquivo(path.join(dir_base, dic[id]), path.join(dir_base, f'{id}.{extensao}'))
			except:
				print('Err')


def obter_diferenca_anotacoes_imagens() -> Set[str]:
	dir = '../../dataset/v7labs/'
	anots = {path.basename(a.rsplit('.', 1)[0]) for a in au.obter_path_arquivos(dir + 'anots')}
	imgs = {path.basename(a.rsplit('.', 1)[0]) for a in au.obter_path_arquivos(dir + 'imgs')}
	return anots.difference(imgs)


def apagar_anotacoes_imagens_ausentes():
	dir = '../../dataset/v7labs/'
	anots = {path.basename(a.rsplit('.', 1)[0]): a for a in au.obter_path_arquivos(dir + 'anots')}
	imgs = {path.basename(a.rsplit('.', 1)[0]): a for a in au.obter_path_arquivos(dir + 'imgs')}
	arq_ausente_nomes = {nome for nome in anots}.difference({nome for nome in imgs})
	for nome_arq in arq_ausente_nomes:
		remove(anots[nome_arq])


async def baixar_dataset(dir_anots: Optional[str] = None, dir_imgs: Optional[str] = None):
	# tags_id_nome = {3241: 'viral', 3243: 'bacterial', 8190: 'covid'}
	cor_dic_id_url = {
		const.anot_bacteria_cor: extrair_id_url_json('datasets/v7labs/json_causas/covid.json'),
		const.anot_covid_cor: extrair_id_url_json('datasets/v7labs/json_causas/bacteria.json'),
		const.anot_virus_cor: extrair_id_url_json('datasets/v7labs/json_causas/virus.json', {8190})
	}
	tarefas: List[Coroutine] = []

	if (dir_anots):
		anots_nome = {
			int(path.basename(arq_path).rsplit('.', 1)[0])
			for arq_path in au.obter_path_arquivos(dir_anots)
		}
		tarefas += [
			baixar_anotacoes(
				[id for id in cor_dic_id_url[cor_rgb] if id not in anots_nome],
				path.join(dir_anots, str(cor_rgb)), cor_rgb
			)
			for cor_rgb in cor_dic_id_url
		]

	if (dir_imgs):
		imgs_nome = {
			int(path.basename(arq_path).rsplit('.', 1)[0])
			for arq_path in au.obter_path_arquivos(dir_imgs)
		}
		tarefas += [
			baixar_imagens(
				[dic_id_url[id] for id in dic_id_url if id not in imgs_nome],
				dir_imgs
			)
			for dic_id_url in list(cor_dic_id_url.values())
		]

	return await asyncio.gather(*tarefas)


def adicionar_causa_nome_arquivo(
		dir_anots: Optional[str] = None, dir_imgs: Optional[str] = None
):
	anot_bacteria_cor = {
		id: 'bacterial' for id in
		extrair_ids_arquivos_json('v7labs/json_causas/bacteria.json')
	}
	anot_covid_cor = {
		id: 'covid' for id in
		extrair_ids_arquivos_json('v7labs/json_causas/covid.json')
	}
	anot_virus_cor = {
		id: 'viral' for id in
		extrair_ids_arquivos_json('v7labs/json_causas/virus.json', {8190})
	}
	ids_causas = {**anot_bacteria_cor, **anot_covid_cor, **anot_virus_cor}
	anots_path = au.obter_path_arquivos(dir_anots)
	imgs_path = au.obter_path_arquivos(dir_imgs)

	def renomear_com_causa(ids_causa: Dict[int, str], arqs_path: Set[str]):
		for anot_path in arqs_path:
			arq_nome_ext = path.basename(anot_path)
			arq_ext = arq_nome_ext.rsplit('.', 1)[1]
			arq_nome = arq_nome_ext.rsplit('.', 1)[0]
			au.renomear_arquivo(
				anot_path,
				path.join(path.dirname(anot_path), f'{arq_nome}_{ids_causa[int(arq_nome)]}.{arq_ext}')
			)

	renomear_com_causa(ids_causas, anots_path)
	renomear_com_causa(ids_causas, imgs_path)


v7_dir = '../../dataset/v7labs/'
# asyncio.get_event_loop().run_until_complete(baixar_dataset(f'{v7_dir}/anot', None))
adicionar_causa_nome_arquivo('../../dataset/custom/SegmentationClass', '../../dataset/custom/JPEGImages')
