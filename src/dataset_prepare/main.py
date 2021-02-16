import shutil
from os import path
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set

import cv2
from PIL import Image as PILImage

import arquivo_util as au
import cli_util as cli
import imagem_util as iu


def gerar_lista_arquivos(
		dir_treino: str, dir_val: str, dir_saida: str = None,
		porcent_treino=0.7, porcent_val=0.2, porcent_teste=0.1,
		dataset_pneumonia=True, formato_VOC=False
):
	def criar_arq_lista_nomes(nome_arq_saida: str, nomes_arq: List[str]):
		Path(dir_saida).mkdir(parents=True, exist_ok=True)

		with open(nome_arq_saida, 'w') as arq:
			ult_indice = len(nomes_arq) - 1
			arq.writelines([nome + '\n' if i != ult_indice else nome for i, nome in enumerate(nomes_arq)])

	dir_saida = dir_saida or path.join(path.dirname(dir_treino), 'ImageSets')
	dic: Dict[str, Set[str]] = au.gerar_lista_validacao_treino(
		dir_treino, dir_val, restringir_somente_treino=True
	)
	arqs_nome = dic['treino']
	fatia_nome_porcent = {'train': porcent_treino, 'val': porcent_val, 'test': porcent_teste}
	porcent_inic = 0

	if (dataset_pneumonia):
		arq_nome_bacteria = sorted({nome for nome in arqs_nome if nome.endswith('bacterial')})
		arq_nome_covid = sorted({nome for nome in arqs_nome if nome.endswith('covid')})
		arq_nome_virus = sorted({nome for nome in arqs_nome if nome.endswith('viral')})

		for nome_subset in fatia_nome_porcent:
			porcent_fim = porcent_inic + fatia_nome_porcent[nome_subset]
			arqs_bacteria = arq_nome_bacteria[
			                int(len(arq_nome_bacteria) * porcent_inic):int(len(arq_nome_bacteria) * porcent_fim)]
			arqs_covid = arq_nome_covid[int(len(arq_nome_covid) * porcent_inic):int(len(arq_nome_covid) * porcent_fim)]
			arqs_virus = arq_nome_virus[int(len(arq_nome_virus) * porcent_inic):int(len(arq_nome_virus) * porcent_fim)]
			arqs_nome_salvar = [*arqs_bacteria, *arqs_covid, *arqs_virus]
			criar_arq_lista_nomes(f'{dir_saida}/{nome_subset}.txt', arqs_nome_salvar)
			porcent_inic += fatia_nome_porcent[nome_subset]
	else:
		for nome_subset in fatia_nome_porcent:
			porcent_fim = porcent_inic + fatia_nome_porcent[nome_subset]
			arqs_virus = sorted(arqs_nome)[int(len(arqs_nome) * porcent_inic):int(len(arqs_nome) * porcent_fim)]
			criar_arq_lista_nomes(f'{dir_saida}/{nome_subset}.txt', arqs_virus)
			porcent_inic += fatia_nome_porcent[nome_subset]

	if formato_VOC:
		linhas = []
		nomes_sets = []
		for nome_subset in fatia_nome_porcent:
			if fatia_nome_porcent[nome_subset] == 0.0:
				continue
			nomes_sets.append(nome_subset)

			with open(f'{dir_saida}/{nome_subset}.txt', 'r') as arq:
				linhas += [l if l.endswith('\n') else f'{l}\n' for l in arq.readlines()]

		with open(f'{dir_saida}/trainval.txt', 'w') as arq:
			arq.writelines(linhas)


def padronizar_imgs(
		arquivos_path: Iterable[str],
		cores_indice: Optional[Dict[Tuple[int, int, int], Tuple[int, int, int]]] = None,
		tamanho: Optional[Iterable[int]] = None,
		formato: Optional[str] = None, dir_saida: Optional[str] = None,
		equalizar=False, aplicar_transf_morf=False
):
	l = len(list(arquivos_path))
	cli.print_barra_prog(0, l)

	if dir_saida:
		Path(dir_saida).mkdir(exist_ok=True, parents=True)

	for i, arq_path in enumerate(arquivos_path):
		img = PILImage.open(arq_path).convert('RGB')
		formato = formato or ('png' if arq_path.endswith('.png') else 'jpeg')

		if cores_indice is not None:
			img = iu.substituir_cores_por_indice(img, cores_indice)

		if tamanho is not None:
			img = iu.redimensionar(img, tuple(tamanho))

		arq_novo_nome = path.basename(arq_path).rsplit('.', 1)[0] + f'.{formato.lower()}'

		if cores_indice is not None:
			with open(path.join(dir_saida or path.dirname(arq_path), arq_novo_nome), 'wb') as saida:
				saida.write(iu.converter_img_para_bytes(img.convert('L'), formato))

		else:
			if equalizar:
				img_open_cv = iu.imagepil_para_opencv_hist_equal(img)
			elif aplicar_transf_morf:
				img_open_cv = iu.aplicar_transf_morf(iu.imagepil_para_opencv(img))
			else:
				img_open_cv = iu.imagepil_para_opencv(img)
			cv2.imwrite(path.join(dir_saida or path.dirname(arq_path), arq_novo_nome), img_open_cv)

		cli.print_barra_prog(i + 1, l)


def particionar_dados(
		imgs_path: str, anots_path: str, dir_saida: str = None,
		img_ext='jpeg', anot_ext='png'
):
	arqs_sets_path = path.join(path.dirname(anots_path), 'ImageSets')
	nomes_arq_sets = ['test', 'train', 'val']

	for nome_set in nomes_arq_sets:
		Path(path.join(dir_saida or imgs_path, nome_set)).mkdir()
		Path(path.join(dir_saida or anots_path, f'{nome_set}_gt')).mkdir()

	def copiar_img_anot(nome_set: str, nome_arq: str):
		img_nome = f'{nome_arq}.{img_ext}'
		anot_nome = f'{nome_arq}.{anot_ext}'
		shutil.copyfile(
			path.join(imgs_path, img_nome),
			path.join(f'{imgs_path}/{nome_set}', img_nome)
		)
		shutil.copyfile(
			path.join(anots_path, anot_nome),
			path.join(f'{anots_path}/{nome_set}_gt', anot_nome)
		)

	for nome_arq_set in nomes_arq_sets:
		with open(path.join(arqs_sets_path, f'{nome_arq_set}.txt'), 'r') as arq_set:
			linhas_nome_arq = arq_set.readlines()
			qtd_arqs = len(linhas_nome_arq)
			cli.print_msg(f'Copiando e organizando - Dataset {nome_arq_set}')
			cli.print_barra_prog(0, qtd_arqs)

			for i, arq_nome in enumerate(linhas_nome_arq):
				copiar_img_anot(nome_arq_set, arq_nome.strip())
				cli.print_barra_prog(i + 1, qtd_arqs)


def padronizar_dir_completo(
		dir_imgs: str, dir_anots: str,
		dir_saida_imgs: str = None, dir_saida_anots: str = None,
		tamanho: Iterable[int] = None,
		cor_indice: Dict[Tuple[int, int, int], Tuple[int, int, int]] = None,
		img_ext='jpeg', anot_ext='png', equalizar=False, transf_morf=False
):
	msg_conversao_cor = 'Conversão de cor' if cor_indice else ''
	msg_tamanho = 'Padronização de tamanho' if tamanho else ''
	msgs = ' e '.join([msg for msg in [msg_conversao_cor, msg_tamanho] if msg.strip()])
	cli.print_msg(f'{msgs}: Iniciada!')

	anotacoes_path = au.obter_path_arquivos(dir_anots)
	imagens_path = au.obter_path_arquivos(dir_imgs)

	dir_saida_anots = dir_saida_anots or path.dirname(dir_anots)
	dir_saida_imgs = dir_saida_imgs or path.dirname(dir_imgs)

	padronizar_imgs(anotacoes_path, cor_indice, tamanho, anot_ext, dir_saida=dir_saida_anots)
	padronizar_imgs(
		imagens_path, tamanho=tamanho, formato=img_ext,
		dir_saida=dir_saida_imgs, equalizar=equalizar, aplicar_transf_morf=transf_morf
	)
	cli.print_msg(f'{msgs}: Finalizada!')


def padronizar_particionar_imgs(
		dir_imgs: str, dir_anots: str, dir_sets_txt: str,
		dir_raiz_saida: str, tamanho: Iterable[int] = None,
		cor_indice: Dict[Tuple[int, int, int], Tuple[int, int, int]] = None,
		img_ext='jpeg', anot_ext='png', equalizar=False, transf_morf=False
):
	msg_conversao_cor = 'Conversão de cor' if cor_indice else ''
	msg_tamanho = 'Padronização de tamanho' if tamanho else ''
	msgs = ' e '.join([msg for msg in [msg_conversao_cor, msg_tamanho] if msg.strip()])
	cli.print_msg(f'{msgs}: Iniciada!')
	particoes = ['test', 'train', 'val']

	for part in particoes:
		arqs_nome: Set[str] = set()

		with open(path.join(dir_sets_txt, f'{part}.txt'), 'r') as set_txt:
			arqs_nome = {linha.strip() for linha in set_txt.readlines()}

		anotacoes_path = [
			path.join(dir_anots, path.basename(arq_nome))
			for arq_nome in au.obter_path_arquivos(dir_anots)
			if path.basename(arq_nome).rsplit('.', 1)[0] in arqs_nome
		]
		imagens_path = [
			path.join(dir_imgs, path.basename(arq_nome))
			for arq_nome in au.obter_path_arquivos(dir_imgs)
			if path.basename(arq_nome).rsplit('.', 1)[0] in arqs_nome
		]

		dir_saida_anots = path.join(dir_raiz_saida, f'{part}_gt') or path.dirname(dir_anots)
		dir_saida_imgs = path.join(dir_raiz_saida, part) or path.dirname(dir_imgs)

		cli.print_msg(f'Padronizando imagens - Dataset {part}')
		padronizar_imgs(anotacoes_path, cor_indice, tamanho, anot_ext, dir_saida=dir_saida_anots)
		padronizar_imgs(
			imagens_path, tamanho=tamanho, formato=img_ext, dir_saida=dir_saida_imgs,
			equalizar=equalizar, aplicar_transf_morf=transf_morf
		)

	cli.print_msg(f'{msgs}: Finalizada!')


def padronizar_particionar_imgs_VOC(
		dir_imgs: str, dir_anots: str, dir_sets_txt: str,
		dir_raiz_saida: str, tamanho: Iterable[int] = None,
		cor_indice: Dict[Tuple[int, int, int], Tuple[int, int, int]] = None,
		img_ext='jpeg', anot_ext='png', equalizar=False, transf_morf=False
):
	msg_conversao_cor = 'Conversão de cor' if cor_indice else ''
	msg_tamanho = 'Padronização de tamanho' if tamanho else ''
	msgs = ' e '.join([msg for msg in [msg_conversao_cor, msg_tamanho] if msg.strip()])
	cli.print_msg(f'{msgs}: Iniciada!')
	particoes = ['test', 'train', 'val']
	dir_saida_anot = 'SegmentationClass'
	dir_saida_img = 'JPEGImages'

	for part in particoes:
		with open(path.join(dir_sets_txt, f'{part}.txt'), 'r') as set_txt:
			arqs_nome = {linha.strip() for linha in set_txt.readlines()}

		anotacoes_path = [
			path.join(dir_anots, path.basename(arq_nome))
			for arq_nome in au.obter_path_arquivos(dir_anots)
			if path.basename(arq_nome).rsplit('.', 1)[0] in arqs_nome
		]
		imagens_path = [
			path.join(dir_imgs, path.basename(arq_nome))
			for arq_nome in au.obter_path_arquivos(dir_imgs)
			if path.basename(arq_nome).rsplit('.', 1)[0] in arqs_nome
		]
		dir_saida_anots = path.join(dir_raiz_saida, dir_saida_anot) or path.dirname(dir_anots)
		dir_saida_imgs = path.join(dir_raiz_saida, dir_saida_img) or path.dirname(dir_imgs)

		cli.print_msg(f'Padronizando imagens - Dataset {part}')
		padronizar_imgs(anotacoes_path, cor_indice, tamanho, anot_ext, dir_saida=dir_saida_anots)
		padronizar_imgs(
			imagens_path, tamanho=tamanho, formato=img_ext, dir_saida=dir_saida_imgs,
			equalizar=equalizar, aplicar_transf_morf=transf_morf
		)

	cli.print_msg(f'{msgs}: Finalizada!')


def gerar_nome_diretorio_saida(
		tamanho: Iterable[int], porc_treino: float, porc_val: float,
		porc_teste: float, nome_base: str, equalizar_hist=False,
		transf_morf=False, voc=False, cor_indice=True, exibir_causa=True
) -> str:
	nome_partes = [
		nome_base or '',
		'x'.join([str(t) for t in tamanho]),
		str(int(porc_treino * 100)),
		str(int(porc_val * 100)),
		str(int(porc_teste * 100)),
		'hist-equal' if equalizar_hist else '',
		'transf-morf' if transf_morf else '',
		'VOC' if voc else '',
		'' if cor_indice else 'cor',
		'causa' if exibir_causa else '',
	]
	return '_'.join([parte for parte in nome_partes if parte])


def remover_causa_imagesets(dir_imagesets: str, sep='_'):
	import os
	_path = path.join(dir_imagesets, 'ImageSets')
	arqs_nome = [path.join(_path, arq) for arq in os.listdir(_path)]

	for nome_set in arqs_nome:
		linhas = []
		with open(nome_set, 'r') as arq_leitura:
			linhas = arq_leitura.readlines()

		with open(nome_set, 'w') as arq_escrita:
			linhas_novas = [l.split(sep)[0] + '\n' for l in linhas]
			arq_escrita.writelines(linhas_novas)


def remover_causa_imagens(dir_imagens: str, sep='_'):
	imagens_path = au.obter_path_arquivos(dir_imagens)
	for p in imagens_path:
		dir = path.dirname(p)
		arq_nome_ext = path.basename(p)
		arq_nome = arq_nome_ext.rsplit('.', 1)[0].split(sep)[0]
		ext = arq_nome_ext.rsplit('.', 1)[1]
		au.renomear_arquivo(p, path.join(dir, f'{arq_nome}.{ext}'))


def main():
	args = cli.ler_args()
	print(args.cor_indice)
	cor_indice = {
		(1, 1, 1): (0, 0, 0),
		# covid
		(0, 192, 128): (1, 1, 1),
		# bacterial
		(128, 192, 128): (2, 2, 2),
		# viral
		(128, 64, 128): (3, 3, 3)
	}

	padronizar = args.cor_indice or args.tamanho
	padronizar_particionar = args.particionar and padronizar

	dir_saida_raiz = path.dirname(args.dir_anotacoes if args.dir_anotacoes[-1] != '/' else args.dir_anotacoes[:-1])
	dir_saida = gerar_nome_diretorio_saida(
		args.tamanho, args.porcent_treino, args.porcent_val,
		args.porcent_teste, dir_saida_raiz, args.equalizar_histograma,
		args.transformacao_morfologica, args.voc, args.cor_indice,
		args.causa
	)

	if padronizar_particionar:
		gerar_lista_arquivos(
			args.dir_anotacoes, args.dir_imagens, path.join(dir_saida, 'ImageSets'),
			porcent_treino=args.porcent_treino, porcent_val=args.porcent_val,
			porcent_teste=args.porcent_teste, formato_VOC=True
		)
		if (args.voc):
			padronizar_particionar_imgs_VOC(
				args.dir_imagens, args.dir_anotacoes, path.join(dir_saida, 'ImageSets'),
				dir_saida, args.tamanho, cor_indice if args.cor_indice else None, args.imagem_formato,
				args.anotacao_formato, args.equalizar_histograma, args.transformacao_morfologica
			)
		else:
			padronizar_particionar_imgs(
				args.dir_imagens, args.dir_anotacoes, path.join(dir_saida, 'ImageSets'),
				dir_saida, args.tamanho, cor_indice if args.cor_indice else None, args.imagem_formato,
				args.anotacao_formato, args.equalizar_histograma, args.transformacao_morfologica
			)

	elif args.particionar:
		gerar_lista_arquivos(
			args.dir_anotacoes, args.dir_imagens, porcent_treino=args.porcent_treino,
			porcent_val=args.porcent_val, porcent_teste=args.porcent_teste
		)
		particionar_dados(
			args.dir_imagens, args.dir_anotacoes, dir_saida,
			args.imagem_formato, args.anotacao_formato
		)

	elif padronizar:
		gerar_lista_arquivos(
			args.dir_anotacoes, args.dir_imagens, porcent_treino=args.porcent_treino,
			porcent_val=args.porcent_val, porcent_teste=args.porcent_teste
		)
		padronizar_dir_completo(
			args.dir_imagens, args.dir_anotacoes,
			path.join(dir_saida, path.basename(args.dir_imagens)),
			path.join(dir_saida, path.basename(args.dir_anotacoes)),
			args.tamanho, cor_indice, args.imagem_formato, args.anotacao_formato,
			args.equalizar_histograma, args.transformacao_morfologica
		)

	if not args.causa:
		remover_causa_imagesets(dir_saida)
		remover_causa_imagens(dir_saida)

	return 0


main()
