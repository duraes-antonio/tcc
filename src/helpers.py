import asyncio
from os import path
from typing import List

from aiohttp import ClientSession


async def baixar_imagem_remota(arq_nome: str, path_saida: str, url_download: str, sessao: ClientSession):
	async with sessao.get(url_download) as resposta:
		if resposta.status != 200:
			print(resposta)

		arquivo = await resposta.read()

		with open(path.join(path_saida, arq_nome), 'wb') as handle:
			handle.write(arquivo)


async def baixar_imagens_remotas(arq_nome: List[str], path_saida: List[str], url_download: List[str]):
	async with ClientSession() as sessaoHttp:
		lista_params = [(arq_nome[i], path_saida[i], url_download[i], sessaoHttp) for i in range(len(arq_nome))]
		await asyncio.gather(*[baixar_imagem_remota(*params) for params in lista_params])
