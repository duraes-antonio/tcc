import re

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000, max_df=0.5, smooth_idf=True)


def carregar_stopword() -> [str]:
	with open('stopwords.txt', 'rt', encoding='utf-8') as arq:
		return [l.strip() for l in arq.readlines()]


def tokenizar_doc(caminho: str):
	with open(caminho, 'rt', encoding='utf-8') as arq:
		regex = re.compile('[1-9|.+\-?\n!:,;]')
		doc = regex.sub(' ', arq.read())
		stopw = carregar_stopword()
		return [plv.strip().lower()
		        for plv in doc.split(' ') if plv not in stopw]


def main():
	# TODO: abrir documento
	# TODO: TOkenizar o documento
	plvs = tokenizar_doc('texto.txt')

	X = vectorizer.fit_transform(plvs)
	svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
	svd_model.fit(X)

	terms = vectorizer.get_feature_names()

	for i, comp in enumerate(svd_model.components_):
		terms_comp = zip(terms, comp)
		sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
		print(f"Topic {i}: {', '.join([t[0] for t in sorted_terms])}")
	return 0


main()
