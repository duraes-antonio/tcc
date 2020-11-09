from typing import Dict, Tuple

anot_bacteria_cor = (0, 192, 128)
anot_covid_cor = (128, 192, 128)
anot_virus_cor = (128, 64, 128)

causa_cor: Dict[str, Tuple[int, int, int]] = {
	'bacterial': anot_bacteria_cor,
	'covid': anot_covid_cor,
	'viral': anot_virus_cor
}
