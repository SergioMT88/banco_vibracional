import re


def segmentar_texto(texto: str, min_chars: int = 200):
	"""
	Segmenta um texto em trechos úteis.

	Estratégia:
	1) Divide por linhas em branco (parágrafos)
	2) Se o parágrafo for grande, divide em sentenças
	3) Filtra trechos muito curtos
	"""
	if not texto:
		return []

	blocos = re.split(r"\n\s*\n", texto)
	trechos = []

	for b in blocos:
		t = " ".join(l.strip() for l in b.splitlines())
		t = " ".join(t.split())
		if len(t) >= min_chars:
			trechos.append(t)
		else:
			frases = re.split(r"(?<=[.!?])\s+", t)
			for f in frases:
				f = f.strip()
				if len(f) >= min_chars:
					trechos.append(f)

	return trechos
