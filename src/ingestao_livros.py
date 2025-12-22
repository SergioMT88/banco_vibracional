import os
import requests


def carregar_ids_gutenberg(path="data/gutenberg_ids.txt"):
    """
    Lê um arquivo com IDs do Project Gutenberg, um por linha.
    Ignora linhas vazias e comentários (#).
    """
    if not os.path.exists(path):
        print(f"[GUTENBERG] Arquivo {path} não encontrado. Crie e adicione IDs.")
        return []

    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha or linha.startswith("#"):
                continue
            try:
                gid = int(linha.split()[0])
                ids.append(gid)
            except ValueError:
                print(f"[GUTENBERG] Linha inválida em {path}: {linha}")

    print(f"[GUTENBERG] {len(ids)} IDs carregados de {path}")
    return ids


def _tentar_baixar_gutenberg_txt(gutenberg_id: int) -> str | None:
    """
    Tenta baixar o texto bruto de um livro do Project Gutenberg
    usando alguns padrões comuns de URL.
    Retorna o texto ou None em caso de falha.
    """

    # Padrões comuns de URL de texto no Gutenberg
    padroes = [
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}.txt",
        f"https://www.gutenberg.org/files/{gutenberg_id}/{gutenberg_id}-0.txt",
        f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt",
    ]

    for url in padroes:
        try:
            print(f"[GUTENBERG] Tentando baixar {url}")
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200 and len(resp.text) > 1000:
                print(f"[GUTENBERG] Download bem-sucedido para ID {gutenberg_id}")
                return resp.text
        except Exception as e:
            print(f"[GUTENBERG] Erro ao acessar {url}: {e}")

    print(f"[GUTENBERG] Não foi possível baixar o livro ID {gutenberg_id}")
    return None


def _limpar_texto_gutenberg(raw: str) -> str:
    """
    Remove cabeçalhos e rodapés padrão do Project Gutenberg.
    Usa marcadores *** START OF e *** END OF quando disponíveis.
    """

    texto = raw

    # Tentar remover cabeçalho
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
    ]
    start_idx = -1
    for m in start_markers:
        start_idx = texto.find(m)
        if start_idx != -1:
            break

    if start_idx != -1:
        # pula a linha do marcador
        texto = texto[start_idx:]
        texto = texto.split("\n", 1)[-1]

    # Tentar remover rodapé
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_idx = -1
    for m in end_markers:
        end_idx = texto.find(m)
        if end_idx != -1:
            break

    if end_idx != -1:
        texto = texto[:end_idx]

    return texto.strip()


def _dividir_em_trechos(texto: str, min_chars: int = 300):
    """
    Divide o texto em parágrafos / blocos,
    filtrando trechos muito curtos.
    """
    # separa por linhas em branco (parágrafos)
    blocos_brutos = texto.split("\n\n")
    trechos = []

    for b in blocos_brutos:
        t = " ".join(l.strip() for l in b.splitlines())
        t = " ".join(t.split())  # normaliza espaços
        if len(t) >= min_chars:
            trechos.append(t)

    return trechos


def add_from_gutenberg(db, gutenberg_id: int):
    """
    Baixa um livro do Project Gutenberg, limpa o texto,
    divide em trechos e adiciona ao banco vibracional.
    """

    raw = _tentar_baixar_gutenberg_txt(gutenberg_id)
    if raw is None:
        return

    texto_limpo = _limpar_texto_gutenberg(raw)
    trechos = _dividir_em_trechos(texto_limpo, min_chars=300)

    print(f"\n[GUTENBERG] Extraindo {len(trechos)} trechos do livro ID {gutenberg_id}")

    for trecho in trechos:
        cid = db.add_concept(trecho)
        print(f"  [+] [{cid}] {trecho[:120]}...")