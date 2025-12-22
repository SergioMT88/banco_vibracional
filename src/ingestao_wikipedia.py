import wikipedia

# Configura o idioma da Wikipedia
wikipedia.set_lang("pt")


def carregar_links_wikipedia(path="data/wikipedia_links.txt"):
    """
    Lê um arquivo contendo links da Wikipedia, um por linha.
    Retorna uma lista de URLs limpos.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            links = [linha.strip() for linha in f.readlines() if linha.strip()]
        print(f"[WIKI] {len(links)} links carregados de {path}")
        return links
    except FileNotFoundError:
        print(f"[WIKI] Arquivo {path} não encontrado. Crie o arquivo e adicione links.")
        return []


def add_from_wikipedia(db, url):
    """
    Acessa uma página da Wikipedia, extrai o resumo e adiciona frases como conceitos.
    """
    try:
        page = wikipedia.page(url=url)
        resumo = page.summary

        # Divide o resumo em frases úteis
        frases = [f.strip() for f in resumo.split(".") if len(f.strip()) > 20]

        print(f"\n[WIKI] Extraindo {len(frases)} conceitos da página: {page.title}")

        for frase in frases:
            cid = db.add_concept(frase)
            print(f"  [+] [{cid}] {frase}")

    except Exception as e:
        print(f"[WIKI] Erro ao acessar {url}: {e}")