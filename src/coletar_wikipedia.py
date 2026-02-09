import os
import requests

API_URL = "https://pt.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json",
    "Accept-Language": "pt-BR,pt;q=0.9"
}

ARTIGOS_IA = [
    "Inteligência artificial",
    "Aprendizado de máquina",
    "Rede neural artificial",
    "Aprendizado profundo",
    "Rede neural convolucional",
    "Rede neural recorrente",
    "Transformer (modelo de linguagem)",
    "Retropropagação",
    "Gradiente descendente",
    "Sobreajuste",
    "Regularização (matemática)",
    "Processamento de linguagem natural",
    "Visão computacional",
    "Aprendizado por reforço",
    "Rede adversária generativa",
    "Autoencoder",
    "Mecanismo de atenção",
    "Long short-term memory",
    "Máquina de vetores de suporte",
    "K-means"
]

ARTIGOS_NATUREZA = [
    "Pesca",
    "Rede de pesca",
    "Oceano",
    "Biologia marinha",
    "Ecossistema",
    "Recife de coral",
    "Biodiversidade",
    "Aquicultura",
    "Peixe",
    "Mar"
]

def baixar_artigo(titulo):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": titulo,
        "format": "json",
        "redirects": 1
    }

    r = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)

    # Se a resposta não for JSON, retorna vazio
    try:
        data = r.json()
    except:
        print("[ERRO] Resposta inválida da Wikipedia (provável bloqueio).")
        print("Conteúdo recebido:", r.text[:200])
        return ""

    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")

def salvar_texto(pasta, titulo, texto):
    os.makedirs(pasta, exist_ok=True)
    nome = titulo.replace(" ", "_").replace("/", "_") + ".txt"
    caminho = os.path.join(pasta, nome)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"[OK] Salvo:", caminho)

def coletar_wikipedia(pasta_saida):
    # Combina os tópicos para criar ambiguidade (Rede Neural vs Rede de Pesca)
    todos_artigos = ARTIGOS_IA + ARTIGOS_NATUREZA
    for titulo in todos_artigos:
        print(f"[INFO] Baixando:", titulo)
        texto = baixar_artigo(titulo)
        if texto.strip():
            salvar_texto(pasta_saida, titulo, texto)
        else:
            print(f"[ERRO] Artigo vazio ou bloqueado:", titulo)

if __name__ == "__main__":
    coletar_wikipedia("data/ia_wikipedia")