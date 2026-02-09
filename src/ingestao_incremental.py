import os
from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal
from src.utils import segmentar_texto

def carregar_textos(pasta):
    textos = []
    if not os.path.isdir(pasta):
        print(f"[ERRO] Pasta não encontrada: {pasta}")
        return textos
    for nome in os.listdir(pasta):
        if nome.endswith(".txt"):
            caminho = os.path.join(pasta, nome)
            try:
                with open(caminho, "r", encoding="utf-8") as f:
                    textos.append(f.read())
            except Exception as e:
                print(f"[ERRO] Falha ao ler {caminho}: {e}")
    return textos

def ingestao_incremental(pasta_textos, pasta_cristal):
    print("Carregando cristal existente...")
    banco = carregar_cristal(VibrationalDB, pasta_cristal)

    print("Carregando novos textos...")
    textos = carregar_textos(pasta_textos)

    if not textos:
        print("Nenhum arquivo .txt encontrado. Encerrando.")
        return

    print(f"{len(textos)} arquivos encontrados.")

    novos_conceitos = []
    for texto in textos:
        segmentos = segmentar_texto(texto)
        novos_conceitos.extend(segmentos)

    if not novos_conceitos:
        print("Nenhum conceito novo após segmentação. Encerrando.")
        return

    print(f"Segmentação concluída: {len(novos_conceitos)} novos conceitos.")

    print("Gerando embeddings...")
    novos_embeddings = banco.encode_lote(novos_conceitos)

    print("Atualizando banco...")
    banco.adicionar_conceitos(novos_conceitos, novos_embeddings)

    print("Recalculando espectro...")
    banco.recalcular_espectro()

    print("Salvando cristal atualizado...")
    banco.salvar(pasta_cristal)

    print("Ingestão concluída com sucesso!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python -m src.ingestao_incremental <pasta_textos> <pasta_cristal>")
        exit(1)

    ingestao_incremental(sys.argv[1], sys.argv[2])