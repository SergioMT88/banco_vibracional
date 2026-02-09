import numpy as np
from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal


def test_integridade():
    print("\n=== TESTE 01 — INTEGRIDADE ESTRUTURAL ===")

    # Carrega o cristal usando o carregador REAL do seu projeto
    db = carregar_cristal(VibrationalDB, "data/cristal_vibracional")

    # 1. Embeddings
    print("\n[1] Embeddings")
    assert db.embeddings is not None
    print("Shape:", db.embeddings.shape)

       # 2. Grafo
    print("\n[2] Grafo")
    assert db.graph is not None
    print("Tipo:", type(db.graph))

    # Para matrizes esparsas:
    print("Shape:", db.graph.shape)
    print("Número de arestas (nnz):", db.graph.getnnz())
    
    # 3. Modos e autovalores
    print("\n[3] Espectro")
    assert db.modes is not None
    assert db.eigenvalues is not None
    print("Modos shape:", db.modes.shape)
    print("Autovalores shape:", db.eigenvalues.shape)
    print("Menor autovalor:", db.eigenvalues.min())

    # 4. Estado vibracional
    print("\n[4] Estado vibracional")
    db.ensure_state()
    print("Shape:", db.state.shape)

    print("\n=== FIM TESTE 01 ===")


if __name__ == "__main__":
    test_integridade()