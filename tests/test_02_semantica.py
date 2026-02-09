import numpy as np
from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal

CONSULTAS = [
    "o que é fotossíntese",
    "como funciona um motor elétrico",
    "o que é memória RAM",
]

def test_semantica():
    print("\n=== TESTE 02 — COERÊNCIA SEMÂNTICA ===")

    # Carrega o cristal usando o carregador real
    db = carregar_cristal(VibrationalDB, "data/cristal_vibracional")

    for consulta in CONSULTAS:
        print(f"\n--- Consulta: {consulta} ---")

        # Embedding da consulta
        q_emb = db.encoder.encode(consulta)

        # Similaridade vetorial com todos os conceitos
        sims = db._cosine_sim_all(q_emb)

        # Top 5
        idx = np.argsort(sims)[::-1][:5]
        top = sims[idx]

        print("Top IDs:", idx)
        print("Top sims:", top)
        print("Média sim_vec:", top.mean())

        print("\nTextos dos conceitos:")
        for i in idx:
            print(f"  {i}: {db.concepts[i][:150]}...")

    print("\n=== FIM TESTE 02 ===")


if __name__ == "__main__":
    test_semantica()