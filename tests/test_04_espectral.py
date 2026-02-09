from src.banco_vibracional import VibrationalDB
import numpy as np

def test_espectral():
    print("\n=== TESTE 04 — COERÊNCIA ESPECTRAL ===")

    db = VibrationalDB.load("data/cristal_vibracional")

    consulta = "reflexos de sapo"
    q_emb = db.encode(consulta)
    sims_vec = db._cosine_sim_all(q_emb)
    sims_spec = db._spectral_similarity(sims_vec)

    idx = np.argsort(sims_spec)[::-1][:5]

    print("Top IDs espectrais:", idx)
    print("Top sims_spec:", sims_spec[idx])

    print("\nTextos:")
    for i in idx:
        print(f"  {i}: {db.concepts[i][:120]}...")

    print("\n=== FIM TESTE 04 ===")


if __name__ == "__main__":
    test_espectral()