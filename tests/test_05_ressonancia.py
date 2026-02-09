from src.banco_vibracional import VibrationalDB

def test_ressonancia():
    print("\n=== TESTE 05 — RESSONÂNCIA HÍBRIDA ===")

    db = VibrationalDB.load("data/cristal_vibracional")

    consulta = "como reflexos simples evoluem para consciência"
    resultados = db.query_with_state(consulta)

    for r in resultados:
        print("\nID:", r["id"])
        print("sim_vec:", r["sim_vec"])
        print("sim_spec:", r["sim_spec"])
        print("state_contrib:", r["state_contrib"])
        print("ressonancia_total:", r["ressonancia_total"])
        print("texto:", r["texto"][:200])

    print("\n=== FIM TESTE 05 ===")


if __name__ == "__main__":
    test_ressonancia()