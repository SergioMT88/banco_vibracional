import os
from src.banco_vibracional import VibrationalDB
from src.cristal import salvar_cristal, carregar_cristal

CRISTAL_PATH = "data/cristal_vibracional"


def cristal_existe():
    required = [
        "conceitos.json",
        "embeddings.npy",
        "grafo.pkl",
        "modos.npy",
        "eigenvalues.npy",
        "meta.json",
    ]
    return all(os.path.exists(os.path.join(CRISTAL_PATH, f)) for f in required)


def criar_cristal():
    print("\n=== Criando novo cristal vibracional ===")

    db = VibrationalDB()

    # Aqui você adiciona seus textos iniciais
    # Exemplo:
    # db.add_concept("texto 1")
    # db.add_concept("texto 2")
    # ...
    # Ou usa seu pipeline de ingestão

    print("Construindo grafo...")
    db.build_graph()

    print("Calculando Laplaciana...")
    db.compute_laplacian()

    print("Calculando modos espectrais...")
    db.compute_spectral_modes()

    salvar_cristal(db, CRISTAL_PATH)

    print("\n=== Cristal criado com sucesso ===")
    return db


def carregar_cristal_existente():
    print("\n=== Carregando cristal existente ===")
    db = carregar_cristal(VibrationalDB, CRISTAL_PATH)
    print("=== Cristal carregado ===")
    return db


def testar_fluxo_vibracional(db):
    print("\n=== Teste de ressonância ===")

    # 1) Excita o cristal com uma ideia profunda
    estimulo = "how does consciousness emerge from simple animal reflexes"
    print(f"\n> Excitando o cristal com: {estimulo}")
    #db.excite(estimulo, strength=0.25, diffusion_time=0.4, base_fraction=0.05)
    db.excite(estimulo, diffusion_time=0.4, base_fraction=0.01)


    # 2) Consulta contextual
    consulta = "learning habits in animals and machines"
    print(f"\n> Consulta contextual: {consulta}")
    resultados = db.query_with_state(consulta, top_k=5, alpha=0.5, gamma=0.15)

    for r in resultados:
        print("\nID:", r["id"])
        #print("Texto:", r["text"][:200], "...")
        print("Texto:", r["texto"][:200], "...")
        print("Sim vetorial:", r["sim_vec"])
        print("Sim espectral:", r["sim_spec"])
        print("Estado contrib:", r["state_contrib"])
        #print("Ressonância total:", r["similarity"])
        print("Ressonância total:", r["ressonancia_total"])


    # 3) Decaimento
    print("\n> Decaindo estado vibracional...")
    db.decay_state(rate=0.1)


if __name__ == "__main__":
    if cristal_existe():
        db = carregar_cristal_existente()
    else:
        db = criar_cristal()

    testar_fluxo_vibracional(db)