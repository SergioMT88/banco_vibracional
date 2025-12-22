import os
import json
import numpy as np
import pickle


def salvar_cristal(db, pasta, salvar_estado=True):
    os.makedirs(pasta, exist_ok=True)

    # conceitos
    with open(os.path.join(pasta, "conceitos.json"), "w", encoding="utf-8") as f:
        json.dump(db.concepts, f, ensure_ascii=False, indent=2)

    # embeddings
    np.save(os.path.join(pasta, "embeddings.npy"), db.embeddings)

    # grafo
    with open(os.path.join(pasta, "grafo.pkl"), "wb") as f:
        pickle.dump(db.graph, f)

    # modos e autovalores
    np.save(os.path.join(pasta, "modos.npy"), db.modes)
    np.save(os.path.join(pasta, "eigenvalues.npy"), db.eigenvalues)

    # normalização opcional
    if db.mean is not None:
        np.save(os.path.join(pasta, "mean.npy"), db.mean)
    if db.std is not None:
        np.save(os.path.join(pasta, "std.npy"), db.std)

    # estado vibracional
    if salvar_estado and db.state is not None:
        np.save(os.path.join(pasta, "state.npy"), db.state)

    # metadados
    meta = {
        "embedding_dim": db.embedding_dim,
        "n_modes": db.n_modes,
        "k_neighbors": db.k_neighbors,
        "beta": db.beta,
        "num_concepts": len(db.concepts),
    }

    with open(os.path.join(pasta, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def carregar_cristal(DBClass, pasta):
    # metadados
    with open(os.path.join(pasta, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    db = DBClass(
        embedding_dim=meta["embedding_dim"],
        n_modes=meta["n_modes"],
        k_neighbors=meta["k_neighbors"],
        beta=meta["beta"],
    )

    # conceitos
    with open(os.path.join(pasta, "conceitos.json"), "r", encoding="utf-8") as f:
        db.concepts = json.load(f)

    # embeddings
    db.embeddings = np.load(os.path.join(pasta, "embeddings.npy"))

    # grafo
    with open(os.path.join(pasta, "grafo.pkl"), "rb") as f:
        db.graph = pickle.load(f)

    # modos
    db.modes = np.load(os.path.join(pasta, "modos.npy"))
    db.eigenvalues = np.load(os.path.join(pasta, "eigenvalues.npy"))

    # normalização
    mean_path = os.path.join(pasta, "mean.npy")
    std_path = os.path.join(pasta, "std.npy")

    db.mean = np.load(mean_path) if os.path.exists(mean_path) else None
    db.std = np.load(std_path) if os.path.exists(std_path) else None

    # estado vibracional
    state_path = os.path.join(pasta, "state.npy")
    if os.path.exists(state_path):
        db.state = np.load(state_path)
    else:
        db.ensure_state()

    return db