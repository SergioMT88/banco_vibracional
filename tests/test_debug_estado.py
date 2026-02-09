import numpy as np
from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal

def test_debug_estado():
    print("\n" + "="*80)
    print("DEBUG: ANÁLISE DETALHADA DO PROBLEMA DO ESTADO")
    print("="*80)

    # Carrega o cristal
    db = carregar_cristal(VibrationalDB, "data/cristal_vibracional")
    db.ensure_state()

    print(f"\n[INFO] Banco de dados:")
    print(f"  Total de conceitos: {len(db.concepts)}")
    print(f"  Dimensão de embedding: {db.embedding_dim}")
    print(f"  Número de modos espectrais: {db.n_modes}")
    print(f"  K vizinhos: {db.k_neighbors}")

    # Limpa estado
    db.state = np.zeros(len(db.concepts))

    texto_excitar = "sensações visuais"
    consulta_nao = "história da matemática"

    print(f"\n[PASSO 1] Query SEM estado")
    print("-" * 80)

    # Embedding da consulta
    q_emb = db.encode(consulta_nao)
    sims_vec_antes = db._cosine_sim_all(q_emb)
    sims_vec_antes = np.nan_to_num(sims_vec_antes, nan=0.0)

    # Top 5 SEM estado
    idx_top_antes = np.argsort(sims_vec_antes)[::-1][:5]

    print(f"Consulta: '{consulta_nao}'")
    print(f"\nSimilaridade vetorial RAW (top 5):")
    for i, idx in enumerate(idx_top_antes):
        print(f"  #{i+1} ID {idx} | sims_vec={sims_vec_antes[idx]:.6f} | texto={db.concepts[idx][:60]}...")

    print(f"\n[PASSO 2] Excitar com '{texto_excitar}'")
    print("-" * 80)

    db.excite(texto_excitar, diffusion_time=1.0, base_fraction=0.01)

    estado_info = db.state[db.state > 0.001]
    print(f"Nós com amplificação > 0.001: {len(estado_info)}")
    print(f"Amplificação máxima: {db.state.max():.6f}")
    print(f"Amplificação média (nós > 0.001): {estado_info.mean():.6f}")

    # Pegue os top IDs que foram excitados
    idx_excitados = np.argsort(db.state)[::-1][:5]
    print(f"\nTop 5 nós excitados:")
    for i, idx in enumerate(idx_excitados):
        if db.state[idx] > 0.001:
            print(f"  #{i+1} ID {idx} | state={db.state[idx]:.6f} | texto={db.concepts[idx][:60]}...")

    print(f"\n[PASSO 3] Query MESMA (não relacionada) COM estado")
    print("-" * 80)

    # Recalcula similaridade vetorial (não deve mudar!)
    sims_vec_depois = db._cosine_sim_all(q_emb)
    sims_vec_depois = np.nan_to_num(sims_vec_depois, nan=0.0)

    print(f"Consulta: '{consulta_nao}' (MESMA)")
    print(f"\nSimilaridade vetorial RAW (top 5):")
    
    idx_top_depois_vec = np.argsort(sims_vec_depois)[::-1][:5]
    for i, idx in enumerate(idx_top_depois_vec):
        print(f"  #{i+1} ID {idx} | sims_vec={sims_vec_depois[idx]:.6f} | state={db.state[idx]:.6f}")

    # Agora com query_with_state
    resultado_depois = db.query_with_state(consulta_nao, top_k=5)
    
    print(f"\nRanking FINAL (com estado via query_with_state):")
    for r in resultado_depois:
        sim_vec_raw = sims_vec_depois[r['id']]
        print(f"  ID {r['id']} | sims_vec_raw={sim_vec_raw:.6f} | state={r['state_contrib']:.6f} | sims_total={r['ressonancia_total']:.6f}")

    print(f"\n[PASSO 4] ANÁLISE DO PROBLEMA")
    print("-" * 80)

    # Para os top 5 antes
    print(f"\nNós que DEVERIAM estar no top (matematica):")
    for idx in idx_top_antes[:3]:
        sim_vec = sims_vec_depois[idx]
        state = db.state[idx]
        print(f"  ID {idx} | sims_vec_raw={sim_vec:.6f} | state={state:.6f} | contribuição ao ranking: {sim_vec:.4f}")

    # Para os nós que GANHARAM (visual)
    print(f"\nNós que GANHARAM (visual excitado):")
    for r in resultado_depois[:3]:
        sim_vec = sims_vec_depois[r['id']]
        state = r['state_contrib']
        print(f"  ID {r['id']} | sims_vec_raw={sim_vec:.6f} | state={state:.6f} | ressonancia_total={r['ressonancia_total']:.4f}")

    print(f"\n[DIAGNÓSTICO]")
    print("-" * 80)

    # Comparar
    melhor_math = sims_vec_depois[idx_top_antes[0]]
    pior_visual = resultado_depois[-1]['ressonancia_total']
    melhor_visual = resultado_depois[0]['ressonancia_total']

    print(f"\nNó matemática melhor (ID {idx_top_antes[0]}): sims_vec={melhor_math:.6f}")
    print(f"Nó visual melhor (ID {resultado_depois[0]['id']}): sims_total={melhor_visual:.6f}")
    print(f"Diferença em sims_vec: {abs(melhor_math - sims_vec_depois[resultado_depois[0]['id']]):.6f}")
    print(f"\nSe sims_vec_raw de matemática > visual, mas visual venceu,")
    print(f"então o problema está em: NORMALIZAÇÃO ou COMBINAÇÃO em query_with_state()")

    print("\n" + "="*80)
    print("FIM DEBUG")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_debug_estado()