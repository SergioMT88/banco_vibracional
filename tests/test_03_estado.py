import numpy as np
from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal

def test_estado():
    print("\n" + "="*80)
    print("TESTE 03 — ESTADO VIBRACIONAL (VERSÃO CORRIGIDA)")
    print("="*80)

    # Carrega o cristal
    db = carregar_cristal(VibrationalDB, "data/cristal_vibracional")
    db.ensure_state()

    # Texto que será usado para excitar o cristal
    texto_excitar = "sensações visuais"

    # Consultas para medir o efeito
    consulta_rel = "percepção visual"
    consulta_nao = "história da matemática"

    print("\n[1] ANTES DE EXCITAR")
    print("-" * 80)

    # Limpa estado antes
    db.state = np.zeros(len(db.concepts))

    antes_rel = db.query_with_state(consulta_rel, top_k=5)
    antes_nao = db.query_with_state(consulta_nao, top_k=5)

    ids_antes_rel = [r['id'] for r in antes_rel]
    ids_antes_nao = [r['id'] for r in antes_nao]

    print(f"\nConsulta RELACIONADA: '{consulta_rel}'")
    print("IDs no top 5:")
    for r in antes_rel:
        print(f"  ID {r['id']} | score={r['ressonancia_total']:.4f} | state={r['state_contrib']:.4f}")

    print(f"\nConsulta NÃO RELACIONADA: '{consulta_nao}'")
    print("IDs no top 5:")
    for r in antes_nao:
        print(f"  ID {r['id']} | score={r['ressonancia_total']:.4f} | state={r['state_contrib']:.4f}")

    print("\n[2] EXCITANDO O CRISTAL")
    print("-" * 80)
    print(f"Texto de excitação: '{texto_excitar}'")
    
    db.excite(texto_excitar, diffusion_time=1.0, base_fraction=0.01)
    
    estado_info = db.state[db.state > 0.01]
    print(f"Nós com amplificação > 0.01: {len(estado_info)}")
    print(f"Amplificação máxima: {db.state.max():.4f}")
    print(f"Amplificação média (nós excitados): {estado_info.mean():.4f}")

    print("\n[3] DEPOIS DE EXCITAR")
    print("-" * 80)

    depois_rel = db.query_with_state(consulta_rel, top_k=5)
    depois_nao = db.query_with_state(consulta_nao, top_k=5)

    ids_depois_rel = [r['id'] for r in depois_rel]
    ids_depois_nao = [r['id'] for r in depois_nao]

    print(f"\nConsulta RELACIONADA: '{consulta_rel}'")
    print("IDs no top 5:")
    for r in depois_rel:
        print(f"  ID {r['id']} | score={r['ressonancia_total']:.4f} | state={r['state_contrib']:.4f}")

    print(f"\nConsulta NÃO RELACIONADA: '{consulta_nao}'")
    print("IDs no top 5:")
    for r in depois_nao:
        print(f"  ID {r['id']} | score={r['ressonancia_total']:.4f} | state={r['state_contrib']:.4f}")

    print("\n[4] ANÁLISE COMPARATIVA")
    print("-" * 80)

    # Overlap de IDs
    overlap_rel = len(set(ids_antes_rel) & set(ids_depois_rel))
    overlap_nao = len(set(ids_antes_nao) & set(ids_depois_nao))

    print(f"\nConsulta RELACIONADA:")
    print(f"  Overlap de IDs: {overlap_rel}/5 ({overlap_rel*20:.0f}%)")
    print(f"  IDs que mudaram: {set(ids_antes_rel) ^ set(ids_depois_rel)}")

    print(f"\nConsulta NÃO RELACIONADA:")
    print(f"  Overlap de IDs: {overlap_nao}/5 ({overlap_nao*20:.0f}%)")
    print(f"  IDs que mudaram: {set(ids_antes_nao) ^ set(ids_depois_nao)}")

    # Mudança de scores
    print(f"\nMudança de scores (consulta relacionada):")
    for i, id_no in enumerate(ids_antes_rel):
        score_antes = antes_rel[i]['ressonancia_total']
        if id_no in ids_depois_rel:
            idx_depois = ids_depois_rel.index(id_no)
            score_depois = depois_rel[idx_depois]['ressonancia_total']
            delta = score_depois - score_antes
            delta_pct = (delta / score_antes * 100) if score_antes != 0 else 0
            print(f"  ID {id_no}: {score_antes:.4f} → {score_depois:.4f} ({delta:+.4f}, {delta_pct:+.1f}%)")
        else:
            print(f"  ID {id_no}: SAIU do top 5")

    print(f"\nMudança de scores (consulta não relacionada):")
    for i, id_no in enumerate(ids_antes_nao):
        score_antes = antes_nao[i]['ressonancia_total']
        if id_no in ids_depois_nao:
            idx_depois = ids_depois_nao.index(id_no)
            score_depois = depois_nao[idx_depois]['ressonancia_total']
            delta = score_depois - score_antes
            delta_pct = (delta / score_antes * 100) if score_antes != 0 else 0
            print(f"  ID {id_no}: {score_antes:.4f} → {score_depois:.4f} ({delta:+.4f}, {delta_pct:+.1f}%)")
        else:
            print(f"  ID {id_no}: SAIU do top 5")

    print("\n[5] DIAGNÓSTICO")
    print("-" * 80)

    if overlap_rel == 5 and overlap_nao == 5:
        print("✅ SELETIVIDADE PERFEITA")
        print("   Estado está sendo proporcional e não contamina resultados")
    elif overlap_rel >= 4:
        print("⚠️  SELETIVIDADE FUNCIONAL")
        print("   Estado afeta levemente, mas ordem permanece similar")
    else:
        print("❌ SELETIVIDADE FRACA")
        print("   Estado está reordenando resultados indevidamente")

    print("\n" + "="*80)
    print("FIM TESTE 03")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_estado()