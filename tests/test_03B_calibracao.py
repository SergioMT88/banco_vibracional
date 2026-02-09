import numpy as np
from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal

def medir_separabilidade(db, consulta_rel, consulta_nao, top_k=5):
    """Mede a separabilidade vibracional entre consultas relacionadas e não relacionadas."""
    rel = db.query_with_state(consulta_rel, top_k=top_k)
    nao = db.query_with_state(consulta_nao, top_k=top_k)

    media_rel = np.mean([r["state_contrib"] for r in rel])
    media_nao = np.mean([r["state_contrib"] for r in nao])

    return media_rel - media_nao, media_rel, media_nao


def medir_estabilidade(db, consulta, top_k=5):
    """Mede quanto o ranking mudou após excitação."""
    antes = db.query_with_state(consulta, top_k=top_k)
    ids_antes = set([r["id"] for r in antes])

    depois = db.query_with_state(consulta, top_k=top_k)
    ids_depois = set([r["id"] for r in depois])

    overlap = len(ids_antes & ids_depois) / top_k
    return overlap


def test_calibracao():
    print("\n=== TESTE 03B — CALIBRAÇÃO AUTOMÁTICA ===")

    # Carrega o cristal
    db = carregar_cristal(VibrationalDB, "data/cristal_vibracional")
    db.ensure_state()

    # Consultas de teste
    consulta_rel = "sensações visuais"
    consulta_nao = "história da matemática"

    # Excitação inicial
    db.excite("sensações visuais", diffusion_time=0.01, base_fraction=0.002)

    # Espaço de busca
    gammas = [0.3, 0.5, 0.7, 0.85, 0.9]
    amps = [1.0, 2.0, 3.0, 5.0]
    diffs = [0.001, 0.01, 0.05]
    bases = [0.001, 0.002, 0.005]

    melhor_score = -999
    melhor_cfg = None

    for gamma in gammas:
        for amp in amps:
            for diff in diffs:
                for base in bases:

                    # Reset do estado
                    db.state = np.zeros_like(db.state)

                    # Nova excitação
                    db.excite("sensações visuais", diffusion_time=diff, base_fraction=base)

                    # Ajuste temporário do método
                    db.gamma = gamma
                    db.amp = amp

                    # Métricas
                    separabilidade, media_rel, media_nao = medir_separabilidade(db, consulta_rel, consulta_nao)
                    estabilidade_rel = medir_estabilidade(db, consulta_rel)
                    estabilidade_nao = medir_estabilidade(db, consulta_nao)

                    # Score composto
                    score = (
                        separabilidade * 2.0 +
                        estabilidade_nao * 1.0 -
                        abs(estabilidade_rel - 0.5) * 0.5
                    )

                    if score > melhor_score:
                        melhor_score = score
                        melhor_cfg = {
                            "gamma": gamma,
                            "amplificacao": amp,
                            "diffusion_time": diff,
                            "base_fraction": base,
                            "separabilidade": separabilidade,
                            "media_rel": media_rel,
                            "media_nao": media_nao,
                            "estabilidade_rel": estabilidade_rel,
                            "estabilidade_nao": estabilidade_nao,
                            "score": score
                        }

    print("\n=== MELHOR COMBINAÇÃO ENCONTRADA ===")
    for k, v in melhor_cfg.items():
        print(f"{k}: {v}")

    print("\n=== FIM TESTE 03B ===")


if __name__ == "__main__":
    test_calibracao()