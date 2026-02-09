import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.banco_vibracional import VibrationalDB

def demo_seletividade():
    print("\n" + "="*60)
    print("DEMO: ROBUSTEZ E SELETIVIDADE (O 'NÃO' DO SISTEMA)")
    print("="*60)
    print("Objetivo: Provar que o contexto NÃO contamina assuntos não relacionados.")
    
    # 1. Setup do Banco em Memória
    print("\n[1] Inicializando Banco...")
    db = VibrationalDB(n_modes=2, k_neighbors=2)
    
    # Conceitos: Natureza (A), Tech (B), Culinária (C)
    concepts = [
        "O pescador usou a rede no mar",           # A
        "Peixes nadam nos recifes de coral",       # A
        "O servidor de rede caiu ontem",           # B
        "Latência da rede wi-fi está alta",        # B
        "Receita de bolo de cenoura com chocolate",# C
        "Como cozinhar macarrão al dente",         # C
    ]
    db.adicionar_conceitos(concepts)
    db.recalcular_espectro()
    print(f"    Banco populado com {len(concepts)} conceitos.")
    
    # 2. Baseline: Query sobre Culinária (sem contexto)
    query_unrelated = "bolo de cenoura"
    print(f"\n[2] Query Baseline: '{query_unrelated}'")
    res_base = db.query_with_state(query_unrelated, top_k=1)
    print(f"    Resultado: {res_base[0]['texto']}")
    print(f"    Score: {res_base[0]['ressonancia_total']:.4f}")
    
    # 3. Excitar com Tech (contexto forte, mas irrelevante para bolo)
    contexto = "servidores e internet"
    print(f"\n[3] Excitando cristal com: '{contexto}'")
    db.excite(contexto, diffusion_time=0.5)
    
    # 4. Query novamente com o contexto Tech ativo
    print(f"\n[4] Query com Contexto Ativo: '{query_unrelated}'")
    print(f"    (Esperamos que o sistema IGNORE o contexto de Tech para esta pergunta)")
    res_context = db.query_with_state(query_unrelated, top_k=1)
    print(f"    Resultado: {res_context[0]['texto']}")
    print(f"    Score: {res_context[0]['ressonancia_total']:.4f}")
    print(f"    Contribuição do Estado: {res_context[0]['state_contrib']:.4f}")
    
    # 5. Validação
    id_base = res_base[0]['id']
    id_context = res_context[0]['id']
    
    print("\n" + "-"*60)
    if id_base == id_context:
        print("RESULTADO: SUCESSO ✅")
        print("O sistema manteve o foco no 'Bolo' e ignorou o ruído de 'Tech'.")
        print("Isso prova que a Ressonância Seletiva funciona.")
    else:
        print("RESULTADO: FALHA ❌")
        print("O contexto contaminou a busca.")
    print("="*60 + "\n")

if __name__ == "__main__":
    demo_seletividade()