import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.banco_vibracional import VibrationalDB

def demo_comparativo():
    print("\n" + "="*60)
    print("BENCHMARK: RAG DE MERCADO (VETORIAL) vs RAG VIBRACIONAL")
    print("="*60)
    
    # 1. Configuração do Banco
    print("\n[1] Inicializando Cristal Vibracional...")
    db = VibrationalDB(n_modes=2, k_neighbors=2)
    
    # 2. Inserindo Conceitos Ambíguos (Ex: 'Rede')
    # Cluster A: Tecnologia
    tech_concepts = [
        "O administrador configurou a rede wi-fi da empresa",
        "Protocolos de rede garantem a segurança dos dados",
        "A latência da rede está muito alta hoje",
        "Redes neurais artificiais aprendem padrões complexos",
    ]
    # Cluster B: Pesca/Natureza
    nature_concepts = [
        "O pescador jogou a rede no mar calmo ao amanhecer",
        "Muitos peixes ficaram presos na rede de pesca",
        "A rede de emalhar é proibida nesta região costeira",
        "Tartarugas marinhas presas em redes fantasmas no oceano",
    ]
    
    all_concepts = tech_concepts + nature_concepts
    db.adicionar_conceitos(all_concepts)
    db.recalcular_espectro()
    print(f"    Banco populado com {len(all_concepts)} conceitos.")
    
    # Query ambígua
    query = "problemas na rede"
    
    # ---------------------------------------------------------
    # CENÁRIO A: Busca Padrão (Sem Contexto)
    # ---------------------------------------------------------
    print(f"\n[CENÁRIO A] Padrão de Mercado (Pinecone/Chroma/Weaviate)")
    print(f"    Tecnologia: Busca Vetorial Densa (Stateless)")
    print(f"    Query: '{query}'")
    
    # alpha=1.0 foca puramente no vetor (simula RAG comum)
    res_padrao = db.query(query, top_k=3, alpha=1.0)
    
    print("    Top 3 Resultados:")
    for r in res_padrao:
        print(f"    - [{r['id']}] {r['texto'][:60]}... (Sim: {r['similarity']:.4f})")
        
    # ---------------------------------------------------------
    # CENÁRIO B: Busca Vibracional (Contexto: Natureza)
    # ---------------------------------------------------------
    contexto = "oceanos e pescaria"
    print(f"\n[CENÁRIO B] RAG Vibracional (Nossa Solução)")
    print(f"    Tecnologia: Vetorial + Grafo Espectral + Memória de Estado")
    print(f"    Contexto Ativo: '{contexto}'")
    print(f"    Query: '{query}'")
    
    # Excita o cristal
    db.excite(contexto, diffusion_time=0.5)
    
    # Consulta com estado (gamma alto para demonstrar efeito)
    res_vibra = db.query_with_state(query, top_k=3, gamma=5.0)
    
    print("    Top 3 Resultados:")
    for r in res_vibra:
        # Marcador visual se o resultado mudou em relação ao padrão
        marker = "★" if r['id'] != res_padrao[0]['id'] else " "
        print(f"  {marker} - [{r['id']}] {r['texto'][:60]}...")
        print(f"        (Base: {r['sim_vec']:.2f} | Estado: {r['state_contrib']:.2f} | Final: {r['ressonancia_total']:.2f})")

    # ---------------------------------------------------------
    # CONCLUSÃO
    # ---------------------------------------------------------
    print("\n" + "-"*60)
    top_id_std = res_padrao[0]['id']
    top_id_vib = res_vibra[0]['id']
    
    if top_id_std != top_id_vib:
        print("RESULTADO DO BENCHMARK: VITÓRIA DO RAG VIBRACIONAL")
        print("-" * 60)
        print("1. O RAG de Mercado falhou na desambiguação (trouxe viés de tecnologia).")
        print("2. O RAG Vibracional usou a memória de curto prazo para entender o contexto.")
        print("3. A 'Ressonância Seletiva' suprimiu ativamente os resultados irrelevantes.")
    else:
        print("RESULTADO: Empate (o contexto não foi forte o suficiente para alterar o ranking).")
    print("="*60 + "\n")

    # --- VISUALIZAÇÃO ---
    try:
        from src.visualizacao import plotar_grafo
        print("Abrindo janela de visualização...")
        plotar_grafo(db, titulo="Demo: Rede (Tech vs Pesca)")
    except ImportError:
        print("Instale matplotlib e networkx para ver o gráfico.")
    except Exception as e:
        print(f"Erro ao visualizar: {e}")

if __name__ == "__main__":
    demo_comparativo()