import sys
import os
import numpy as np

# Garante que a raiz do projeto esteja no path para imports funcionarem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal

def chat_loop():
    print("Carregando cristal vibracional...")
    
    # Caminho onde os dados ingeridos são salvos
    dados_path = os.path.join(os.path.dirname(__file__), "..", "data", "cristal_vibracional")
    
    try:
        db = carregar_cristal(VibrationalDB, dados_path)
        db.ensure_state()
        print(f"Cristal carregado! {len(db.concepts)} conceitos na memória.")
    except FileNotFoundError:
        print(f"\n[AVISO] Não foi encontrado um cristal salvo em: {dados_path}")
        print("Para usar o chat, primeiro popule o banco rodando:")
        print("  1. python -m src.coletar_wikipedia")
        print("  2. python -m src.ingestao_incremental data/ia_wikipedia data/cristal_vibracional")
        return

    print("\n" + "="*60)
    print("CHAT VIBRACIONAL - MEMÓRIA DE CURTO PRAZO ATIVA")
    print("="*60)
    print("O sistema lembrará do contexto das mensagens anteriores.")
    print("Comandos:")
    print("  /debug  - Ver o que o sistema está 'pensando' (estado ativo)")
    print("  /plot   - Visualizar o grafo atual")
    print("  /reset  - Esquecer o contexto (limpar estado)")
    print("  /sair   - Encerrar")
    print("-" * 60)

    while True:
        try:
            texto = input("\nVocê: ").strip()
        except KeyboardInterrupt:
            break

        if not texto:
            continue
            
        if texto.lower() in ["/sair", "sair", "exit"]:
            break
            
        if texto.lower() == "/reset":
            if db.state is not None:
                db.state = np.zeros_like(db.state)
            print("[!] Memória de curto prazo limpa.")
            continue
            
        if texto.lower() == "/debug":
            if db.state is None or np.all(db.state == 0):
                print("[!] Estado vibracional neutro.")
            else:
                # Mostra os 5 conceitos mais ativos no "pensamento" do sistema
                idx = np.argsort(db.state)[::-1][:5]
                print("\n[Estado Vibracional - Top Ativações]")
                for i in idx:
                    val = db.state[i]
                    if abs(val) > 0.01:
                        print(f"  {val:.4f} | {db.concepts[i][:60]}...")
            continue

        if texto.lower() == "/plot":
            try:
                from src.visualizacao import plotar_grafo
                print("Gerando visualização do estado mental...")
                plotar_grafo(db, titulo="Estado Atual do Chat")
            except Exception as e:
                print(f"Erro ao plotar: {e}")
            continue

        # --- FLUXO VIBRACIONAL ---
        
        # 1. Consulta (influenciada pelo estado anterior)
        resultados = db.query_with_state(texto, top_k=1, gamma=2.0)
        
        if resultados:
            top = resultados[0]
            print(f"Bot: {top['texto']}")
            print(f"     (Ressonância: {top['ressonancia_total']:.2f} | Contexto: {top['state_contrib']:.2f})")
        else:
            print("Bot: ... (sem conhecimento suficiente)")

        # 2. Excitação (atualiza o estado com o novo input)
        # O input do usuário "vibra" no grafo e altera o contexto para a próxima rodada
        db.excite(texto, diffusion_time=0.5)
        
        # 3. Decaimento (esquecimento natural)
        # A cada interação, a memória antiga enfraquece um pouco (15%)
        db.decay_state(rate=0.30)

if __name__ == "__main__":
    chat_loop()