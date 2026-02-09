import sys
import os
import numpy as np

# Garante que a raiz do projeto esteja no path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal

# --- CONFIGURAÇÃO DA LLM ---
# Se tiver uma chave OpenAI, descomente e configure.
# Caso contrário, usaremos um "Simulador de LLM" que mostra o prompt gerado.
USE_REAL_LLM = False

def chamar_llm(prompt_sistema, prompt_usuario):
    """
    Função que envia o contexto para a LLM gerar a resposta final.
    """
    if USE_REAL_LLM:
        # Exemplo com OpenAI (requer pip install openai)
        # import openai
        # client = openai.Client(api_key="SUA_CHAVE")
        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": prompt_sistema},
        #         {"role": "user", "content": prompt_usuario}
        #     ]
        # )
        # return response.choices[0].message.content
        pass
    else:
        # SIMULAÇÃO: Mostra o que seria enviado para a LLM
        print("\n" + "="*20 + " PROMPT GERADO PARA LLM " + "="*20)
        print(f"[SYSTEM]\n{prompt_sistema}\n")
        print(f"[USER]\n{prompt_usuario}")
        print("="*65)
        return ">> (Aqui a LLM responderia baseada no contexto acima) <<"

def chat_llm_loop():
    print("Carregando Cérebro Vibracional...")
    dados_path = os.path.join(os.path.dirname(__file__), "..", "data", "cristal_vibracional")
    
    try:
        db = carregar_cristal(VibrationalDB, dados_path)
        db.ensure_state()
    except FileNotFoundError:
        print("Erro: Banco não encontrado. Rode a ingestão primeiro.")
        return

    print("\n" + "="*60)
    print("CHAT RAG VIBRACIONAL + LLM")
    print("="*60)
    print("O sistema recupera memórias relevantes e monta o prompt.")

    historico_conversa = []

    while True:
        try:
            texto = input("\nVocê: ").strip()
        except KeyboardInterrupt:
            break

        if not texto or texto.lower() in ["sair", "exit"]:
            break
            
        if texto.lower() == "/reset":
            db.state = np.zeros_like(db.state)
            historico_conversa = []
            print("[!] Memória limpa.")
            continue

        # 1. RECUPERAÇÃO VIBRACIONAL
        # Usa o estado para desambiguar a busca
        resultados = db.query_with_state(texto, top_k=3, gamma=1.5)
        
        # Filtra resultados com score muito baixo (ruído)
        contextos_validos = [r for r in resultados if r['ressonancia_total'] > 0.8]
        
        if not contextos_validos:
            # Se nada vibrar, tenta busca vetorial pura (fallback)
            resultados = db.query(texto, top_k=2, alpha=1.0)
            contextos_validos = resultados

        # 2. MONTAGEM DO CONTEXTO
        bloco_contexto = ""
        print(f"\n[RAG] Recuperado {len(contextos_validos)} memórias:")
        for i, r in enumerate(contextos_validos):
            score = r.get('ressonancia_total', r.get('similarity', 0.0))
            print(f"  {i+1}. {r['texto'][:80]}... (Score: {score:.2f})")
            bloco_contexto += f"- {r['texto']}\n"

        # 3. CONSTRUÇÃO DO PROMPT
        prompt_sistema = (
            "Você é um assistente inteligente com acesso a uma memória vibracional.\n"
            "Use APENAS o contexto abaixo para responder à pergunta do usuário.\n"
            "Se o contexto não for suficiente, diga que não sabe.\n\n"
            "CONTEXTO RECUPERADO:\n"
            f"{bloco_contexto}"
        )

        # 4. CHAMADA LLM
        resposta = chamar_llm(prompt_sistema, texto)
        
        if USE_REAL_LLM:
            print(f"\nBot: {resposta}")
        
        # 5. ATUALIZAÇÃO DO ESTADO (MEMÓRIA)
        # O que o usuário disse passa a fazer parte do contexto mental
        db.excite(texto, diffusion_time=0.5)
        
        # Decaimento natural (agora funciona de verdade!)
        db.decay_state(rate=0.5)
        
        historico_conversa.append(f"User: {texto}")

if __name__ == "__main__":
    chat_llm_loop()