import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class VibrationalDB:
    """
    Cristal vibracional:
    - memória: conceitos + embeddings
    - estrutura: grafo + laplaciana + modos
    - consulta: ressonância vetorial + espectral
    - estado: ativação vibracional contínua
    - plasticidade futura: expansão e reorganização
    """

    def __init__(
        self,
        embedding_dim=384,
        n_modes=8,
        k_neighbors=5,
        beta=0.5,
        encoder_model_name="paraphrase-multilingual-MiniLM-L12-v2",
    ):
        # --- hiperparâmetros ---
        self.embedding_dim = embedding_dim
        self.n_modes = n_modes
        self.k_neighbors = k_neighbors
        self.beta = beta

        # --- encoder semântico ---
        self.encoder = SentenceTransformer(encoder_model_name)

        # --- memória básica ---
        self.concepts = []           # lista de textos
        self.embeddings = None       # matriz (N x D)

        # --- estrutura de grafo / espectral ---
        self.graph = None            # matriz de adjacência W (sparse)
        self.laplacian = None        # L = D - W
        self.modes = None            # autovetores (N x n_modes)
        self.eigenvalues = None      # autovalores (n_modes,)

        # --- normalização global opcional ---
        self.mean = None
        self.std = None

        # --- estado vibracional (ativação dos nós) ---
        self.state = None            # vetor (N,)

    # =========================================================
    # PERCEPÇÃO / REPRESENTAÇÃO
    # =========================================================
    def encode(self, text: str):
        """
        Codifica um texto em embedding.
        """
        emb = self.encoder.encode(text)
        emb = np.asarray(emb, dtype=np.float32)
        return emb

    def add_concept(self, text: str):
        """
        Adiciona um novo conceito ao banco.
        """
        emb = self.encode(text)

        if self.embeddings is None:
            self.embeddings = emb.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

        self.concepts.append(text)

        # atualiza (ou inicializa) estado vibracional
        self.ensure_state()

        return len(self.concepts) - 1

    def encode_lote(self, textos: list[str]):
        """
        Gera embeddings em lote para uma lista de textos.
        """
        if textos is None or len(textos) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        embs = self.encoder.encode(textos)
        embs = np.asarray(embs, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs

    def adicionar_conceitos(self, textos: list[str], embeddings: np.ndarray | None = None):
        """
        Adiciona conceitos em lote, com embeddings pré-computados opcionais.
        """
        if textos is None or len(textos) == 0:
            return

        if embeddings is None:
            embeddings = self.encode_lote(textos)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        self.concepts.extend(textos)
        self.ensure_state()

    def recalcular_espectro(self):
        """
        Recalcula grafo, Laplaciana e modos espectrais após atualização do banco.
        """
        self.build_graph()
        self.compute_laplacian()
        self.compute_spectral_modes()

    def salvar(self, pasta: str, salvar_estado: bool = True):
        """
        Salva o cristal vibracional em disco.
        """
        from src.cristal import salvar_cristal

        salvar_cristal(self, pasta, salvar_estado=salvar_estado)

    # =========================================================
    # ESTRUTURA: GRAFO / LAPLACIANA / MODOS ESPECTRAIS
    # =========================================================
    def build_graph(self):
        """
        Constrói o grafo de similaridade entre embeddings usando KNN.
        """
        if self.embeddings is None or len(self.concepts) < 2:
            raise ValueError("Poucos conceitos para construir o grafo.")

        X = self.embeddings
        n = X.shape[0]

        # normaliza embeddings para produto interno virar coseno
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        Xn = X / norms

        nn = NearestNeighbors(
            n_neighbors=min(self.k_neighbors + 1, n),
            metric="cosine",
        )
        nn.fit(Xn)
        dist, idx = nn.kneighbors(Xn)

        rows, cols, data = [], [], []

        for i in range(n):
            for j, d in zip(idx[i], dist[i]):
                if i == j:
                    continue
                sim = 1.0 - float(d)
                if sim <= 0:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(sim)

        W = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
        W = 0.5 * (W + W.T)  # simetriza

        self.graph = W.tocsr()

        # garante consistência do estado
        self.ensure_state()

    def compute_laplacian(self):
        """
        Calcula a Laplaciana do grafo.
        """
        if self.graph is None:
            raise ValueError("Grafo não foi construído.")

        W = self.graph.tocsr()
        degrees = np.array(W.sum(axis=1)).ravel()
        D = sp.diags(degrees)
        self.laplacian = D - W

    def compute_spectral_modes(self):
        """
        Calcula os modos espectrais (autovetores de menor autovalor da Laplaciana).
        """
        if self.laplacian is None:
            raise ValueError("Laplaciana não foi calculada.")

        n = self.laplacian.shape[0]
        k = min(self.n_modes + 1, n)

        vals, vecs = spla.eigsh(self.laplacian, k=k, which="SM")

        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]

        # descarta o modo trivial (autovalor 0)
        self.eigenvalues = vals[1 : self.n_modes + 1]
        self.modes = vecs[:, 1 : self.n_modes + 1]

        # garante consistência do estado
        self.ensure_state()

    # =========================================================
    # INTERNAL: SIMILARIDADE VETORIAL
    # =========================================================
    def _cosine_sim_all(self, q_emb):
        """
        Similaridade de coseno entre query e todos os embeddings.
        """
        X = self.embeddings
        X_norm = np.linalg.norm(X, axis=1) + 1e-8
        q_norm = np.linalg.norm(q_emb) + 1e-8
        return (X @ q_emb) / (X_norm * q_norm)

    # =========================================================
    # CONSULTA INSTANTÂNEA (SEM ESTADO)
    # =========================================================
    def query(self, text: str, top_k: int = 5, alpha: float = 0.5):
        """
        Consulta robusta combinando:
        - similaridade vetorial (coseno)
        - similaridade espectral (suavizada no grafo)
        
        alpha:
        - 1.0 => só vetorial
        - 0.0 => só espectral
        """
        if self.embeddings is None or len(self.concepts) == 0:
            raise ValueError("Banco vibracional vazio.")

        if not isinstance(text, str) or len(text.strip()) == 0:
            text = " "

        n = len(self.concepts)
        k = max(1, min(top_k, n))

        # 1) Embedding da consulta
        try:
            q_emb = self.encode(text)
        except Exception:
            q_emb = np.zeros(self.embedding_dim, dtype=np.float32)

        # 2) Similaridade vetorial
        sims_vec = self._cosine_sim_all(q_emb)
        sims_vec = np.nan_to_num(sims_vec, nan=0.0)

        # 3) Similaridade espectral (suavizada)
        sims_spec = self._spectral_similarity(sims_vec)

        # 4) Normalização por Escala (Max) para preservar proporções
        def normalize_scale(arr):
            # Clip em zero para ignorar similaridades negativas (irrelevantes)
            arr_clipped = np.maximum(arr, 0.0)
            a_max = arr_clipped.max()
            if a_max < 1e-8:
                return np.zeros_like(arr, dtype=np.float32) + 0.01
            # Normaliza pelo máximo, mantendo a proporção relativa ao zero
            # Floor de 0.2 para permitir que o contexto resgate itens distantes
            return (arr_clipped / a_max * 0.8 + 0.2).astype(np.float32)

        sims_vec_norm = normalize_scale(sims_vec)
        sims_spec_norm = normalize_scale(sims_spec)

        # 5) Combinação híbrida
        sims_total = alpha * sims_vec_norm + (1.0 - alpha) * sims_spec_norm
        sims_total = np.nan_to_num(sims_total, nan=0.0)

        # 6) Ordenação e resultados
        idx = np.argsort(sims_total)[::-1][:k]

        resultados = []
        for i in idx:
            resultados.append(
                {
                    "id": int(i),
                    "texto": self.concepts[i],
                    "sim_vec": float(sims_vec[i]),
                    "sim_spec": float(sims_spec[i]),
                    "similarity": float(sims_total[i]),
                }
            )

        return resultados

    # =========================================================
    # ESTADO VIBRACIONAL
    # =========================================================
    def ensure_state(self):
        """
        Garante que o vetor de estado vibracional exista e tenha o tamanho correto.
        """
        n = len(self.concepts)
        if n == 0:
            self.state = None
            return
        if self.state is None or len(self.state) != n:
            self.state = np.zeros(n, dtype=np.float32)

    def excite(self, text: str, diffusion_time: float = 0.01, base_fraction: float = 0.001):
        """
        Emite um pulso unitário que se propaga radialmente pelo grafo.
        
        MUDANÇA: Substitui estado ao invés de acumular indefinidamente
        """

        if self.embeddings is None or len(self.concepts) == 0:
            raise ValueError("Banco vibracional vazio.")

        self.ensure_state()

        # 1) Embedding do estímulo
        q_emb = self.encode(text)

        # 2) Similaridade vetorial
        sims_vec = self._cosine_sim_all(q_emb)
        sims_vec = np.nan_to_num(sims_vec, nan=0.0)

        n = len(self.concepts)
        m = max(1, int(base_fraction * n))

        # 3) Seleciona os nós mais próximos como fonte inicial
        base_idx = np.argsort(sims_vec)[::-1][:m]

        # 4) Pulso unitário
        a0 = np.zeros(n, dtype=np.float32)
        a0[base_idx] = 1.0

        # Se não há modos, retorna localmente
        if self.modes is None or self.eigenvalues is None:
            self.state = a0  # CORREÇÃO: Substitui ao invés de +=
            return

        # 5) Projeção nos modos
        U = self.modes
        lamb = self.eigenvalues

        c0 = U.T @ a0

        # 6) Difusão radial (heat kernel)
        decay = np.exp(-lamb * diffusion_time)
        c_t = decay * c0

        # 7) Reconstrução no espaço dos nós
        a_t = U @ c_t
        a_t = np.nan_to_num(a_t, nan=0.0)

        # 8) Normalização pelo MÁXIMO (Pico = 1.0)
        # Isso garante que a excitação comece forte e decaia de verdade
        norm = np.abs(a_t).max()
        if norm > 1e-8:
            a_t_norm = a_t / norm
        else:
            a_t_norm = np.zeros_like(a_t)

        # 9) SUBSTITUIÇÃO (não acúmulo)
        self.state = a_t_norm.astype(np.float32)  # MUDANÇA CRÍTICA
        self.state = np.clip(self.state, -1.0, 1.0)


    def decay_state(self, rate: float = 0.1):
        """
        Faz o estado vibracional decair (esquecimento parcial).
        
        rate: fração que se aproxima de zero (0.1 = 10% de relaxamento por chamada)
        """
        if self.state is None:
            return
        self.state *= (1.0 - rate)

    # =========================================================
    # CONSULTA CONTEXTUAL (COM ESTADO)
    # =========================================================
    def query_with_state(self, text: str, top_k: int = 5, alpha: float = 0.5, gamma: float = 0.5):
        """
        Consulta híbrida: vetorial + espectral + RESSONÂNCIA SELETIVA.
        
        NOVO PARADIGMA:
        - Estado vibracional é GLOBAL (afeta todo o banco)
        - MAS só entra em RESSONÂNCIA com o que vibra IGUAL
        - Ressonância = similaridade entre sims_hybrid e estado
        - Resultado: Seletividade verdadeira por frequência de significado
        
        FÍSICA DO CONCEITO:
        - Estado é como um campo vibrante no espaço semântico
        - Query também cria uma vibração (sims_hybrid)
        - Nós com estado SIMILAR à query ressoam fortemente
        - Nós com estado DIFERENTE não ressoam
        """

        if self.embeddings is None or len(self.concepts) == 0:
            raise ValueError("Banco vibracional vazio.")

        self.ensure_state()

        # ========================================================================
        # PARTE 1: CALCULAR SIMILARIDADES (independente de estado)
        # ========================================================================
        
        # 1) Embedding da consulta
        q_emb = self.encode(text)

        # 2) Similaridade vetorial RAW
        sims_vec = self._cosine_sim_all(q_emb)
        sims_vec = np.nan_to_num(sims_vec, nan=0.0)

        # 3) Similaridade espectral (desacoplada de estado)
        sims_spec = self._spectral_similarity(sims_vec)
        sims_spec = np.nan_to_num(sims_spec, nan=0.0)

        # ========================================================================
        # PARTE 2: NORMALIZAR COM MIN-MAX (preserva ordem)
        # ========================================================================
        
        # Normalização por Escala (Max): preserva o zero e a proporção
        # Min-max força o menor valor a zero, o que mata a multiplicação do boost
        def normalize_scale(arr):
            arr_clipped = np.maximum(arr, 0.0)
            a_max = arr_clipped.max()
            if a_max < 1e-8:
                return np.zeros_like(arr, dtype=np.float32) + 0.01
            # Floor de 0.2 para permitir que o contexto resgate itens distantes
            return (arr_clipped / a_max * 0.8 + 0.2).astype(np.float32)
        
        sims_vec_norm = normalize_scale(sims_vec)
        sims_spec_norm = normalize_scale(sims_spec)

        # ========================================================================
        # PARTE 3: COMBINAR VETORIAL + ESPECTRAL (híbrido)
        # ========================================================================
        
        # Combinação híbrida (sem estado ainda)
        sims_hybrid = alpha * sims_vec_norm + (1.0 - alpha) * sims_spec_norm
        # ========================================================================
        # PARTE 4: RESSONÂNCIA SELETIVA (estado só afeta o que vibra igual)
        # ========================================================================
        
        # NOVO CONCEITO: Ressonância por similaridade de frequência
        # 
        # Estado bruto (respeita o decaimento)
        # Se o estado decaiu para 0.1, o boost será fraco (0.1), permitindo troca de assunto
        state_norm = np.clip(self.state, -1.0, 1.0)
        
        # RESSONÂNCIA: O estado define a região de ressonância.
        # A seletividade vem do próprio estado (que é local), não da comparação com a query.
        # Removemos a penalidade de diferença para permitir que o contexto "resgate" conceitos.
        resonancia = np.abs(state_norm)
        
        # Boost = estado * ressonância
        # Preserva o sinal do estado:
        # - Estado positivo (+) * ressonância (+) = Boost positivo (Amplifica)
        # - Estado negativo (-) * ressonância (+) = Boost negativo (Suprime)
        boost = state_norm * resonancia
        
        # ========================================================================
        # PARTE 5: APLICAR BOOST COM RESSONÂNCIA
        # ========================================================================
        
        # Fórmula final: sims_total = sims_hybrid * (1 + gamma * boost)
        # 
        # CASOS:
        # 1. sims_hybrid=0.9, state=0.9, resonancia=1.0, boost=0.9
        #    sims_total = 0.9 * (1 + 0.5 * 0.9) = 0.9 * 1.45 = 1.305
        #    (amplificado: ressoam!)
        #
        # 2. sims_hybrid=0.9, state=0.1, resonancia=0.2, boost=0.02
        #    sims_total = 0.9 * (1 + 0.5 * 0.02) = 0.9 * 1.01 = 0.909
        #    (quase sem amplificação: não ressoam!)
        #
        # 3. sims_hybrid=0.1, state=0.1, resonancia=1.0, boost=0.1
        #    sims_total = 0.1 * (1 + 0.5 * 0.1) = 0.1 * 1.05 = 0.105
        #    (pouco amplificado: estado baixo)
        
        sims_total = sims_hybrid * (1.0 + gamma * boost)
        sims_total = np.clip(sims_total, 0.0, 2.0)

        # ========================================================================
        # PARTE 6: ORDENAÇÃO E RESULTADOS
        # ========================================================================
        
        idx_sorted = np.argsort(sims_total)[::-1]
        idx_top = idx_sorted[:top_k]

        resultados = []
        for i in idx_top:
            resultados.append({
                "id": i,
                "texto": self.concepts[i],
                "sim_vec": float(sims_vec[i]),
                "sim_spec": float(sims_spec[i]),
                "state_contrib": float(self.state[i]),
                "resonancia": float(resonancia[i]),
                "boost": float(boost[i]),
                "ressonancia_total": float(sims_total[i])
            })

        return resultados

    def _spectral_similarity(self, sims_vec):
        """
        Projeta similaridade vetorial na estrutura espectral do grafo.
        
        PRINCÍPIOS:
        - Espectro é ESTRUTURAL (constante, propriedade do grafo)
        - Estado é DINÂMICO (temporário, propriedade da sessão)
        - Similaridade espectral NÃO deve ser afetada por estado
        
        ALGORITMO:
        1. Projeta sims_vec nos autovetores (modos espectrais)
        2. Pondera por 1/(1+λ) para priorizar baixas frequências
        3. Reconstrói no espaço original
        
        RESULTADO:
        - Preserva ordem relativa dos scores
        - Suaviza ruído de alta frequência
        - Desacoplado completamente de estado vibracional
        
        Args:
            sims_vec: Vetor de similaridade (N,) de uma query
        
        Returns:
            Similaridade espectral (N,) com mesma estrutura que sims_vec
        """
        
        # ========================================================================
        # VERIFICAÇÃO: Espectro disponível?
        # ========================================================================
        
        if self.modes is None or self.eigenvalues is None:
            # Se não há espectro, retorna vetorial puro
            return sims_vec.copy().astype(np.float32)
        
        # ========================================================================
        # EXTRAÇÃO DE COMPONENTES
        # ========================================================================
        
        U = self.modes          # Matriz de autovetores (N × N)
        lamb = self.eigenvalues # Autovalores (N,) — sempre ≥ 0 para Laplaciana
        
        # ========================================================================
        # PASSO 1: PROJETAR NO ESPAÇO ESPECTRAL
        # ========================================================================
        # Transforma vetor original em coordenadas espectrais
        # c[i] = <sims_vec, u_i> onde u_i é o i-ésimo autovetor
        
        c0 = U.T @ sims_vec  # (N × N) @ (N) → (N)
        c0 = np.nan_to_num(c0, nan=0.0)  # Remove NaNs se houver
        
        # ========================================================================
        # PASSO 2: PONDERAR POR AUTOVALOR
        # ========================================================================
        # Baixas frequências (λ pequeno) recebem peso alto
        # Altas frequências (λ grande) recebem peso baixo
        # Isso suaviza ruído naturalmente, sem difusão artificial
        
        # Peso = 1 / (1 + λ)
        # Com λ ≥ 0: peso ∈ (0, 1]
        # λ=0 → peso=1 (modo zero, máxima influência)
        # λ→∞ → peso→0 (ruído de alta frequência, mínima influência)
        
        weights = 1.0 / (1.0 + lamb + 1e-8)  # +1e-8 para evitar divisão por zero
        weights = np.nan_to_num(weights, nan=0.0)
        
        # Aplica pesos às componentes espectrais
        c_weighted = weights * c0
        
        # ========================================================================
        # PASSO 3: RECONSTRUIR NO ESPAÇO ORIGINAL
        # ========================================================================
        # Transforma de volta para coordenadas do espaço original
        # sims_spec = Σ c_weighted[i] * u_i
        
        sims_spec = U @ c_weighted  # (N × N) @ (N) → (N)
        sims_spec = np.nan_to_num(sims_spec, nan=0.0)
        
        # ========================================================================
        # PASSO 4: CLIPPING E NORMALIZAÇÃO FINAL
        # ========================================================================
        # Garante que resultado está em escala razoável [0, max(sims_vec)]
        
        sims_spec = np.clip(sims_spec, 0.0, sims_vec.max() + 0.1)
        
        return sims_spec.astype(np.float32)


    # ============================================================================
    # DOCUMENTAÇÃO ADICIONAL
    # ============================================================================

    """
    EXEMPLO DE COMPORTAMENTO:

    Input: sims_vec = [0.99, 0.99, 0.99, 0.97, 0.50, 0.01, 0.001, ...]
        (query "história da matemática")

    Sem _spectral_similarity():
    sims_total = [0.99, 0.99, 0.99, 0.97, 0.50, 0.01, 0.001, ...]
    (scores brutos, podem ter ruído)

    Com _spectral_similarity():
    Projeta em modos espectrais
    Pondera por 1/(1+λ)
    Reconstrói
    sims_spec = [0.985, 0.985, 0.985, 0.965, 0.48, 0.008, 0.0008, ...]
    (ordem MANTIDA, mas suavizado — ruído removido)

    Depois que excita com "visual":
    Estado afeta apenas nós visuais
    Mas _spectral_similarity() não "vê" estado
    Matemática permanece no topo onde deveria estar

    COMPARAÇÃO COM VERSÃO ANTERIOR:

    Antes:
    decay = exp(-λ * 1.0)  # Decay forte
    Resultado: Suaviza demais, mistura tudo
    Problema: Excitação em um lado do grafo afeta o outro lado

    Depois:
    weights = 1.0 / (1.0 + λ)  # Ponderação suave
    Resultado: Prioriza baixas frequências, mantém ordem
    Benefício: Excitação fica local, não contamina queries distantes
    """ 