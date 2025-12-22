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
        - similaridade espectral (modos da Laplaciana)
        
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

        # 1) embedding da consulta
        try:
            q_emb = self.encode(text)
        except Exception:
            q_emb = np.zeros(self.embedding_dim, dtype=np.float32)

        # 2) similaridade vetorial
        sims_vec = self._cosine_sim_all(q_emb)
        sims_vec = np.nan_to_num(sims_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # 3) componente espectral
        if self.modes is None or self.modes.shape[0] != n:
            sims_spec_norm = np.zeros_like(sims_vec)
            sims_total = sims_vec
        else:
            m = min(self.n_modes, n)
            base_idx = np.argsort(sims_vec)[::-1][:m]

            q_spec = self.modes[base_idx].mean(axis=0)  # (n_modes,)

            sims_spec_raw = self.modes @ q_spec         # (n,)

            sims_spec_raw = np.nan_to_num(sims_spec_raw, nan=0.0)

            std_spec = sims_spec_raw.std()
            if std_spec < 1e-8:
                sims_spec_norm = np.zeros_like(sims_spec_raw)
            else:
                sims_spec_norm = (sims_spec_raw - sims_spec_raw.mean()) / (std_spec + 1e-8)

            std_vec = sims_vec.std()
            if std_vec < 1e-8:
                sims_vec_norm = np.zeros_like(sims_vec)
            else:
                sims_vec_norm = (sims_vec - sims_vec.mean()) / (std_vec + 1e-8)

            sims_total = alpha * sims_vec_norm + (1.0 - alpha) * sims_spec_norm

        sims_total = np.nan_to_num(sims_total, nan=0.0)

        idx = np.argsort(sims_total)[::-1][:k]

        resultados = []
        for i in idx:
            resultados.append(
                {
                    "id": int(i),
                    "text": self.concepts[i],
                    "sim_vec": float(sims_vec[i]),
                    "sim_spec": float(sims_spec_norm[i]),
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

    def excite(self, text: str, diffusion_time: float = 1.0, base_fraction: float = 0.01):
        """
        Emite um pulso unitário que se propaga radialmente pelo grafo.

        - O pulso inicial é sempre 1.0 nos nós-base.
        - Não existe mais 'strength'.
        - A propagação é radial, controlada apenas por diffusion_time.
        - O resultado é normalizado antes de entrar no estado vibracional.
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

        # 4) Pulso unitário (não depende mais de força)
        a0 = np.zeros(n, dtype=np.float32)
        a0[base_idx] = 1.0

        # Normaliza para virar uma distribuição
        a0 /= a0.sum()

        # Se não há modos, acumula localmente
        if self.modes is None or self.eigenvalues is None:
            self.state += a0
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

        # 8) Normalização radial (z-score)
        std = a_t.std()
        if std < 1e-8:
            a_t_norm = np.zeros_like(a_t)
        else:
            a_t_norm = (a_t - a_t.mean()) / (std + 1e-8)

        # 9) Atualiza o estado vibracional
        self.state += a_t_norm.astype(np.float32)
        self.state = np.tanh(self.state)


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
    import numpy as np

    def query_with_state(self, text: str, top_k: int = 5, alpha: float = 0.5, gamma: float = 0.4):
        """
        Consulta híbrida: vetorial + espectral + estado vibracional,
        com GATING SIGMOIDAL para impedir alucinações estruturais.
        """

        if self.embeddings is None or len(self.concepts) == 0:
            raise ValueError("Banco vibracional vazio.")

        self.ensure_state()

        # 1) Embedding da consulta
        q_emb = self.encode(text)

        # 2) Similaridade vetorial
        sims_vec = self._cosine_sim_all(q_emb)
        sims_vec = np.nan_to_num(sims_vec, nan=0.0)

        # Normalização vetorial
        sims_vec_norm = (sims_vec - sims_vec.min()) / (sims_vec.max() - sims_vec.min() + 1e-8)

        # 3) Similaridade espectral (agora correta)
        sims_spec = self._spectral_similarity(sims_vec)
        sims_spec = np.nan_to_num(sims_spec, nan=0.0)

        # Normalização espectral
        sims_spec_norm = (sims_spec - sims_spec.min()) / (sims_spec.max() - sims_spec.min() + 1e-8)

        # 4) Estado vibracional normalizado
        state_norm = (self.state - self.state.min()) / (self.state.max() - self.state.min() + 1e-8)

        # 5) === GATING SIGMOIDAL ===
        t = 0.0      # limiar
        k = 8.0      # dureza do portão

        g = 1 / (1 + np.exp(-k * (sims_vec_norm - t)))

        # Combinação gated
        sims_hybrid = g * sims_spec_norm + (1 - g) * sims_vec_norm

        # 6) Combinação final com estado vibracional
        sims_total = (1 - gamma) * sims_hybrid + gamma * state_norm

        # 7) Ordenação
        idx_sorted = np.argsort(sims_total)[::-1]
        idx_top = idx_sorted[:top_k]

        # 8) Monta resultados
        resultados = []
        for i in idx_top:
            resultados.append({
                "id": i,
                "texto": self.concepts[i],
                "sim_vec": float(sims_vec[i]),
                "sim_spec": float(sims_spec[i]),
                "state_contrib": float(self.state[i]),
                "ressonancia_total": float(sims_total[i])
            })

        return resultados

    def _spectral_similarity(self, sims_vec, diffusion_time=1.0):
        """
        Similaridade espectral baseada na difusão do vetor de similaridade vetorial.
        A query NÃO é projetada nos modos — apenas o vetor sims_vec (dimensão N).
        """

        if self.modes is None or self.eigenvalues is None:
            return np.zeros(len(self.concepts), dtype=np.float32)

        U = self.modes                # matriz de modos (N x N)
        lamb = self.eigenvalues       # autovalores (N)

        # Projeta a similaridade vetorial nos modos
        c0 = U.T @ sims_vec           # (N x N) @ (N) → (N)

        # Difusão espectral (heat kernel)
        decay = np.exp(-lamb * diffusion_time)
        c_t = decay * c0

        # Reconstrói no espaço dos nós
        sims_spec = U @ c_t           # (N x N) @ (N) → (N)
        sims_spec = np.nan_to_num(sims_spec, nan=0.0)

        return sims_spec.astype(np.float32)