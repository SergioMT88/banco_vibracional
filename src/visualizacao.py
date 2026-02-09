import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA

def plotar_grafo(db, titulo="Grafo Vibracional"):
    """
    Visualiza o grafo de conceitos usando PCA para redução de dimensionalidade.
    Pinta os nós de acordo com o estado vibracional atual.
    """
    if len(db.concepts) < 2:
        print("Poucos conceitos para visualizar.")
        return

    print(f"Gerando visualização para {len(db.concepts)} conceitos...")

    # 1. Coordenadas 2D via PCA nos embeddings (mapa semântico)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(db.embeddings)
    
    # 2. Construir grafo para desenhar conexões
    # Usamos a matriz de adjacência esparsa do DB
    try:
        G = nx.from_scipy_sparse_array(db.graph)
    except AttributeError:
        # Fallback para versões antigas do networkx
        G = nx.from_scipy_sparse_matrix(db.graph)
    
    # 3. Configurar cores baseadas no estado
    if db.state is None:
        colors = np.zeros(len(db.concepts))
    else:
        colors = db.state

    plt.figure(figsize=(12, 8))
    
    # Desenhar nós
    # cmap='coolwarm': Azul (negativo) -> Branco (zero) -> Vermelho (positivo)
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, 
        cmap='coolwarm', 
        s=300, 
        alpha=0.9,
        edgecolors='#333333',
        linewidths=1.0,
        vmin=-1.0, vmax=1.0
    )
    
    # Desenhar arestas (apenas as mais fortes para limpeza visual)
    # Iteramos direto da matriz esparsa para performance
    cx = db.graph.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i < j and v > 0.5: # Apenas conexões fortes
            p1 = coords[i]
            p2 = coords[j]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.15, linewidth=0.5)

    # Rótulos dos conceitos
    for i, txt in enumerate(db.concepts):
        # Plota o texto se o nó tiver ativação relevante ou se forem poucos nós
        if len(db.concepts) < 30 or abs(colors[i]) > 0.1:
            plt.text(coords[i, 0]+0.02, coords[i, 1]+0.02, txt[:40]+"...", fontsize=9, alpha=0.8)

    plt.title(titulo, fontsize=14)
    plt.colorbar(scatter, label="Ativação Vibracional (-1 a +1)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()