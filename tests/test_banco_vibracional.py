import pytest
import numpy as np
import os
import shutil
from src.banco_vibracional import VibrationalDB
from src.cristal import carregar_cristal

@pytest.fixture
def db_vazia():
    """Retorna uma instância vazia de VibrationalDB."""
    return VibrationalDB(n_modes=4, k_neighbors=3)


@pytest.fixture
def db_populada():
    """Retorna uma instância populada com dois clusters de conceitos."""
    db = VibrationalDB(n_modes=4, k_neighbors=3)
    
    conceitos_tech = [
        "redes neurais artificiais processam dados",
        "python é uma linguagem de programação versátil",
        "inteligência artificial generativa cria conteúdo",
    ]
    conceitos_natureza = [
        "a floresta amazônica é um ecossistema diverso",
        "recifes de coral são estruturas subaquáticas",
        "biologia marinha estuda a vida nos oceanos",
    ]
    
    db.adicionar_conceitos(conceitos_tech + conceitos_natureza)
    db.recalcular_espectro()
    return db


def test_ingestao_e_estrutura(db_populada):
    """Verifica se a ingestão e o cálculo da estrutura funcionam."""
    db = db_populada
    assert len(db.concepts) == 6
    assert db.embeddings.shape == (6, 384)
    assert db.graph is not None
    assert db.laplacian is not None
    assert db.modes.shape == (6, 4)
    assert db.eigenvalues.shape == (4,)
    assert db.state.shape == (6,)


def test_query_simples(db_populada):
    """Testa a consulta sem estado (query)."""
    db = db_populada
    # Usamos alpha alto para priorizar a similaridade vetorial direta
    resultados = db.query("linguagem de programação", top_k=1, alpha=0.8)
    
    assert len(resultados) == 1
    # O resultado mais próximo de "linguagem de programação" deve ser sobre Python.
    assert "python" in resultados[0]["texto"]


def test_salvar_e_carregar(db_populada, tmp_path):
    """Testa se o cristal pode ser salvo e carregado corretamente."""
    db = db_populada
    pasta_cristal = tmp_path / "cristal_teste"

    # Excita o estado para ter um valor não-zero
    db.excite("oceanos")
    assert np.any(db.state != 0)

    # Salva
    db.salvar(str(pasta_cristal))

    # Carrega em uma nova instância
    db_carregado = carregar_cristal(VibrationalDB, str(pasta_cristal))

    # Comparações
    assert db_carregado.concepts == db.concepts
    assert np.allclose(db_carregado.embeddings, db.embeddings)
    assert np.allclose(db_carregado.modes, db.modes)
    assert np.allclose(db_carregado.eigenvalues, db.eigenvalues)
    assert np.allclose(db_carregado.state, db.state)


def test_fluxo_vibracional_e_ressonancia_seletiva(db_populada):
    """
    Teste principal que valida o fluxo de excitação e consulta com estado.
    Verifica se a ressonância seletiva funciona como esperado.
    """
    db = db_populada
    
    # IDs dos conceitos para facilitar a verificação
    id_python = 1
    id_amazonia = 3

    # --- PASSO 1: Consulta de linha de base (sem estado) ---
    # Uma consulta ambígua como "sistemas" pode retornar qualquer coisa.
    # Não faremos asserções fortes aqui, apenas observamos.
    resultados_base = db.query_with_state("sistemas complexos", top_k=1)
    print(f"\nResultado base para 'sistemas complexos': {resultados_base[0]['texto'][:50]}...")


    # --- PASSO 2: Excitar o cristal com um contexto de "Natureza" ---
    db.excite("biologia e ecologia", diffusion_time=0.5)

    # Verifica se o estado foi ativado corretamente:
    # Os nós de natureza (índices 3, 4, 5) devem ter mais energia que os de tecnologia (0, 1, 2).
    estado_tech = np.sum(np.abs(db.state[:3]))
    estado_natureza = np.sum(np.abs(db.state[3:]))
    assert estado_natureza > estado_tech


    # --- PASSO 3: Teste de Ressonância Contextual ---
    # A mesma consulta ambígua ("sistemas complexos") agora deve ser atraída pelo
    # estado vibracional ativo, favorecendo os conceitos de natureza.
    resultados_com_estado = db.query_with_state("sistemas complexos", top_k=1, gamma=8.0)
    
    print(f"Resultado com estado 'Natureza' para 'sistemas complexos': {resultados_com_estado[0]['texto'][:50]}...")
    assert resultados_com_estado[0]['id'] >= 3 # Deve ser um conceito de natureza


    # --- PASSO 4: Teste de Ressonância SELETIVA (o mais importante) ---
    # Agora, fazemos uma consulta específica e não relacionada ao estado ("software").
    # O estado de "natureza" NÃO DEVE contaminar ou interferir no resultado.
    # A "ressonância" deve ser baixa, e a similaridade híbrida deve dominar.
    resultados_software = db.query_with_state("linguagem de programação", top_k=1)

    print(f"Resultado com estado 'Natureza' para 'software': {resultados_software[0]['texto'][:50]}...")
    # O resultado DEVE ser o conceito de Python, provando que o estado não interferiu.
    assert resultados_software[0]['id'] == id_python

    # Verificando a física: a ressonância para o nó de Python deve ser baixa
    ressonancia_python = [r for r in resultados_software if r['id'] == id_python][0]['resonancia']
    # Verificando a física: a ressonância para o nó da Amazônia deve ser baixa também
    # (pois a query não tem a ver com Amazônia)
    # Para obter esse valor, precisamos consultar com top_k maior
    resultados_software_full = db.query_with_state("linguagem de programação", top_k=6)
    ressonancia_amazonia = [r for r in resultados_software_full if r['id'] == id_amazonia][0]['resonancia']

    # A ressonância deve ser baixa em ambos os casos, pois o estado (natureza)
    # e a query (software) estão em "frequências" diferentes.
    assert ressonancia_python < 0.5
    assert ressonancia_amazonia < 0.5


    # --- PASSO 5: Teste de Decaimento ---
    db.decay_state(rate=1.0)
    # O estado deve ter quase voltado a zero.
    assert np.allclose(np.sum(np.abs(db.state)), 0, atol=1e-3)


# Para rodar os testes, navegue até a pasta raiz do projeto no terminal e execute:
# pytest -v -s
#
# A flag -s é útil para ver as saídas de print() dos testes.