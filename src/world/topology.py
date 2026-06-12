"""Construcción de grafos de interacción entre agentes."""

import numpy as np
import networkx as nx

from src.agents.base import BaseAgent


def build_small_world(agents: list[BaseAgent], k: int = 6, p: float = 0.1, seed: int = 42) -> nx.Graph:
    """Grafo Small-World (Watts-Strogatz)."""
    n = len(agents)
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    # Asignar agentes como atributos de nodos
    for i, agent in enumerate(agents):
        G.nodes[i]["agent"] = agent
        G.nodes[i]["agent_id"] = agent.agent_id
    return G


def build_homophily_network(agents: list[BaseAgent], avg_degree: int = 6, seed: int = 42) -> nx.Graph:
    """
    Grafo basado en homofilia demográfica.
    Probabilidad de conexión proporcional a similitud.
    """
    n = len(agents)
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    for i, agent in enumerate(agents):
        G.add_node(i, agent=agent, agent_id=agent.agent_id)

    # Calcular probabilidades de conexión
    target_edges = n * avg_degree // 2

    for _ in range(target_edges * 3):  # sobremuestrear para compensar rechazos
        if G.number_of_edges() >= target_edges:
            break
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j or G.has_edge(i, j):
            continue

        sim = _demographic_similarity(agents[i], agents[j])
        if rng.random() < sim:
            G.add_edge(i, j)

    return G


def build_scale_free(agents: list[BaseAgent], m: int = 3, seed: int = 42) -> nx.Graph:
    """Grafo Scale-Free (Barabási-Albert)."""
    n = len(agents)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    for i, agent in enumerate(agents):
        G.nodes[i]["agent"] = agent
        G.nodes[i]["agent_id"] = agent.agent_id
    return G


def _demographic_similarity(a: BaseAgent, b: BaseAgent) -> float:
    """Similitud demográfica entre dos agentes [0, 1]."""
    score = 0.0
    n = 0

    # Región
    if a.profile.region == b.profile.region:
        score += 1.0
    n += 1

    # GSE
    gse_order = {"ABC1": 1, "C2": 2, "C3": 3, "D": 4, "E": 5}
    ga = gse_order.get(a.profile.gse, 3)
    gb = gse_order.get(b.profile.gse, 3)
    score += 1.0 - abs(ga - gb) / 4.0
    n += 1

    # Edad
    score += max(0, 1.0 - abs(a.profile.age - b.profile.age) / 40.0)
    n += 1

    return score / n


def get_neighbors(G: nx.Graph, node_id: int, agents: list[BaseAgent]) -> list[BaseAgent]:
    """Retorna los agentes vecinos de un nodo."""
    neighbor_ids = list(G.neighbors(node_id))
    return [agents[j] for j in neighbor_ids]


def build_network(agents: list[BaseAgent], network_type: str, **kwargs) -> nx.Graph:
    """Factory para construir el tipo de red apropiado."""
    builders = {
        "small_world": build_small_world,
        "homophily": build_homophily_network,
        "scale_free": build_scale_free,
    }
    builder = builders.get(network_type)
    if builder is None:
        raise ValueError(f"Tipo de red no soportado: {network_type}")
    return builder(agents, **kwargs)
