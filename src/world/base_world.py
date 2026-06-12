"""Clase base abstracta para mundos de simulación."""

from abc import ABC, abstractmethod

import networkx as nx

from src.agents.base import BaseAgent
from src.utils.config import SimulationConfig


class BaseWorld(ABC):
    """Mundo base de simulación."""

    def __init__(self, agents: list[BaseAgent], network: nx.Graph, config: SimulationConfig):
        self.agents = agents
        self.network = network
        self.config = config
        self.history: list[dict] = []
        self.current_step = 0

    @abstractmethod
    def step(self) -> dict:
        """Ejecuta un paso de simulación. Retorna métricas."""
        ...

    def run(self, n_steps: int | None = None) -> list[dict]:
        """Ejecuta n_steps pasos de simulación."""
        n_steps = n_steps or self.config.n_steps
        for _ in range(n_steps):
            metrics = self.step()
            self.history.append(metrics)
            self.current_step += 1
        return self.history

    def get_all_opinions(self, topic: str) -> list[float]:
        """Retorna las opiniones de todos los agentes sobre un tema."""
        return [a.get_opinion(topic) for a in self.agents]

    def get_snapshot(self) -> list[dict]:
        """Snapshot del estado de todos los agentes."""
        return [a.get_state() for a in self.agents]
