"""Clase base abstracta para todos los tipos de agentes."""

from abc import ABC, abstractmethod

from src.data.profiler import AgentProfile


class BaseAgent(ABC):
    """Agente base del mundo simulado CEP."""

    def __init__(self, profile: AgentProfile):
        self.profile = profile
        self.step_count = 0

    @property
    def agent_id(self) -> str:
        return self.profile.agent_id

    @abstractmethod
    def step(self, neighbors: list["BaseAgent"], topic: str) -> dict:
        """Ejecuta un paso de simulación. Retorna métricas del paso."""
        ...

    @abstractmethod
    def get_opinion(self, topic: str) -> float:
        """Retorna la posición actual sobre un tema [-1, 1]."""
        ...

    def get_all_opinions(self) -> dict[str, float]:
        """Retorna todas las opiniones actuales."""
        return dict(self.profile.current_opinions)

    def get_state(self) -> dict:
        """Snapshot del estado actual del agente."""
        return {
            "agent_id": self.agent_id,
            "step": self.step_count,
            "opinions": self.get_all_opinions(),
            "archetype_id": self.profile.archetype_id,
            "political_position": self.profile.political_position,
            "gse": self.profile.gse,
            "region": self.profile.region,
            "age": self.profile.age,
        }
