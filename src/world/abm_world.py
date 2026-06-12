"""Mundo ABM clásico (Opción B) - simulación sin LLM."""

import numpy as np
from rich.progress import track

from src.agents.abm_agent import ABMAgent
from src.interactions.topics import TOPICS, get_random_topic
from src.world.base_world import BaseWorld
from src.world.topology import get_neighbors


class ABMWorld(BaseWorld):
    """
    Mundo con dinámica de opiniones tipo Deffuant-Weisbuch.
    Rápido, sin llamadas a API.
    """

    def __init__(self, agents: list[ABMAgent], network, config):
        super().__init__(agents, network, config)
        self.rng = np.random.default_rng(config.random_seed)
        # Temas sobre los que interactúan
        self.active_topics = list(TOPICS.keys())[:4]  # Usar 4 temas principales

    def step(self) -> dict:
        """Un paso: cada agente interactúa con un vecino aleatorio."""
        n_interactions = 0
        n_changes = 0
        total_delta = 0.0

        # Activación aleatoria
        order = self.rng.permutation(len(self.agents))

        for i in order:
            agent = self.agents[i]
            neighbors = get_neighbors(self.network, i, self.agents)
            if not neighbors:
                continue

            # Elegir un tema aleatorio
            topic_id = self.active_topics[self.rng.integers(0, len(self.active_topics))]

            result = agent.step(neighbors, topic_id)
            n_interactions += 1
            if result.get("changed"):
                n_changes += 1
                total_delta += abs(result.get("delta", 0))

        # Calcular métricas
        metrics = {
            "step": self.current_step,
            "n_interactions": n_interactions,
            "n_changes": n_changes,
            "change_rate": n_changes / max(n_interactions, 1),
            "avg_delta": total_delta / max(n_changes, 1),
        }

        # Opiniones por tema
        for topic in self.active_topics:
            opinions = self.get_all_opinions(topic)
            metrics[f"mean_{topic}"] = float(np.mean(opinions))
            metrics[f"std_{topic}"] = float(np.std(opinions))
            metrics[f"polarization_{topic}"] = float(np.var(opinions))

        return metrics

    def run_with_progress(self, n_steps: int | None = None) -> list[dict]:
        """Ejecuta con barra de progreso."""
        n_steps = n_steps or self.config.n_steps
        for _ in track(range(n_steps), description="Simulación ABM..."):
            metrics = self.step()
            self.history.append(metrics)
            self.current_step += 1
        return self.history
