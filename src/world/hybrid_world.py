"""Mundo híbrido (Opción C) - LLM + validación estadística."""

import numpy as np

from src.agents.hybrid_agent import HybridAgent
from src.world.llm_world import LLMWorld


class HybridWorld(LLMWorld):
    """
    Como LLMWorld, pero usa HybridAgents que corrigen drift estadístico.
    Cada M steps, recalcula los arquetipos.
    """

    def __init__(self, agents: list[HybridAgent], network, config, archetype_data: dict):
        super().__init__(agents, network, config)
        self.archetype_data = archetype_data
        self.recalc_interval = 10  # Recalcular arquetipos cada 10 steps

    async def async_step(self) -> dict:
        """Step híbrido: conversación LLM + corrección estadística."""
        metrics = await super().async_step()

        # La corrección ocurre automáticamente en HybridAgent.update_opinions_from_response

        # Cada N steps, reportar distribución vs. arquetipo
        if self.current_step > 0 and self.current_step % self.recalc_interval == 0:
            drift_report = self._compute_drift_report()
            metrics["drift_report"] = drift_report

        return metrics

    def _compute_drift_report(self) -> dict:
        """Compara distribución actual vs. arquetipos originales."""
        report = {}
        for k, arch in self.archetype_data["archetypes"].items():
            agents_in_cluster = [
                a for a in self.agents if a.profile.archetype_id == k
            ]
            if not agents_in_cluster:
                continue

            centroid = arch["centroid"]
            for dim in ["eje_izq_der", "eval_gobierno", "democracia"]:
                opinions = [a.get_opinion(dim) for a in agents_in_cluster]
                current_mean = float(np.mean(opinions))
                report[f"cluster_{k}_{dim}_mean"] = current_mean

        return report
