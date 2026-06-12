"""Agente ABM clásico (Opción B) - sin LLM, reglas numéricas."""

import numpy as np

from src.agents.base import BaseAgent
from src.data.profiler import AgentProfile


class ABMAgent(BaseAgent):
    """
    Agente con dinámica de opiniones basada en reglas numéricas.
    Implementa Bounded Confidence (Deffuant-Weisbuch).
    """

    def __init__(
        self,
        profile: AgentProfile,
        epsilon: float = 0.3,
        mu: float = 0.5,
        noise_sigma: float = 0.01,
    ):
        super().__init__(profile)
        self.epsilon = epsilon
        self.mu = mu
        self.noise_sigma = noise_sigma

        # Apertura al cambio: jóvenes y más educados son más abiertos
        self.openness = self._compute_openness()

    def _compute_openness(self) -> float:
        """Apertura al cambio basada en demografía CEP."""
        base = 0.5
        # Edad: jóvenes más abiertos
        if self.profile.age < 30:
            base += 0.15
        elif self.profile.age > 60:
            base -= 0.15
        # Educación alta → más apertura
        edu = str(self.profile.education or "")
        if "universitaria" in edu.lower() or "postgrado" in edu.lower():
            base += 0.1
        # Interés político alto → más arraigado (menos cambio)
        if self.profile.political_interest and self.profile.political_interest > 5:
            base -= 0.1
        return np.clip(base, 0.1, 0.9)

    def step(self, neighbors: list[BaseAgent], topic: str) -> dict:
        """Interacción bounded confidence con vecinos."""
        if not neighbors:
            self.step_count += 1
            return {"changed": False}

        rng = np.random.default_rng(self.step_count + hash(self.agent_id) % 10000)
        partner = neighbors[rng.integers(0, len(neighbors))]

        my_opinion = self.get_opinion(topic)
        their_opinion = partner.get_opinion(topic)

        delta = abs(my_opinion - their_opinion)
        changed = False

        if delta < self.epsilon:
            # Influencia ponderada por homofilia y apertura
            homophily_weight = self._homophily(partner)
            effective_mu = self.mu * self.openness * homophily_weight

            new_opinion = my_opinion + effective_mu * (their_opinion - my_opinion)
            # Ruido estocástico
            new_opinion += rng.normal(0, self.noise_sigma)
            new_opinion = np.clip(new_opinion, -1, 1)

            self.profile.current_opinions[topic] = float(new_opinion)
            changed = True

        self.step_count += 1
        return {
            "changed": changed,
            "delta": delta,
            "partner": partner.agent_id,
            "topic": topic,
        }

    def get_opinion(self, topic: str) -> float:
        return self.profile.current_opinions.get(topic, 0.0)

    def _homophily(self, other: BaseAgent) -> float:
        """Similitud demográfica con otro agente [0, 1]."""
        score = 0.0
        n = 0

        # Misma región
        if self.profile.region == other.profile.region:
            score += 1.0
        n += 1

        # GSE similar
        gse_order = {"ABC1": 1, "C2": 2, "C3": 3, "D": 4, "E": 5}
        my_gse = gse_order.get(self.profile.gse, 3)
        their_gse = gse_order.get(other.profile.gse, 3)
        score += 1.0 - abs(my_gse - their_gse) / 4.0
        n += 1

        # Edad similar
        age_diff = abs(self.profile.age - other.profile.age)
        score += max(0, 1.0 - age_diff / 40.0)
        n += 1

        # Misma religión
        if self.profile.religion and other.profile.religion:
            if self.profile.religion == other.profile.religion:
                score += 1.0
            n += 1

        return score / n if n > 0 else 0.5
