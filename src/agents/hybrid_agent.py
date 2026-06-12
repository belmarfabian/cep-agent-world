"""Agente híbrido (Opción C) - LLM + restricción estadística."""

import numpy as np

from src.agents.llm_agent import LLMAgent
from src.data.profiler import AgentProfile


class HybridAgent(LLMAgent):
    """
    Combina generación LLM con validación estadística.
    El agente conversa como LLM, pero sus cambios de opinión
    se anclan a la distribución empírica de su arquetipo.
    """

    def __init__(
        self,
        profile: AgentProfile,
        archetype_centroid: dict[str, float],
        archetype_std: dict[str, float],
        drift_correction: float = 0.5,
    ):
        super().__init__(profile)
        self.archetype_centroid = archetype_centroid
        self.archetype_std = archetype_std
        self.drift_correction = drift_correction

    def update_opinions_from_response(self, response: str) -> dict[str, float]:
        """Actualiza opiniones con corrección de drift estadístico."""
        # Primero, parsear como LLM normal
        raw_updates = super().update_opinions_from_response(response)

        # Luego, aplicar corrección estadística
        corrected = {}
        for topic, new_value in raw_updates.items():
            corrected_value = self._apply_drift_correction(topic, new_value)
            self.profile.current_opinions[topic] = corrected_value
            corrected[topic] = corrected_value

        return corrected

    def _apply_drift_correction(self, topic: str, new_value: float) -> float:
        """
        Si la nueva opinión se aleja demasiado del arquetipo,
        la atenúa proporcionalmente.
        """
        # Mapear topic a variable de centroide
        topic_to_var = {
            "eje_izq_der": "posicion_politica",
            "eval_gobierno": "eval_gobierno",
            "democracia": "posicion_politica",
        }
        var = topic_to_var.get(topic)
        if var is None or var not in self.archetype_centroid:
            return new_value

        centroid = self.archetype_centroid[var]
        std = self.archetype_std.get(var, 1.0)

        if std < 0.01:
            return new_value

        # Normalizar centroide a [-1, 1] si es necesario
        if var == "posicion_politica":
            centroid_norm = (centroid - 5) / 5
        elif var == "eval_gobierno":
            centroid_norm = centroid * 2 - 1
        else:
            centroid_norm = centroid

        # Calcular z-score
        z = abs(new_value - centroid_norm) / (std + 0.01)

        if z > 2.0:
            # Atenuar: mezcla entre nuevo valor y centroide
            alpha = self.drift_correction
            corrected = new_value * (1 - alpha) + centroid_norm * alpha
            return float(np.clip(corrected, -1, 1))

        return new_value
