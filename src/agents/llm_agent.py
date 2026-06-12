"""Agente LLM puro (Opción A) - personalidad vía prompt de Claude."""

import json
import re

from src.agents.base import BaseAgent
from src.data.profiler import AgentProfile
from src.interactions.prompts import build_system_prompt


class LLMAgent(BaseAgent):
    """
    Agente cuyo comportamiento es generado por un LLM.
    Cada agente tiene un system prompt derivado de su perfil CEP.
    """

    def __init__(self, profile: AgentProfile):
        super().__init__(profile)
        self.system_prompt = build_system_prompt(profile)
        self.conversation_history: list[dict] = []
        self._last_response: str = ""

    def step(self, neighbors: list[BaseAgent], topic: str) -> dict:
        """
        Nota: En Opción A, el step real es asíncrono y se ejecuta
        desde LLMWorld. Este método solo registra el turno.
        """
        self.step_count += 1
        return {"step": self.step_count}

    def get_opinion(self, topic: str) -> float:
        return self.profile.current_opinions.get(topic, 0.0)

    def add_message(self, role: str, content: str):
        """Agrega un mensaje al historial de conversación."""
        self.conversation_history.append({"role": role, "content": content})

    def get_messages(self) -> list[dict]:
        return list(self.conversation_history)

    def clear_history(self):
        """Limpia historial para nueva conversación."""
        self.conversation_history = []

    def update_opinions_from_response(self, response: str) -> dict[str, float]:
        """
        Parsea la respuesta del LLM para extraer posiciones actualizadas.
        Espera un bloque JSON en la respuesta.
        """
        # Buscar JSON en la respuesta
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            try:
                positions = json.loads(json_match.group())
                for topic, value in positions.items():
                    if isinstance(value, (int, float)) and -1 <= value <= 1:
                        self.profile.current_opinions[topic] = float(value)
                return positions
            except (json.JSONDecodeError, ValueError):
                pass
        return {}
