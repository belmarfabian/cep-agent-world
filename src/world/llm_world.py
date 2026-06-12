"""Mundo LLM (Opción A) - agentes conversan vía Claude."""

import asyncio

import numpy as np

from src.agents.llm_agent import LLMAgent
from src.interactions.conversation import run_batch_conversations
from src.interactions.topics import TOPICS, Topic
from src.utils.llm_client import LLMClient
from src.world.base_world import BaseWorld


class LLMWorld(BaseWorld):
    """
    Mundo donde agentes LLM conversan sobre temas políticos/sociales.
    Las conversaciones se ejecutan de forma asíncrona con rate limiting.
    """

    def __init__(self, agents: list[LLMAgent], network, config):
        super().__init__(agents, network, config)
        self.client = LLMClient(config)
        self.rng = np.random.default_rng(config.random_seed)
        self.conversation_logs: list[dict] = []
        self.active_topics = list(TOPICS.values())[:4]

    def step(self) -> dict:
        """Ejecuta un paso de forma síncrona (wrapper)."""
        return asyncio.run(self.async_step())

    async def async_step(self) -> dict:
        """Ejecuta un paso: selecciona pares y ejecuta conversaciones."""
        pairs = self._select_pairs()
        topics = [self._random_topic() for _ in pairs]

        logs = await run_batch_conversations(
            pairs=pairs,
            topics=topics,
            client=self.client,
            n_turns=self.config.conversation_turns,
        )

        self.conversation_logs.extend(logs)

        # Métricas
        n_changes = sum(
            1 for log in logs
            for changes in log["opinion_changes"].values()
            if changes
        )

        metrics = {
            "step": self.current_step,
            "n_conversations": len(pairs),
            "n_opinion_changes": n_changes,
            "total_input_tokens": self.client.total_input_tokens,
            "total_output_tokens": self.client.total_output_tokens,
            "estimated_cost_usd": self.client.estimated_cost_usd,
        }

        # Opiniones agregadas
        for topic_id in ["eje_izq_der", "eval_gobierno", "democracia"]:
            opinions = self.get_all_opinions(topic_id)
            if any(o != 0 for o in opinions):
                metrics[f"mean_{topic_id}"] = float(np.mean(opinions))
                metrics[f"std_{topic_id}"] = float(np.std(opinions))

        return metrics

    def _select_pairs(self) -> list[tuple[LLMAgent, LLMAgent]]:
        """Selecciona pares de agentes para conversar según el grafo."""
        edges = list(self.network.edges())
        if not edges:
            return []

        # Seleccionar un subconjunto de pares por paso
        n_pairs = min(len(edges), max(1, len(self.agents) // 4))
        selected = self.rng.choice(len(edges), size=n_pairs, replace=False)

        pairs = []
        for idx in selected:
            i, j = edges[idx]
            pairs.append((self.agents[i], self.agents[j]))
        return pairs

    def _random_topic(self) -> Topic:
        idx = self.rng.integers(0, len(self.active_topics))
        return self.active_topics[idx]

    def close(self):
        self.client.close()
