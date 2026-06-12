"""Motor de conversación entre agentes LLM."""

import asyncio

from src.agents.llm_agent import LLMAgent
from src.interactions.prompts import REPORT_POSITION_PROMPT
from src.interactions.topics import Topic
from src.utils.llm_client import LLMClient


async def run_conversation(
    agent_a: LLMAgent,
    agent_b: LLMAgent,
    topic: Topic,
    client: LLMClient,
    n_turns: int = 4,
) -> dict:
    """
    Ejecuta una conversación entre dos agentes LLM sobre un tema.

    Returns:
        dict con el log de la conversación y los cambios de opinión.
    """
    log = {
        "agent_a": agent_a.agent_id,
        "agent_b": agent_b.agent_id,
        "topic": topic.id,
        "turns": [],
        "opinion_changes": {},
    }

    # Limpiar historiales para esta conversación
    agent_a.clear_history()
    agent_b.clear_history()

    # Agente A inicia con el tema
    initial_msg = topic.prompt_seed
    agent_a.add_message("user", f"Alguien te pregunta: {initial_msg}")

    response_a = await client.chat(
        system=agent_a.system_prompt,
        messages=agent_a.get_messages(),
    )
    agent_a.add_message("assistant", response_a)
    log["turns"].append({"speaker": agent_a.agent_id, "text": response_a})

    # Turnos de conversación
    current_speaker, other_speaker = agent_b, agent_a
    last_message = response_a

    for turn in range(n_turns - 1):
        current_speaker.add_message("user", last_message)

        response = await client.chat(
            system=current_speaker.system_prompt,
            messages=current_speaker.get_messages(),
        )
        current_speaker.add_message("assistant", response)
        log["turns"].append({"speaker": current_speaker.agent_id, "text": response})

        last_message = response
        current_speaker, other_speaker = other_speaker, current_speaker

    # Pedir posiciones actualizadas
    opinions_before_a = dict(agent_a.profile.current_opinions)
    opinions_before_b = dict(agent_b.profile.current_opinions)

    for agent in [agent_a, agent_b]:
        agent.add_message("user", REPORT_POSITION_PROMPT)
        position_response = await client.chat(
            system=agent.system_prompt,
            messages=agent.get_messages(),
            max_tokens=100,
        )
        agent.update_opinions_from_response(position_response)

    log["opinion_changes"] = {
        agent_a.agent_id: _compute_changes(opinions_before_a, agent_a.profile.current_opinions),
        agent_b.agent_id: _compute_changes(opinions_before_b, agent_b.profile.current_opinions),
    }

    return log


def _compute_changes(before: dict, after: dict) -> dict:
    """Calcula el delta de opiniones."""
    changes = {}
    for key in set(before) | set(after):
        old = before.get(key, 0.0)
        new = after.get(key, 0.0)
        if old != new:
            changes[key] = {"before": old, "after": new, "delta": new - old}
    return changes


async def run_batch_conversations(
    pairs: list[tuple[LLMAgent, LLMAgent]],
    topics: list[Topic],
    client: LLMClient,
    n_turns: int = 4,
) -> list[dict]:
    """Ejecuta múltiples conversaciones en paralelo con rate limiting."""
    tasks = [
        run_conversation(a, b, topic, client, n_turns)
        for (a, b), topic in zip(pairs, topics)
    ]
    return await asyncio.gather(*tasks)
