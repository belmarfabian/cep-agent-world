"""Catálogo de temas de discusión derivados de la encuesta CEP."""

from dataclasses import dataclass


@dataclass
class Topic:
    id: str
    name: str
    description: str
    prompt_seed: str  # Frase para iniciar la conversación
    opinion_dimensions: list[str]  # Qué dimensiones de opinión afecta


TOPICS = {
    "pensiones": Topic(
        id="pensiones",
        name="Sistema de pensiones",
        description="Reforma al sistema de pensiones: AFP individual vs. sistema de reparto estatal",
        prompt_seed="¿Qué opinas del sistema de pensiones en Chile? ¿Debería mantenerse el sistema de AFP o cambiarse a uno de reparto?",
        opinion_dimensions=["eje_izq_der", "eval_gobierno"],
    ),
    "seguridad": Topic(
        id="seguridad",
        name="Seguridad pública",
        description="Políticas de seguridad: mano dura vs. políticas sociales preventivas",
        prompt_seed="¿Cómo crees que debería enfrentarse la delincuencia en Chile?",
        opinion_dimensions=["eje_izq_der"],
    ),
    "inmigracion": Topic(
        id="inmigracion",
        name="Inmigración",
        description="Política migratoria: apertura vs. restricción",
        prompt_seed="¿Qué opinas de la inmigración en Chile? ¿Hay que restringirla más o ser más abiertos?",
        opinion_dimensions=["eje_izq_der"],
    ),
    "gobierno": Topic(
        id="gobierno",
        name="Evaluación del gobierno",
        description="Evaluación de la gestión del gobierno actual",
        prompt_seed="¿Cómo evalúas la gestión del gobierno actual?",
        opinion_dimensions=["eval_gobierno", "eje_izq_der"],
    ),
    "democracia": Topic(
        id="democracia",
        name="Democracia y autoritarismo",
        description="Preferencia por la democracia vs. formas autoritarias de gobierno",
        prompt_seed="¿Crees que la democracia es siempre la mejor forma de gobierno o hay situaciones donde otro tipo de gobierno podría funcionar mejor?",
        opinion_dimensions=["democracia"],
    ),
    "desigualdad": Topic(
        id="desigualdad",
        name="Desigualdad económica",
        description="Rol del Estado en reducir la desigualdad: más Estado vs. más mercado",
        prompt_seed="¿Qué debería hacer el Estado frente a la desigualdad económica en Chile?",
        opinion_dimensions=["eje_izq_der", "eval_gobierno"],
    ),
    "educacion": Topic(
        id="educacion",
        name="Educación",
        description="Educación pública vs. privada, gratuidad universitaria",
        prompt_seed="¿Qué piensas sobre la educación en Chile? ¿Debería ser gratuita para todos?",
        opinion_dimensions=["eje_izq_der"],
    ),
    "medioambiente": Topic(
        id="medioambiente",
        name="Medio ambiente",
        description="Protección ambiental vs. desarrollo económico",
        prompt_seed="¿Qué tan importante es proteger el medio ambiente aunque implique menos crecimiento económico?",
        opinion_dimensions=["eje_izq_der"],
    ),
}


def get_topic(topic_id: str) -> Topic:
    if topic_id not in TOPICS:
        raise ValueError(f"Tema no encontrado: {topic_id}. Disponibles: {list(TOPICS.keys())}")
    return TOPICS[topic_id]


def get_random_topic(rng) -> Topic:
    import random
    keys = list(TOPICS.keys())
    return TOPICS[keys[rng.integers(0, len(keys))]]
