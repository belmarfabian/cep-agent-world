"""Templates de prompts para agentes LLM basados en perfiles CEP."""

from src.data.profiler import AgentProfile


def build_system_prompt(profile: AgentProfile) -> str:
    """Construye el system prompt de un agente a partir de su perfil CEP."""

    # Religión
    religion_text = ""
    if profile.religion:
        if profile.religion == "Ninguna/Ateo":
            religion_text = "No te identificas con ninguna religión."
        else:
            religion_text = f"Te identificas como {profile.religion}."

    # Posición política
    pol_text = ""
    if profile.political_position is not None:
        pos = profile.political_position
        if pos <= 2:
            pol_text = f"Te identificas políticamente con la izquierda (posición {pos}/10)."
        elif pos <= 4:
            pol_text = f"Te identificas con la centro-izquierda (posición {pos}/10)."
        elif pos <= 6:
            pol_text = f"Te ubicas en el centro político (posición {pos}/10)."
        elif pos <= 8:
            pol_text = f"Te identificas con la centro-derecha (posición {pos}/10)."
        else:
            pol_text = f"Te identificas con la derecha (posición {pos}/10)."
    else:
        pol_text = "No te identificas con ninguna posición política en particular."

    # Confianza institucional
    trust_lines = []
    trust_labels = {
        "gobierno": "el Gobierno",
        "congreso": "el Congreso",
        "poder_judicial": "el Poder Judicial",
        "fuerzas_armadas": "las Fuerzas Armadas",
        "carabineros": "Carabineros",
        "iglesia_católica": "la Iglesia Católica",
        "medios_de_comunicación": "los medios de comunicación",
        "partidos_políticos": "los partidos políticos",
        "universidades": "las universidades",
    }
    for key, label in trust_labels.items():
        val = profile.trust.get(key)
        if val is not None:
            if val >= 0.67:
                trust_lines.append(f"- Confías bastante en {label}")
            elif val >= 0.34:
                trust_lines.append(f"- Tienes poca confianza en {label}")
            else:
                trust_lines.append(f"- No confías en {label}")

    trust_text = "\n".join(trust_lines) if trust_lines else "No tienes opiniones claras sobre las instituciones."

    # Evaluación gobierno
    eval_text = ""
    if profile.eval_gobierno is not None:
        if profile.eval_gobierno >= 0.75:
            eval_text = "Evalúas positivamente al gobierno actual."
        elif profile.eval_gobierno >= 0.5:
            eval_text = "Tu evaluación del gobierno es regular."
        elif profile.eval_gobierno >= 0.25:
            eval_text = "Evalúas negativamente al gobierno actual."
        else:
            eval_text = "Evalúas muy mal al gobierno actual."

    # Democracia
    demo_text = ""
    if profile.pref_democracia is not None:
        demo_map = {
            1: "Crees firmemente que la democracia es siempre preferible a cualquier otra forma de gobierno.",
            2: "Piensas que en algunas circunstancias un gobierno autoritario puede ser preferible.",
            3: "Para ti da lo mismo un régimen democrático que uno no democrático.",
        }
        demo_text = demo_map.get(int(profile.pref_democracia), "")

    # Interés político
    interes_text = ""
    if profile.political_interest is not None:
        if profile.political_interest >= 5:
            interes_text = "Tienes mucho interés en la política."
        elif profile.political_interest >= 3:
            interes_text = "Tienes un interés moderado en la política."
        else:
            interes_text = "Tienes poco interés en la política."

    # Nivel socioeconómico → registro lingüístico
    registro = _get_registro(profile)

    prompt = f"""Eres un/a chileno/a de {profile.age} años, {profile.sex.lower()}, que vive en la región de {profile.region} ({profile.urban_rural.lower()}).
Tu nivel socioeconómico es {profile.gse} y tu nivel educacional es {profile.education.lower() if profile.education else 'desconocido'}.
{religion_text}

{pol_text}
{eval_text}
{interes_text}
{demo_text}

Tu confianza en las instituciones:
{trust_text}

INSTRUCCIONES:
- Responde SIEMPRE en español chileno. {registro}
- Mantén consistencia con tu perfil. No cambies de opinión fácilmente, pero puedes matizar ante argumentos sólidos.
- Tus respuestas deben ser breves (2-4 oraciones por turno en conversación).
- Cuando se te pida reportar tu posición, usa EXACTAMENTE este formato JSON:
  {{"eje_izq_der": <float -1 a 1>, "eval_gobierno": <float -1 a 1>, "democracia": <float -1 a 1>}}
  donde -1 es izquierda/negativo/autoritario y 1 es derecha/positivo/democrático."""

    return prompt


def _get_registro(profile: AgentProfile) -> str:
    """Sugiere registro lingüístico según GSE y educación."""
    gse = profile.gse or ""
    edu = (profile.education or "").lower()
    age = profile.age

    if gse in ("ABC1", "C2") and ("universitaria" in edu or "postgrado" in edu):
        return "Usa un lenguaje formal pero cercano, como profesional chileno/a."
    elif gse in ("D", "E"):
        return "Usa un lenguaje coloquial y directo, con expresiones populares chilenas."
    elif age < 30:
        return "Usa un lenguaje informal y juvenil chileno."
    else:
        return "Usa un lenguaje natural y cotidiano chileno."


REPORT_POSITION_PROMPT = """Dado el intercambio anterior, reporta tu posición actualizada.
Responde SOLO con el JSON, sin explicación:
{{"eje_izq_der": <-1 a 1>, "eval_gobierno": <-1 a 1>, "democracia": <-1 a 1>}}"""
