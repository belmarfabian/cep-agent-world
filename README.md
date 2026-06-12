# CEP Agent World

Mundo simulado de agentes basado en la Encuesta CEP (Chile). Construye una población sintética de 120 agentes a partir de los microdatos de la Encuesta CEP N°95 (2025) y simula dinámicas de opinión sobre temas públicos (pensiones, seguridad, inmigración, evaluación de gobierno, desigualdad, educación, medio ambiente) mediante tres motores:

- **ABM** (`src/agents/abm_agent.py`): modelo de confianza acotada (bounded confidence) con homofilia demográfica, apertura al cambio derivada del perfil sociodemográfico y redes de mundo pequeño.
- **LLM** (`src/agents/llm_agent.py`): agentes conversacionales que encarnan perfiles reales de la encuesta usando la API de Anthropic.
- **Híbrido** (`src/agents/hybrid_agent.py`): combinación de ambos.

## Demo

El dashboard interactivo está en [`data/processed/simulacion_interactiva.html`](data/processed/simulacion_interactiva.html) — se abre directamente en el navegador, sin servidor. Los 1.217 encuestados aparecen sobre un mapa de Chile (rotado, norte a la izquierda) en su región de residencia; al iniciar, toman posición sobre el tema elegido y migran a dos polos de opinión mientras interactúan. Al final se compara la distribución simulada con el margen oficial CEP de la misma pregunta.

## Método

1. **Población sintética.** Cada agente corresponde a un encuestado real de la CEP N°95: edad, sexo, región, NSE (GSE), educación, religión, autoubicación política (escala 1–10) e interés en política.
2. **Opinión inicial.** Para cada tema, la opinión del agente (−1 a +1) se deriva de su autoubicación política, más ruido aleatorio que representa la variabilidad individual por tema.
3. **Red social.** Los agentes se conectan en una red de mundo pequeño (Watts-Strogatz, k=6, p=0.12): la mayoría de los lazos une a personas cercanas y unos pocos atajos cruzan el país.
4. **Dinámica de confianza acotada** (Deffuant-Weisbuch). En cada paso, pares conectados comparan opiniones y solo se influencian si distan menos que la tolerancia ε (0.40 por defecto). El ajuste μ (0.35) se pondera por:
   - **homofilia**: similitud de región, edad y NSE entre los dos agentes;
   - **apertura al cambio**: mayor en jóvenes y universitarios, menor en mayores y muy interesados en política.
5. **Validación.** Tras 60 pasos, la distribución agregada se compara con el margen oficial publicado por el CEP para la misma pregunta.

Los agentes LLM (`SIMULATION_TYPE=llm` o `hybrid`) reemplazan la regla de actualización numérica por conversaciones generadas con la API de Anthropic, donde cada agente argumenta desde su perfil sociodemográfico.

## Estructura

```
src/
├── data/          # carga, limpieza, perfiles y clustering de la encuesta
├── agents/        # agentes ABM, LLM e híbridos
├── world/         # mundos de simulación y topologías de red
├── interactions/  # conversaciones, temas, influencia y prompts
├── analysis/      # métricas y visualización
└── utils/         # configuración y cliente LLM
scripts/
├── run_simulation.py      # simulación principal
├── build_interactive.py   # genera el dashboard HTML
├── simulate_visual.py     # visualizaciones estáticas
├── simulate_gif.py        # animaciones GIF
└── simulate_galton.py     # animación tipo tablero de Galton
```

## Uso

Requiere Python ≥ 3.12.

```bash
pip install -e .
cp .env.example .env       # agregar ANTHROPIC_API_KEY si se usan agentes LLM
python scripts/run_simulation.py
python scripts/build_interactive.py   # regenera el dashboard
```

Los datos crudos (`data/raw/`, microdatos CEP) no se versionan; descargar la Encuesta CEP N°95 desde [cepchile.cl](https://www.cepchile.cl/) y dejar `cep95.csv` en `data/raw/encuesta_95/bases/`.
