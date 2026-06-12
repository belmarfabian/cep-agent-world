# CEP Agent World

Mundo simulado de agentes basado en la Encuesta CEP (Chile). Construye una población sintética de 120 agentes a partir de los microdatos de la Encuesta CEP N°95 (2025) y simula dinámicas de opinión sobre temas públicos (pensiones, seguridad, inmigración, evaluación de gobierno, desigualdad, educación, medio ambiente) mediante tres motores:

- **ABM** (`src/agents/abm_agent.py`): modelo de confianza acotada (bounded confidence) con homofilia demográfica, apertura al cambio derivada del perfil sociodemográfico y redes de mundo pequeño.
- **LLM** (`src/agents/llm_agent.py`): agentes conversacionales que encarnan perfiles reales de la encuesta usando la API de Anthropic.
- **Híbrido** (`src/agents/hybrid_agent.py`): combinación de ambos.

## Demo

El dashboard interactivo está en [`data/processed/simulacion_interactiva.html`](data/processed/simulacion_interactiva.html) — se abre directamente en el navegador, sin servidor. Permite elegir tema, ajustar tolerancia (ε) e influencia (μ), y ver la transición de la red social a la distribución de opiniones agente por agente.

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
