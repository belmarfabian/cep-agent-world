"""Configuración global de la simulación."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class SimulationConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # General
    simulation_type: Literal["llm", "abm", "hybrid"] = "hybrid"
    n_agents: int = 100
    n_steps: int = 50
    random_seed: int = 42

    # Data
    cep_data_path: str = "data/raw/encuesta_95/bases/cep95.csv"
    cep_consolidated_path: str = "data/raw/base_consolidada_2010_2025_03112025.csv"

    # Topología
    network_type: Literal["small_world", "homophily", "scale_free"] = "small_world"
    sw_k: int = 6  # vecinos en small-world
    sw_p: float = 0.1  # prob de rewiring

    # LLM (Opciones A y C)
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"
    llm_temperature: float = 0.7
    conversation_turns: int = 4
    max_concurrent_calls: int = 10

    # ABM (Opción B)
    influence_model: Literal["bounded_confidence", "degroot", "axelrod"] = "bounded_confidence"
    epsilon: float = 0.3
    mu: float = 0.5
    noise_sigma: float = 0.01

    # Híbrido (Opción C)
    n_archetypes: int = 6
    drift_correction_strength: float = 0.5

    def data_path(self, relative: str) -> Path:
        return ROOT_DIR / relative
