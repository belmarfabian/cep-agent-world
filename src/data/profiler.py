"""Genera perfiles de agentes a partir de respondentes CEP."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.data.cleaner import clean_cep95, get_trust_columns


@dataclass
class AgentProfile:
    """Perfil demográfico y actitudinal de un agente."""

    agent_id: str
    source_id: int

    # Sociodemográficas
    age: int
    sex: str
    region: str
    urban_rural: str
    education: str
    gse: str
    religion: Optional[str] = None

    # Posición política
    political_position: Optional[float] = None  # 0-10, izq-der
    political_interest: Optional[float] = None   # 1-7

    # Confianza institucional (0-1)
    trust: dict[str, float] = field(default_factory=dict)

    # Evaluación gobierno (0-1)
    eval_gobierno: Optional[float] = None

    # Democracia
    pref_democracia: Optional[int] = None

    # Polarización afectiva (0-10)
    polarizacion_izq: Optional[float] = None
    polarizacion_der: Optional[float] = None

    # Bienestar
    satisfaccion_vida: Optional[float] = None

    # Estado dinámico (muta durante simulación)
    current_opinions: dict[str, float] = field(default_factory=dict)
    archetype_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "age": self.age,
            "sex": self.sex,
            "region": self.region,
            "urban_rural": self.urban_rural,
            "education": self.education,
            "gse": self.gse,
            "religion": self.religion,
            "political_position": self.political_position,
            "political_interest": self.political_interest,
            "eval_gobierno": self.eval_gobierno,
            "pref_democracia": self.pref_democracia,
            "satisfaccion_vida": self.satisfaccion_vida,
            **{f"trust_{k}": v for k, v in self.trust.items()},
        }


def _safe(val):
    """Convierte NaN a None."""
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def build_profiles(df: pd.DataFrame, max_agents: int | None = None) -> list[AgentProfile]:
    """Construye perfiles de agentes desde un DataFrame CEP limpio."""
    df = clean_cep95(df)
    trust_cols = get_trust_columns(df)

    if max_agents:
        df = df.sample(n=min(max_agents, len(df)), random_state=42)

    profiles = []
    for idx, row in df.iterrows():
        trust_dict = {}
        for col in trust_cols:
            val = row.get(col)
            if pd.notna(val):
                inst_name = col.replace("trust_", "")
                trust_dict[inst_name] = float(val)

        profile = AgentProfile(
            agent_id=f"agent_{idx}",
            source_id=int(row.get("id_bu_encuesta", idx)),
            age=int(row["edad"]) if pd.notna(row["edad"]) else 40,
            sex=row.get("sexo_label", "Desconocido"),
            region=row.get("region_label", "Desconocida"),
            urban_rural=row.get("zona_label", "Urbano"),
            education=row.get("educacion_label", "Desconocida"),
            gse=row.get("gse_label", "Desconocido"),
            religion=_safe(row.get("religion_label")),
            political_position=_safe(row.get("posicion_politica")),
            political_interest=_safe(row.get("interes_politico")),
            trust=trust_dict,
            eval_gobierno=_safe(row.get("eval_gobierno")),
            pref_democracia=_safe(row.get("pref_democracia")),
            polarizacion_izq=_safe(row.get("polarizacion_izq")),
            polarizacion_der=_safe(row.get("polarizacion_der")),
            satisfaccion_vida=_safe(row.get("satisfaccion_vida")),
        )

        # Inicializar opiniones dinámicas desde datos CEP
        opinions = {}
        if profile.political_position is not None:
            opinions["eje_izq_der"] = (profile.political_position - 5) / 5  # [-1, 1]
        if profile.eval_gobierno is not None:
            opinions["eval_gobierno"] = profile.eval_gobierno * 2 - 1  # [-1, 1]
        if profile.pref_democracia is not None:
            opinions["democracia"] = {1: 1.0, 2: -0.5, 3: 0.0}.get(
                int(profile.pref_democracia), 0.0
            )
        profile.current_opinions = opinions
        profiles.append(profile)

    return profiles
