"""Carga de datos CEP desde distintos formatos."""

from pathlib import Path

import pandas as pd


# Mapeo de códigos a etiquetas para las variables CEP clave
REGION_LABELS = {
    1: "Tarapacá", 2: "Antofagasta", 3: "Atacama", 4: "Coquimbo",
    5: "Valparaíso", 6: "O'Higgins", 7: "Maule", 8: "Biobío",
    9: "Araucanía", 10: "Los Lagos", 11: "Aysén", 12: "Magallanes",
    13: "Metropolitana", 14: "Los Ríos", 15: "Arica y Parinacota",
    16: "Ñuble",
}

GSE_LABELS = {1: "ABC1", 2: "C2", 3: "C3", 4: "D", 5: "E"}

EDUCATION_LABELS = {
    1: "Sin estudios", 2: "Básica incompleta", 3: "Básica completa",
    4: "Media incompleta", 5: "Media completa", 6: "Técnica incompleta",
    7: "Técnica completa", 8: "Universitaria incompleta",
    9: "Universitaria completa", 10: "Postgrado incompleto",
    11: "Postgrado completo",
}

TRUST_INSTITUTIONS = {
    "confianza_6_a": "Gobierno",
    "confianza_6_b": "Congreso",
    "confianza_6_c": "Poder Judicial",
    "confianza_6_d": "Fuerzas Armadas",
    "confianza_6_e": "Carabineros",
    "confianza_6_f": "Iglesia Católica",
    "confianza_6_g": "Medios de comunicación",
    "confianza_6_h": "Empresas privadas",
    "confianza_6_i": "Sindicatos",
    "confianza_6_j": "Partidos políticos",
    "confianza_6_k": "Tribunal Constitucional",
    "confianza_6_m": "Universidades",
    "confianza_6_n": "Ministerio Público",
    "confianza_6_o": "Contraloría",
    "confianza_6_p": "Banco Central",
    "confianza_6_r": "PDI",
    "confianza_6_s": "Municipalidades",
    "confianza_6_x": "Tribunal Electoral",
    "confianza_6_ab": "Convención Constitucional",
    "confianza_6_ac": "Consejo Constitucional",
}


def load_cep_csv(path: str | Path) -> pd.DataFrame:
    """Carga un archivo CSV de la encuesta CEP."""
    return pd.read_csv(path, low_memory=False)


def load_cep_spss(path: str | Path) -> pd.DataFrame:
    """Carga un archivo SPSS (.sav) preservando etiquetas."""
    import pyreadstat
    df, meta = pyreadstat.read_sav(str(path))
    # Guardar metadatos como atributos del DataFrame
    df.attrs["column_labels"] = meta.column_names_to_labels
    df.attrs["value_labels"] = meta.variable_value_labels
    return df


def load_cep_stata(path: str | Path) -> pd.DataFrame:
    """Carga un archivo Stata (.dta)."""
    return pd.read_stata(str(path))


def load_cep(path: str | Path) -> pd.DataFrame:
    """Auto-detecta formato y carga."""
    path = Path(path)
    loaders = {".csv": load_cep_csv, ".sav": load_cep_spss, ".dta": load_cep_stata}
    loader = loaders.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Formato no soportado: {path.suffix}")
    return loader(path)
