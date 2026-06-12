"""Limpieza y recodificación de datos CEP."""

import pandas as pd
import numpy as np

from src.data.loader import (
    REGION_LABELS, GSE_LABELS, EDUCATION_LABELS, TRUST_INSTITUTIONS,
)

# Valores que representan missing en la codificación CEP
MISSING_CODES = {-8, -9, 88, 99, 98}


def replace_missing(df: pd.DataFrame, codes: set[int] | None = None) -> pd.DataFrame:
    """Reemplaza códigos de missing CEP con NaN."""
    codes = codes or MISSING_CODES
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df.loc[df[col].isin(codes), col] = np.nan
    return df


def clean_cep95(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de limpieza para la encuesta CEP 95."""
    df = df.copy()
    df = replace_missing(df)

    # Renombrar y recodificar variables clave
    df["sexo_label"] = df["sexo"].map({1: "Hombre", 2: "Mujer"})
    df["region_label"] = df["region_3"].map(REGION_LABELS)
    df["zona_label"] = df["zona_u_r"].map({1: "Urbano", 2: "Rural"})
    df["gse_label"] = df["gse"].map(GSE_LABELS)
    df["educacion_label"] = df["educacion_104_a"].map(EDUCATION_LABELS)

    # Posición política izquierda-derecha (1-10)
    df["posicion_politica"] = df["iden_pol_2"].where(
        df["iden_pol_2"].between(0, 10)
    )

    # Confianza institucional: recodificar a escala 0-1
    # Original: 1=Mucha, 2=Bastante, 3=Poca, 4=Nada → invertir
    for col, label in TRUST_INSTITUTIONS.items():
        if col in df.columns:
            clean_col = f"trust_{label.lower().replace(' ', '_')}"
            # Invertir: 4→0, 3→0.33, 2→0.67, 1→1
            df[clean_col] = df[col].where(df[col].between(1, 4)).map(
                {1: 1.0, 2: 0.67, 3: 0.33, 4: 0.0}
            )

    # Evaluación gobierno: 1=Muy bien ... 5=Muy mal → invertir a 0-1
    df["eval_gobierno"] = df["eval_gob_1"].where(
        df["eval_gob_1"].between(1, 5)
    ).map({1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.0})

    # Interés en política (interes_pol_1_b): 1-7 escala
    df["interes_politico"] = df["interes_pol_1_b"].where(
        df["interes_pol_1_b"].between(1, 7)
    )

    # Democracia vs autoritarismo (democracia_21)
    # 1=Democracia siempre preferible, 2=Autoritarismo a veces, 3=Da lo mismo
    df["pref_democracia"] = df["democracia_21"].where(
        df["democracia_21"].between(1, 3)
    )

    # Religión (religion_14): 1=Católico, 2=Evangélico, 3=Otra, 4=Ninguna
    religion_map = {1: "Católico", 2: "Evangélico", 3: "Otra religión", 4: "Ninguna/Ateo"}
    df["religion_label"] = df["religion_14"].map(religion_map)

    # Polarización afectiva (polarizacion_1_a/b): 0-10
    df["polarizacion_izq"] = df["polarizacion_1_a"].where(
        df["polarizacion_1_a"].between(0, 10)
    )
    df["polarizacion_der"] = df["polarizacion_1_b"].where(
        df["polarizacion_1_b"].between(0, 10)
    )

    # Satisfacción con la vida (bienestar_2): 0-10
    df["satisfaccion_vida"] = df["bienestar_2"].where(
        df["bienestar_2"].between(0, 10)
    )

    return df


def get_trust_columns(df: pd.DataFrame) -> list[str]:
    """Retorna columnas de confianza institucional limpias."""
    return [c for c in df.columns if c.startswith("trust_")]
