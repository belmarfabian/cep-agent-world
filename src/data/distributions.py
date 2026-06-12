"""Extrae distribuciones estadísticas de datos CEP para la Opción B (ABM)."""

import numpy as np
import pandas as pd
from scipy import stats

from src.data.cleaner import clean_cep95, get_trust_columns


def extract_opinion_distributions(df: pd.DataFrame) -> dict[str, dict]:
    """
    Extrae distribuciones marginales de variables de opinión por segmento.

    Returns:
        dict[variable_name, {
            "overall": {"mean": ..., "std": ..., "histogram": ...},
            "by_gse": {gse: {"mean": ..., "std": ...}},
            "by_age_group": {...},
            "by_political_position": {...},
        }]
    """
    df = clean_cep95(df)

    # Grupos de edad
    df["age_group"] = pd.cut(
        df["edad"], bins=[17, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"]
    )

    # Grupos políticos
    df["pol_group"] = pd.cut(
        df["posicion_politica"], bins=[-0.1, 3, 5, 7, 10.1],
        labels=["Izquierda", "Centro-izq", "Centro-der", "Derecha"]
    )

    opinion_vars = {
        "posicion_politica": "Posición política (0-10)",
        "eval_gobierno": "Evaluación gobierno (0-1)",
        "satisfaccion_vida": "Satisfacción vida (0-10)",
    }
    trust_cols = get_trust_columns(df)
    for c in trust_cols:
        opinion_vars[c] = f"Confianza: {c.replace('trust_', '')}"

    distributions = {}
    for var, desc in opinion_vars.items():
        if var not in df.columns:
            continue
        data = df[var].dropna()
        if len(data) < 10:
            continue

        dist = {
            "description": desc,
            "overall": {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "median": float(data.median()),
                "n": int(len(data)),
            },
            "by_gse": _group_stats(df, var, "gse_label"),
            "by_age_group": _group_stats(df, var, "age_group"),
            "by_political_position": _group_stats(df, var, "pol_group"),
        }
        distributions[var] = dist

    return distributions


def _group_stats(df: pd.DataFrame, var: str, group_col: str) -> dict:
    """Estadísticas por grupo."""
    result = {}
    if group_col not in df.columns:
        return result
    for name, group in df.groupby(group_col, observed=True):
        data = group[var].dropna()
        if len(data) >= 5:
            result[str(name)] = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "n": int(len(data)),
            }
    return result


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de correlación entre variables actitudinales."""
    df = clean_cep95(df)
    trust_cols = get_trust_columns(df)
    vars_of_interest = [
        "posicion_politica", "eval_gobierno", "interes_politico",
        "satisfaccion_vida", "polarizacion_izq", "polarizacion_der",
    ] + trust_cols
    available = [c for c in vars_of_interest if c in df.columns]
    return df[available].corr()


def compute_homophily_index(df: pd.DataFrame) -> dict:
    """
    Calcula índice de homofilia observado: correlación entre
    similitud demográfica y similitud de opinión.
    """
    df = clean_cep95(df)
    sample = df.dropna(subset=["posicion_politica", "gse", "edad"]).head(500)

    if len(sample) < 50:
        return {"index": 0.0, "n_pairs": 0}

    pos = sample["posicion_politica"].values
    gse = sample["gse"].values
    age = sample["edad"].values

    # Muestrear pares
    n = len(sample)
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=2000)
    idx_b = rng.integers(0, n, size=2000)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    opinion_dist = np.abs(pos[idx_a] - pos[idx_b])
    demo_dist = (
        np.abs(gse[idx_a] - gse[idx_b]) / 4.0
        + np.abs(age[idx_a] - age[idx_b]) / 80.0
    ) / 2.0

    corr, pval = stats.pearsonr(demo_dist, opinion_dist)
    return {"index": float(corr), "p_value": float(pval), "n_pairs": len(idx_a)}
