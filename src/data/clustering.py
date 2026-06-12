"""Clustering de respondentes CEP para generar arquetipos (Opción C)."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data.cleaner import clean_cep95, get_trust_columns
from src.data.profiler import AgentProfile


# Variables numéricas para clustering
CLUSTER_VARS = [
    "posicion_politica",
    "eval_gobierno",
    "interes_politico",
    "satisfaccion_vida",
    "polarizacion_izq",
    "polarizacion_der",
    "edad",
]


def prepare_clustering_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepara matriz numérica para clustering desde datos CEP limpios."""
    df = clean_cep95(df)
    trust_cols = get_trust_columns(df)
    all_vars = CLUSTER_VARS + trust_cols

    # Filtrar columnas que existen
    available = [c for c in all_vars if c in df.columns]
    matrix = df[available].copy()

    # GSE como numérico
    if "gse" in df.columns:
        matrix["gse_num"] = df["gse"]
        available.append("gse_num")

    # Educación como numérico
    if "educacion_104_a" in df.columns:
        matrix["educ_num"] = df["educacion_104_a"]
        available.append("educ_num")

    return matrix, available


def find_archetypes(
    df: pd.DataFrame,
    n_clusters: int = 6,
    random_state: int = 42,
) -> dict:
    """
    Encuentra arquetipos de chilenos mediante K-means.

    Returns:
        dict con claves:
            - "labels": array de cluster labels por respondente
            - "archetypes": dict[int, dict] con descripción de cada arquetipo
            - "silhouette": score de silueta
            - "scaler": StandardScaler fitted
            - "model": KMeans fitted
    """
    matrix, var_names = prepare_clustering_matrix(df)

    # Imputar NAs con KNN
    imputer = KNNImputer(n_neighbors=5)
    matrix_imputed = pd.DataFrame(
        imputer.fit_transform(matrix),
        columns=matrix.columns,
        index=matrix.index,
    )

    # Normalizar
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix_imputed)

    # K-Means
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = model.fit_predict(matrix_scaled)

    sil = silhouette_score(matrix_scaled, labels)

    # Describir cada arquetipo
    matrix_imputed["cluster"] = labels
    archetypes = {}
    for k in range(n_clusters):
        cluster_data = matrix_imputed[matrix_imputed["cluster"] == k]
        centroid = cluster_data.drop(columns=["cluster"]).mean()
        std = cluster_data.drop(columns=["cluster"]).std()

        # Auto-generar nombre descriptivo
        name = _generate_archetype_name(centroid)

        archetypes[k] = {
            "name": name,
            "centroid": centroid.to_dict(),
            "std": std.to_dict(),
            "n_respondents": len(cluster_data),
            "proportion": len(cluster_data) / len(matrix_imputed),
        }

    return {
        "labels": labels,
        "archetypes": archetypes,
        "silhouette": sil,
        "scaler": scaler,
        "model": model,
        "var_names": list(matrix_imputed.columns[:-1]),
    }


def _generate_archetype_name(centroid: pd.Series) -> str:
    """Genera un nombre descriptivo para un arquetipo."""
    parts = []

    # Posición política
    pos = centroid.get("posicion_politica", 5)
    if pos < 3.5:
        parts.append("Izquierda")
    elif pos < 4.5:
        parts.append("Centro-izquierda")
    elif pos < 5.5:
        parts.append("Centro")
    elif pos < 6.5:
        parts.append("Centro-derecha")
    else:
        parts.append("Derecha")

    # Edad
    age = centroid.get("edad", 45)
    if age < 35:
        parts.append("joven")
    elif age < 55:
        parts.append("adulto/a")
    else:
        parts.append("mayor")

    # NSE
    gse = centroid.get("gse_num", 3)
    if gse <= 2:
        parts.append("NSE alto")
    elif gse <= 3:
        parts.append("NSE medio")
    else:
        parts.append("NSE bajo")

    return ", ".join(parts)


def assign_archetypes(profiles: list[AgentProfile], result: dict) -> list[AgentProfile]:
    """Asigna archetype_id a cada perfil."""
    for i, profile in enumerate(profiles):
        if i < len(result["labels"]):
            profile.archetype_id = int(result["labels"][i])
    return profiles


def evaluate_k_range(
    df: pd.DataFrame,
    k_min: int = 3,
    k_max: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evalúa distintos K y retorna métricas."""
    matrix, _ = prepare_clustering_matrix(df)
    imputer = KNNImputer(n_neighbors=5)
    matrix_imputed = imputer.fit_transform(matrix)
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix_imputed)

    results = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(matrix_scaled)
        sil = silhouette_score(matrix_scaled, labels)
        inertia = model.inertia_
        results.append({"k": k, "silhouette": sil, "inertia": inertia})

    return pd.DataFrame(results)
