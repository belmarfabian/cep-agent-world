"""Métricas de análisis para evaluar las simulaciones."""

import numpy as np
from scipy import stats


def polarization_variance(opinions: list[float]) -> float:
    """Polarización medida como varianza de opiniones."""
    return float(np.var(opinions))


def polarization_bimodality(opinions: list[float]) -> float:
    """
    Coeficiente de bimodalidad de Sarle.
    > 0.555 sugiere distribución bimodal (polarización).
    """
    arr = np.array(opinions)
    n = len(arr)
    if n < 4:
        return 0.0
    skew = stats.skew(arr)
    kurt = stats.kurtosis(arr, fisher=True)
    bc = (skew**2 + 1) / (kurt + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))
    return float(bc)


def esteban_ray_index(opinions: list[float], alpha: float = 1.6, K: float = 1.0) -> float:
    """
    Índice de polarización Esteban-Ray.
    Mide qué tan agrupadas y separadas están las opiniones.
    """
    arr = np.array(opinions)
    n = len(arr)
    if n < 2:
        return 0.0

    # Discretizar en bins para eficiencia
    n_bins = 20
    hist, edges = np.histogram(arr, bins=n_bins, range=(-1, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    probs = hist / n

    er = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            er += K * probs[i]**(1 + alpha) * probs[j] * abs(centers[i] - centers[j])
    return float(er)


def convergence_delta(history: list[dict], topic: str) -> list[float]:
    """Delta promedio de opiniones entre steps sucesivos."""
    deltas = []
    for i in range(1, len(history)):
        prev = history[i - 1].get(f"mean_{topic}", 0)
        curr = history[i].get(f"mean_{topic}", 0)
        deltas.append(abs(curr - prev))
    return deltas


def ks_test_vs_observed(simulated: list[float], observed: list[float]) -> dict:
    """
    Test Kolmogorov-Smirnov: distribución simulada vs. observada.
    p > 0.05 sugiere que las distribuciones son compatibles.
    """
    stat, pval = stats.ks_2samp(simulated, observed)
    return {"ks_statistic": float(stat), "p_value": float(pval)}


def opinion_clusters(opinions: list[float], n_clusters: int = 3) -> dict:
    """Número efectivo de clusters de opinión."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    arr = np.array(opinions).reshape(-1, 1)
    if len(arr) < n_clusters + 1:
        return {"n_effective_clusters": 1, "silhouette": 0.0}

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = model.fit_predict(arr)
    sil = silhouette_score(arr, labels) if len(set(labels)) > 1 else 0.0

    return {
        "n_effective_clusters": len(set(labels)),
        "silhouette": float(sil),
        "cluster_centers": model.cluster_centers_.flatten().tolist(),
    }
