"""Reglas de influencia para la Opción B (ABM clásico)."""

import numpy as np


def bounded_confidence(
    opinion_i: float,
    opinion_j: float,
    epsilon: float = 0.3,
    mu: float = 0.5,
    homophily_weight: float = 1.0,
) -> float | None:
    """
    Modelo Deffuant-Weisbuch: interacción solo si |opinión_i - opinión_j| < epsilon.

    Returns:
        Nueva opinión para agente i, o None si no hubo interacción.
    """
    delta = abs(opinion_i - opinion_j)
    if delta < epsilon:
        effective_mu = mu * homophily_weight
        return opinion_i + effective_mu * (opinion_j - opinion_i)
    return None


def degroot_update(
    opinion_i: float,
    neighbor_opinions: list[float],
    weights: list[float],
    self_weight: float = 0.5,
) -> float:
    """
    Modelo de DeGroot: opinión = promedio ponderado de vecinos + propia.
    """
    total_weight = self_weight + sum(weights)
    weighted_sum = self_weight * opinion_i
    for op, w in zip(neighbor_opinions, weights):
        weighted_sum += w * op
    return weighted_sum / total_weight


def axelrod_interact(
    culture_i: list[int],
    culture_j: list[int],
    rng: np.random.Generator,
) -> list[int] | None:
    """
    Modelo de Axelrod: interacción proporcional al overlap cultural.
    Si interactúan, copian un rasgo aleatorio.

    Returns:
        Nueva cultura para agente i, o None si no interactuaron.
    """
    n_features = len(culture_i)
    overlap = sum(a == b for a, b in zip(culture_i, culture_j))

    # Probabilidad de interacción proporcional al overlap
    prob = overlap / n_features
    if rng.random() < prob and overlap < n_features:
        # Copiar un rasgo donde difieren
        differ_indices = [k for k in range(n_features) if culture_i[k] != culture_j[k]]
        if differ_indices:
            idx = rng.choice(differ_indices)
            new_culture = list(culture_i)
            new_culture[idx] = culture_j[idx]
            return new_culture
    return None
