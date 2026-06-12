"""Visualización de resultados de simulación."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["figure.dpi"] = 150


def plot_opinion_evolution(history: list[dict], topics: list[str], title: str = ""):
    """Evolución temporal de opiniones medias y std."""
    fig, axes = plt.subplots(len(topics), 1, figsize=(10, 3 * len(topics)), sharex=True)
    if len(topics) == 1:
        axes = [axes]

    for ax, topic in zip(axes, topics):
        steps = [h["step"] for h in history]
        means = [h.get(f"mean_{topic}", 0) for h in history]
        stds = [h.get(f"std_{topic}", 0) for h in history]

        means = np.array(means)
        stds = np.array(stds)

        ax.plot(steps, means, "b-", linewidth=2, label="Media")
        ax.fill_between(steps, means - stds, means + stds, alpha=0.2, color="blue")
        ax.set_ylabel(topic)
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Step")
    fig.suptitle(title or "Evolución de opiniones", fontsize=14)
    plt.tight_layout()
    return fig


def plot_opinion_distribution(agents_snapshot: list[dict], topic: str, title: str = ""):
    """Distribución de opiniones en un momento dado."""
    opinions = [a["opinions"].get(topic, 0) for a in agents_snapshot]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(opinions, bins=30, range=(-1, 1), edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel(f"Opinión: {topic}")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title or f"Distribución de opiniones: {topic}")
    ax.axvline(np.mean(opinions), color="red", linestyle="--", label=f"Media: {np.mean(opinions):.2f}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_opinion_by_group(agents_snapshot: list[dict], topic: str, group_key: str):
    """Boxplot de opiniones por grupo demográfico."""
    df = pd.DataFrame(agents_snapshot)
    df["opinion"] = df["opinions"].apply(lambda x: x.get(topic, 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = df.groupby(group_key)["opinion"].apply(list).to_dict()

    labels = list(groups.keys())
    data = [groups[k] for k in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = sns.color_palette("Set2", len(labels))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel(f"Opinión: {topic}")
    ax.set_xlabel(group_key)
    ax.set_title(f"Opiniones sobre {topic} por {group_key}")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_network(network, agents: list, topic: str = "eje_izq_der"):
    """Visualiza la red con nodos coloreados por opinión."""
    import networkx as nx

    opinions = [a.get_opinion(topic) for a in agents]
    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(network, seed=42)

    nodes = nx.draw_networkx_nodes(
        network, pos, ax=ax,
        node_color=opinions, cmap=plt.cm.RdBu_r,
        node_size=50, vmin=-1, vmax=1,
    )
    nx.draw_networkx_edges(network, pos, ax=ax, alpha=0.1, width=0.5)

    plt.colorbar(nodes, ax=ax, label=f"Opinión: {topic}")
    ax.set_title(f"Red de agentes — {topic}")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_convergence(history: list[dict], metric: str = "change_rate"):
    """Gráfico de convergencia de la simulación."""
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = [h["step"] for h in history]
    values = [h.get(metric, 0) for h in history]

    ax.plot(steps, values, "o-", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(f"Convergencia: {metric}")
    plt.tight_layout()
    return fig
