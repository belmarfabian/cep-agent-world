"""GIF simple: puntos que toman posicion ante una pregunta."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
import io

from src.data.loader import load_cep
from src.data.profiler import build_profiles
from src.agents.abm_agent import ABMAgent
from src.world.topology import build_network, get_neighbors

# ── Config ─────────────────────────────────────────────
N_AGENTS = 150
N_STEPS = 60
TOPIC = "pensiones"
QUESTION = "Deberian eliminarse las AFP?"
LABEL_IZQ = "Si, reparto estatal"
LABEL_DER = "No, mantener AFP"

# ── Cargar y crear agentes ─────────────────────────────
print("Preparando simulacion...")
df = load_cep("data/raw/encuesta_95/bases/cep95.csv")
profiles = build_profiles(df, max_agents=N_AGENTS)

agents = []
rng_init = np.random.default_rng(42)
for p in profiles:
    a = ABMAgent(p, epsilon=0.4, mu=0.35, noise_sigma=0.015)
    base = a.profile.current_opinions.get("eje_izq_der", 0.0)
    a.profile.current_opinions[TOPIC] = float(np.clip(base + rng_init.normal(0, 0.3), -1, 1))
    agents.append(a)

network = build_network(agents, "small_world", k=6, p=0.12, seed=42)

# Posicion Y fija para cada agente (solo para spread visual)
rng_y = np.random.default_rng(99)
y_positions = rng_y.uniform(0.05, 0.95, size=N_AGENTS)

# ── Colores ────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list("op", ["#2980b9", "#bdc3c7", "#e74c3c"])
norm = Normalize(vmin=-1, vmax=1)

# ── Simular ────────────────────────────────────────────
print(f"Simulando {N_STEPS} pasos...")
rng = np.random.default_rng(123)
all_opinions = []
all_opinions.append([a.get_opinion(TOPIC) for a in agents])

for step in range(N_STEPS):
    order = rng.permutation(len(agents))
    for i in order:
        neighbors = get_neighbors(network, i, agents)
        if neighbors:
            agents[i].step(neighbors, TOPIC)
    all_opinions.append([a.get_opinion(TOPIC) for a in agents])

# ── Generar frames ─────────────────────────────────────
print("Generando frames...")
images = []

for frame_idx, opinions in enumerate(all_opinions):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#111118")
    ax.set_facecolor("#111118")

    ops = np.array(opinions)

    # Puntos: X = opinion, Y = spread
    ax.scatter(
        ops, y_positions,
        c=ops, cmap=cmap, norm=norm,
        s=35, alpha=0.85, edgecolors="#333", linewidths=0.3,
    )

    # Linea central
    ax.axvline(0, color="#444", linewidth=0.8, linestyle="--", alpha=0.5)

    # Media
    mean_val = np.mean(ops)
    ax.axvline(mean_val, color="#ffd700", linewidth=2, alpha=0.8)

    # Conteos
    n_izq = int(np.sum(ops < -0.2))
    n_centro = int(np.sum(np.abs(ops) <= 0.2))
    n_der = int(np.sum(ops > 0.2))

    # Labels
    ax.text(-0.95, 1.05, LABEL_IZQ, transform=ax.transAxes,
            fontsize=11, color="#2980b9", fontweight="bold",
            ha="left", va="bottom", fontfamily="sans-serif")
    ax.text(0.95, 1.05, LABEL_DER, transform=ax.transAxes,
            fontsize=11, color="#e74c3c", fontweight="bold",
            ha="right", va="bottom", fontfamily="sans-serif")

    # Titulo
    ax.set_title(QUESTION, fontsize=16, color="white", fontweight="bold",
                 pad=30, fontfamily="sans-serif")

    # Conteos abajo
    ax.text(-0.7, -0.08, f"{n_izq}", fontsize=22, color="#2980b9",
            fontweight="bold", ha="center", fontfamily="sans-serif")
    ax.text(0, -0.08, f"{n_centro}", fontsize=22, color="#888",
            fontweight="bold", ha="center", fontfamily="sans-serif")
    ax.text(0.7, -0.08, f"{n_der}", fontsize=22, color="#e74c3c",
            fontweight="bold", ha="center", fontfamily="sans-serif")

    # Step indicator
    ax.text(0.98, -0.08, f"t={frame_idx}", fontsize=10, color="#555",
            ha="right", fontfamily="sans-serif")

    # Limpiar ejes
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.15, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Guardar frame
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="#111118", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    images.append(Image.open(buf).copy())
    buf.close()

# ── GIF ────────────────────────────────────────────────
output = Path("data/processed/simulacion_simple.gif")
images_final = images + [images[-1]] * 10  # pausa al final
images_final[0].save(
    output, save_all=True, append_images=images_final[1:],
    duration=180, loop=0,
)
print(f"\nGIF: {output.resolve()} ({output.stat().st_size/1024:.0f} KB)")

import webbrowser
webbrowser.open(str(output.resolve()))
