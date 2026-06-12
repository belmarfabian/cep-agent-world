"""Genera un GIF animado de la simulacion ante una pregunta especifica."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from PIL import Image
import io

from src.data.loader import load_cep
from src.data.profiler import build_profiles
from src.agents.abm_agent import ABMAgent
from src.world.topology import build_network, get_neighbors

# ── Config ─────────────────────────────────────────────
N_AGENTS = 200
N_STEPS = 60
TOPIC = "pensiones"
QUESTION = "Deberian eliminarse las AFP y pasar a un sistema de reparto estatal?"

print(f"Pregunta: {QUESTION}")
print(f"Tema de simulacion: {TOPIC}")
print(f"Agentes: {N_AGENTS} | Pasos: {N_STEPS}")
print()

# ── Cargar datos y crear agentes ───────────────────────
print("Cargando datos CEP 95...")
df = load_cep("data/raw/encuesta_95/bases/cep95.csv")
profiles = build_profiles(df, max_agents=N_AGENTS)

agents = []
rng_init = np.random.default_rng(42)
for p in profiles:
    a = ABMAgent(p, epsilon=0.4, mu=0.35, noise_sigma=0.015)
    # Opiniones iniciales sobre pensiones derivadas de posicion politica + ruido
    base = a.profile.current_opinions.get("eje_izq_der", 0.0)
    # Izquierda tiende a reparto (-1), derecha a AFP (+1)
    pension_opinion = base + rng_init.normal(0, 0.3)
    a.profile.current_opinions[TOPIC] = float(np.clip(pension_opinion, -1, 1))
    agents.append(a)

# Red con homofilia para mas realismo
print("Construyendo red...")
network = build_network(agents, "small_world", k=6, p=0.12, seed=42)
pos = nx.spring_layout(network, seed=42, k=2.0 / np.sqrt(len(agents)), iterations=80)

print(f"Red: {network.number_of_nodes()} nodos, {network.number_of_edges()} aristas")

# ── Colores ────────────────────────────────────────────
# Azul = a favor de reparto (izq), Rojo = a favor de AFP (der)
cmap = LinearSegmentedColormap.from_list("opinion", [
    "#1a5276",  # -1: fuertemente pro reparto
    "#2e86c1",  # -0.5
    "#d5d8dc",  # 0: indeciso
    "#e74c3c",  # 0.5
    "#922b21",  # +1: fuertemente pro AFP
])
norm = Normalize(vmin=-1, vmax=1)

# ── Simulacion + captura de frames ────────────────────
print("Simulando y capturando frames...")
rng = np.random.default_rng(123)
frames_data = []

# Guardar estado inicial
frames_data.append([a.get_opinion(TOPIC) for a in agents])

for step in range(N_STEPS):
    order = rng.permutation(len(agents))
    changes_this_step = 0
    for i in order:
        neighbors = get_neighbors(network, i, agents)
        if neighbors:
            result = agents[i].step(neighbors, TOPIC)
            if result.get("changed"):
                changes_this_step += 1

    opinions = [a.get_opinion(TOPIC) for a in agents]
    frames_data.append(opinions)

    if step % 10 == 0:
        mean_op = np.mean(opinions)
        std_op = np.std(opinions)
        print(f"  Step {step:3d} | Media: {mean_op:+.3f} | Std: {std_op:.3f} | Cambios: {changes_this_step}")

print(f"\nGenerando GIF con {len(frames_data)} frames...")

# ── Generar frames como imagenes ──────────────────────
images = []
node_x = [pos[i][0] for i in range(len(agents))]
node_y = [pos[i][1] for i in range(len(agents))]

# Precomputar aristas
edge_segments = []
for e0, e1 in network.edges():
    edge_segments.append(([pos[e0][0], pos[e1][0]], [pos[e0][1], pos[e1][1]]))

# Tamanos de nodo segun edad (mayores = mas grandes)
node_sizes = []
for a in agents:
    base_size = 30
    if a.profile.age > 60:
        base_size = 50
    elif a.profile.age > 45:
        base_size = 40
    elif a.profile.age < 30:
        base_size = 25
    node_sizes.append(base_size)

for frame_idx, opinions in enumerate(frames_data):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    # Aristas
    for ex, ey in edge_segments:
        ax.plot(ex, ey, color="#252535", linewidth=0.3, alpha=0.5)

    # Nodos
    colors = [cmap(norm(o)) for o in opinions]
    ax.scatter(node_x, node_y, c=opinions, cmap=cmap, norm=norm,
               s=node_sizes, edgecolors="#444", linewidths=0.4, zorder=5)

    # Info del frame
    step = frame_idx
    mean_op = np.mean(opinions)
    std_op = np.std(opinions)

    # Titulo
    ax.set_title(QUESTION, fontsize=13, color="#e0e0e0", fontweight="bold",
                 pad=15, fontfamily="sans-serif")

    # Subtitulo con metricas
    pro_reparto = sum(1 for o in opinions if o < -0.3)
    indeciso = sum(1 for o in opinions if -0.3 <= o <= 0.3)
    pro_afp = sum(1 for o in opinions if o > 0.3)

    subtitle = f"Step {step}/{N_STEPS}  |  Pro reparto: {pro_reparto}  |  Indecisos: {indeciso}  |  Pro AFP: {pro_afp}"
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
            fontsize=10, color="#888", fontfamily="sans-serif")

    # Barra inferior con distribucion
    hist_ax = fig.add_axes([0.15, 0.02, 0.7, 0.08], facecolor="#0f1117")
    hist_ax.hist(opinions, bins=30, range=(-1, 1), color="#e94560", alpha=0.8, edgecolor="none")
    hist_ax.set_xlim(-1, 1)
    hist_ax.set_yticks([])
    hist_ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    hist_ax.set_xticklabels(["Reparto", "", "Indeciso", "", "AFP"],
                            fontsize=8, color="#888")
    hist_ax.tick_params(colors="#555", length=3)
    for spine in hist_ax.spines.values():
        spine.set_visible(False)
    hist_ax.axvline(mean_op, color="#ffd700", linewidth=1.5, linestyle="--", alpha=0.8)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, aspect=30)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Reparto\nestatal", "Indeciso", "Mantener\nAFP"])
    cbar.ax.tick_params(colors="#888", labelsize=8)
    cbar.outline.set_edgecolor("#333")

    ax.set_xlim(min(node_x) - 0.15, max(node_x) + 0.15)
    ax.set_ylim(min(node_y) - 0.15, max(node_y) + 0.15)
    ax.axis("off")

    # Guardar frame como imagen en memoria
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0f1117", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    images.append(Image.open(buf).copy())
    buf.close()

    if frame_idx % 10 == 0:
        print(f"  Frame {frame_idx}/{len(frames_data)}")

# ── Crear GIF ─────────────────────────────────────────
output_path = Path("data/processed/simulacion_pensiones.gif")
# Repetir ultimo frame para pausa al final
images_with_pause = images + [images[-1]] * 8

images_with_pause[0].save(
    output_path,
    save_all=True,
    append_images=images_with_pause[1:],
    duration=200,  # ms por frame
    loop=0,
)

print(f"\nGIF generado: {output_path.resolve()}")
print(f"  {len(images)} frames, {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# Abrir
import webbrowser
webbrowser.open(str(output_path.resolve()))
