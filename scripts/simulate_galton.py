"""GIF: bolitas se sueltan de la red una a una y caen a la distribucion."""
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
import io

from src.data.loader import load_cep
from src.data.profiler import build_profiles
from src.agents.abm_agent import ABMAgent
from src.world.topology import build_network, get_neighbors

# ── Config ─────────────────────────────────────────────
N_AGENTS = 100
N_STEPS = 40
TOPIC = "pensiones"
QUESTION = "Deberian eliminarse las AFP?"
LABEL_IZQ = "Si, reparto"
LABEL_DER = "No, AFP"

N_BINS = 13
BIN_CENTERS = np.linspace(-1, 1, N_BINS + 1)
BIN_CENTERS = (BIN_CENTERS[:-1] + BIN_CENTERS[1:]) / 2
DOT_H = 0.04

CMAP = LinearSegmentedColormap.from_list("op", ["#2980b9", "#d5dbdb", "#e74c3c"])
NORM = Normalize(vmin=-1, vmax=1)

FALL_DURATION = 8       # frames que tarda cada bolita en caer
BALLS_PER_FRAME = 3     # bolitas que se sueltan por frame

# ── Preparar ───────────────────────────────────────────
print("Preparando...")
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
edges = list(network.edges())

# Layout de red en la mitad superior
pos_net = nx.spring_layout(network, seed=42, k=2.0/np.sqrt(N_AGENTS), iterations=80)
raw_x = np.array([pos_net[i][0] for i in range(N_AGENTS)])
raw_y = np.array([pos_net[i][1] for i in range(N_AGENTS)])
# Normalizar red a zona superior: x en [-0.9, 0.9], y en [0.55, 0.95]
net_x = (raw_x - raw_x.min()) / (raw_x.max() - raw_x.min()) * 1.6 - 0.8
net_y = (raw_y - raw_y.min()) / (raw_y.max() - raw_y.min()) * 0.35 + 0.58

initial_ops = [a.get_opinion(TOPIC) for a in agents]

# Posiciones destino (distribucion apilada) en zona inferior
bin_stacks = np.zeros(N_BINS, dtype=int)
dist_x = np.zeros(N_AGENTS)
dist_y = np.zeros(N_AGENTS)

# Orden de caida aleatorio
rng_drop = np.random.default_rng(77)
drop_order = rng_drop.permutation(N_AGENTS).tolist()

# Precalcular destino de cada agente en orden de caida
agent_bin = []
for i in range(N_AGENTS):
    op = initial_ops[i]
    b = np.clip(int((op + 1) / 2 * N_BINS), 0, N_BINS - 1)
    agent_bin.append(b)

bin_stacks_pre = np.zeros(N_BINS, dtype=int)
agent_stack = [0] * N_AGENTS
for i in drop_order:
    b = agent_bin[i]
    agent_stack[i] = bin_stacks_pre[b]
    bin_stacks_pre[b] += 1

for i in range(N_AGENTS):
    dist_x[i] = BIN_CENTERS[agent_bin[i]]
    dist_y[i] = agent_stack[i] * DOT_H + DOT_H / 2 + 0.02

# Frame de inicio de caida por agente
start_frame = [0] * N_AGENTS
for idx, agent_i in enumerate(drop_order):
    start_frame[agent_i] = 8 + idx // BALLS_PER_FRAME  # 8 frames de red estatica

total_drop_frames = max(start_frame) + FALL_DURATION + 5

# ── Simular interacciones ──────────────────────────────
print(f"Simulando {N_STEPS} pasos...")
rng = np.random.default_rng(123)
opinion_history = [list(initial_ops)]
for step in range(N_STEPS):
    order = rng.permutation(len(agents))
    for i in order:
        neighbors = get_neighbors(network, i, agents)
        if neighbors:
            agents[i].step(neighbors, TOPIC)
    opinion_history.append([a.get_opinion(TOPIC) for a in agents])

# ── Easing ─────────────────────────────────────────────
def ease(t):
    return t * t * (3 - 2 * t)

# ── Render ─────────────────────────────────────────────
FIG_W, FIG_H = 8, 7
images = []

def save(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="#111118")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img

print(f"Generando {total_drop_frames} frames de caida...")

for frame in range(total_drop_frames):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="#111118")
    ax.set_facecolor("#111118")

    # Clasificar agentes: en_red, cayendo, aterrizado
    xs, ys, cs = [], [], []

    # Aristas: solo entre agentes que siguen en la red
    in_network = set()
    for i in range(N_AGENTS):
        if frame < start_frame[i]:
            in_network.add(i)

    # Dibujar aristas de la red
    for a_idx, b_idx in edges:
        if a_idx in in_network and b_idx in in_network:
            ax.plot([net_x[a_idx], net_x[b_idx]],
                    [net_y[a_idx], net_y[b_idx]],
                    color="#333", linewidth=0.3, alpha=0.5, zorder=1)

    # Dibujar cada agente
    for i in range(N_AGENTS):
        op = initial_ops[i]
        sf = start_frame[i]

        if frame < sf:
            # Sigue en la red
            xs.append(net_x[i])
            ys.append(net_y[i])
            cs.append(op)
        elif frame < sf + FALL_DURATION:
            # Cayendo
            elapsed = frame - sf
            t = ease(elapsed / FALL_DURATION)
            x = net_x[i] + (dist_x[i] - net_x[i]) * t
            y = net_y[i] + (dist_y[i] - net_y[i]) * t
            xs.append(x)
            ys.append(y)
            cs.append(op)
        else:
            # Aterrizado
            xs.append(dist_x[i])
            ys.append(dist_y[i])
            cs.append(op)

    ax.scatter(xs, ys, c=cs, cmap=CMAP, norm=NORM,
               s=120, alpha=0.9, edgecolors="#222", linewidths=0.4, zorder=5)

    # Linea separadora red/distribucion
    ax.axhline(0.52, color="#292929", linewidth=0.8, linestyle="-", alpha=0.4)

    # Base de distribucion
    ax.axhline(0.02, color="#333", linewidth=1)

    # Labels
    n_landed = sum(1 for i in range(N_AGENTS) if frame >= start_frame[i] + FALL_DURATION)
    landed_ops = [initial_ops[i] for i in range(N_AGENTS) if frame >= start_frame[i] + FALL_DURATION]
    n_izq = sum(1 for o in landed_ops if o < -0.2)
    n_der = sum(1 for o in landed_ops if o > 0.2)
    n_cen = len(landed_ops) - n_izq - n_der

    ax.text(-0.9, -0.03, LABEL_IZQ, fontsize=9, color="#2980b9",
            fontweight="bold", ha="center", fontfamily="sans-serif")
    ax.text(0.9, -0.03, LABEL_DER, fontsize=9, color="#e74c3c",
            fontweight="bold", ha="center", fontfamily="sans-serif")
    ax.text(-0.9, -0.08, str(n_izq), fontsize=16, color="#2980b9",
            fontweight="bold", ha="center")
    ax.text(0.0, -0.08, str(n_cen), fontsize=16, color="#555",
            fontweight="bold", ha="center")
    ax.text(0.9, -0.08, str(n_der), fontsize=16, color="#e74c3c",
            fontweight="bold", ha="center")

    ax.set_title(QUESTION, fontsize=15, color="white", fontweight="bold",
                 pad=8, fontfamily="sans-serif")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.12, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    images.append(save(fig))
    if frame % 10 == 0:
        print(f"  Frame {frame}/{total_drop_frames}")

# Pausa
images += [images[-1]] * 8

# ── Fase interaccion ───────────────────────────────────
print("Fase interaccion...")

def stacked_positions(opinions):
    bins = np.zeros(N_BINS, dtype=int)
    sxs, sys_ = np.zeros(len(opinions)), np.zeros(len(opinions))
    for i, op in enumerate(opinions):
        b = np.clip(int((op + 1) / 2 * N_BINS), 0, N_BINS - 1)
        sxs[i] = BIN_CENTERS[b]
        sys_[i] = bins[b] * DOT_H + DOT_H / 2 + 0.02
        bins[b] += 1
    return sxs, sys_

for step in range(1, len(opinion_history), 2):
    ops = opinion_history[step]
    sx, sy = stacked_positions(ops)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="#111118")
    ax.set_facecolor("#111118")

    ax.scatter(sx, sy, c=ops, cmap=CMAP, norm=NORM,
               s=120, alpha=0.9, edgecolors="#222", linewidths=0.4, zorder=5)

    ax.axhline(0.02, color="#333", linewidth=1)

    n_izq = sum(1 for o in ops if o < -0.2)
    n_der = sum(1 for o in ops if o > 0.2)
    n_cen = len(ops) - n_izq - n_der

    ax.text(-0.9, -0.03, LABEL_IZQ, fontsize=9, color="#2980b9",
            fontweight="bold", ha="center", fontfamily="sans-serif")
    ax.text(0.9, -0.03, LABEL_DER, fontsize=9, color="#e74c3c",
            fontweight="bold", ha="center", fontfamily="sans-serif")
    ax.text(-0.9, -0.08, str(n_izq), fontsize=16, color="#2980b9",
            fontweight="bold", ha="center")
    ax.text(0.0, -0.08, str(n_cen), fontsize=16, color="#555",
            fontweight="bold", ha="center")
    ax.text(0.9, -0.08, str(n_der), fontsize=16, color="#e74c3c",
            fontweight="bold", ha="center")

    ax.set_title(QUESTION, fontsize=15, color="white", fontweight="bold",
                 pad=8, fontfamily="sans-serif")
    ax.text(0.97, 0.97, f"Interaccion {step}/{N_STEPS}", fontsize=9, color="#555",
            ha="right", va="top", transform=ax.transAxes)

    max_y = max(sy) + 0.05
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.12, max(1.02, max_y))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    images.append(save(fig))

# ── GIF ────────────────────────────────────────────────
output = Path("data/processed/red_a_distribucion.gif")
final = images + [images[-1]] * 10
final[0].save(output, save_all=True, append_images=final[1:],
              duration=130, loop=0)
print(f"\nGIF: {output.resolve()} ({output.stat().st_size/1024:.0f} KB)")

import webbrowser
webbrowser.open(str(output.resolve()))
