"""Genera un dashboard HTML interactivo de la simulación ABM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import warnings
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.loader import load_cep
from src.data.profiler import build_profiles
from src.agents.abm_agent import ABMAgent
from src.world.topology import build_network, get_neighbors
from src.utils.config import SimulationConfig

warnings.filterwarnings("ignore")

# ── Configuracion ──────────────────────────────────────────
N_AGENTS = 200
N_STEPS = 80
SNAPSHOT_EVERY = 5  # guardar snapshot cada N steps

print(f"Cargando datos CEP 95...")
df = load_cep("data/raw/encuesta_95/bases/cep95.csv")
profiles = build_profiles(df, max_agents=N_AGENTS)
print(f"  {len(profiles)} agentes creados")

# Crear agentes
config = SimulationConfig(n_agents=N_AGENTS, n_steps=N_STEPS)
agents = []
for p in profiles:
    a = ABMAgent(p, epsilon=0.35, mu=0.4, noise_sigma=0.02)
    for topic in ["pensiones", "seguridad", "inmigracion", "gobierno"]:
        if topic not in a.profile.current_opinions:
            base = a.profile.current_opinions.get("eje_izq_der", 0.0)
            noise = np.random.default_rng(hash(a.agent_id) % 10000).normal(0, 0.25)
            a.profile.current_opinions[topic] = float(np.clip(base + noise, -1, 1))
    agents.append(a)

# Red
print("Construyendo red...")
network = build_network(agents, "small_world", k=6, p=0.15, seed=42)
pos = nx.spring_layout(network, seed=42, k=1.5/np.sqrt(len(agents)))

# ── Simulacion ─────────────────────────────────────────────
print(f"Simulando {N_STEPS} pasos...")
topics = ["eje_izq_der", "pensiones", "seguridad", "inmigracion"]
rng = np.random.default_rng(42)

history = {t: {"mean": [], "std": [], "min": [], "max": []} for t in topics}
snapshots = []  # lista de (step, [opiniones por agente])

for step in range(N_STEPS):
    order = rng.permutation(len(agents))
    for i in order:
        neighbors = get_neighbors(network, i, agents)
        if neighbors:
            topic = topics[rng.integers(0, len(topics))]
            agents[i].step(neighbors, topic)

    # Registrar historia
    for t in topics:
        ops = [a.get_opinion(t) for a in agents]
        history[t]["mean"].append(float(np.mean(ops)))
        history[t]["std"].append(float(np.std(ops)))
        history[t]["min"].append(float(np.min(ops)))
        history[t]["max"].append(float(np.max(ops)))

    if step % SNAPSHOT_EVERY == 0 or step == N_STEPS - 1:
        snap = []
        for a in agents:
            snap.append({
                "id": a.agent_id,
                "age": a.profile.age,
                "sex": a.profile.sex,
                "region": a.profile.region,
                "gse": a.profile.gse,
                "pol": a.profile.political_position,
                "religion": a.profile.religion or "N/D",
                "opinions": dict(a.profile.current_opinions),
            })
        snapshots.append({"step": step, "agents": snap})

    if step % 10 == 0:
        print(f"  Step {step}/{N_STEPS}")

print("Simulacion completada. Generando dashboard...")

# ── Generar HTML ───────────────────────────────────────────

# Preparar datos de red
edge_x, edge_y = [], []
for e0, e1 in network.edges():
    x0, y0 = pos[e0]
    x1, y1 = pos[e1]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

node_x = [pos[i][0] for i in range(len(agents))]
node_y = [pos[i][1] for i in range(len(agents))]

# Snapshot final
final_snap = snapshots[-1]["agents"]
final_opinions = [s["opinions"].get("eje_izq_der", 0) for s in final_snap]
initial_snap = snapshots[0]["agents"]
initial_opinions = [s["opinions"].get("eje_izq_der", 0) for s in initial_snap]

# Hover text
hover_texts = []
for s in final_snap:
    pol_str = f"{s['pol']:.0f}/10" if s['pol'] else "N/D"
    ops = s["opinions"]
    hover_texts.append(
        f"<b>{s['id']}</b><br>"
        f"{s['sex']}, {s['age']} anos<br>"
        f"{s['region']} | {s['gse']}<br>"
        f"Pos. politica: {pol_str}<br>"
        f"Religion: {s['religion']}<br>"
        f"<br><b>Opiniones finales:</b><br>"
        + "<br>".join(f"  {k}: {v:+.2f}" for k, v in ops.items())
    )

# ── CREAR DASHBOARD ────────────────────────────────────────

# Frames para animacion de la red
frames = []
for snap in snapshots:
    ops = [s["opinions"].get("eje_izq_der", 0) for s in snap["agents"]]
    frames.append(go.Frame(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines",
                       line=dict(width=0.3, color="#cccccc"), hoverinfo="none"),
            go.Scatter(
                x=node_x, y=node_y, mode="markers",
                marker=dict(size=8, color=ops, colorscale="RdBu_r",
                            cmin=-1, cmax=1, line=dict(width=0.5, color="#333")),
                text=[s["id"] for s in snap["agents"]],
                customdata=[json.dumps(s) for s in snap["agents"]],
                hovertemplate="%{text}<extra></extra>",
            ),
        ],
        name=str(snap["step"]),
        layout=go.Layout(title_text=f"Red de agentes - Step {snap['step']}")
    ))

# Slider steps
slider_steps = []
for snap in snapshots:
    slider_steps.append(dict(
        args=[[str(snap["step"])],
              dict(frame=dict(duration=300, redraw=True), mode="immediate")],
        label=str(snap["step"]),
        method="animate",
    ))

# --- PAGINA HTML COMPLETA ---
html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>CEP Agent World - Simulacion</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px 40px;
            border-bottom: 2px solid #0f3460;
        }}
        .header h1 {{ font-size: 24px; color: #e94560; }}
        .header p {{ color: #a0a0a0; margin-top: 5px; }}
        .stats-bar {{
            display: flex; gap: 20px; padding: 15px 40px;
            background: #16213e; border-bottom: 1px solid #0f3460;
        }}
        .stat-card {{
            background: #1a1a2e; border-radius: 8px; padding: 12px 20px;
            border: 1px solid #0f3460; min-width: 150px;
        }}
        .stat-card .label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
        .stat-card .value {{ font-size: 22px; font-weight: bold; color: #e94560; }}
        .dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; padding: 20px 40px; }}
        .panel {{
            background: #1a1a2e; border-radius: 10px; border: 1px solid #0f3460;
            padding: 10px; min-height: 400px;
        }}
        .panel-title {{ font-size: 14px; color: #e94560; padding: 8px 10px; font-weight: 600; }}
        .full-width {{ grid-column: 1 / -1; }}
        .agent-detail {{
            position: fixed; right: 20px; top: 80px; width: 320px;
            background: #1a1a2e; border: 1px solid #e94560; border-radius: 10px;
            padding: 20px; display: none; z-index: 100;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        .agent-detail h3 {{ color: #e94560; margin-bottom: 10px; }}
        .agent-detail .close {{ cursor: pointer; float: right; color: #888; font-size: 18px; }}
        .legend {{ display: flex; align-items: center; gap: 20px; padding: 5px 40px; color: #888; font-size: 12px; }}
        .legend-item {{ display: flex; align-items: center; gap: 5px; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
    </style>
</head>
<body>

<div class="header">
    <h1>CEP Agent World</h1>
    <p>Simulacion de {N_AGENTS} agentes basados en la Encuesta CEP N 95 (Sept-Oct 2025) | {N_STEPS} pasos | Bounded Confidence</p>
</div>

<div class="stats-bar">
    <div class="stat-card">
        <div class="label">Agentes</div>
        <div class="value">{N_AGENTS}</div>
    </div>
    <div class="stat-card">
        <div class="label">Conexiones</div>
        <div class="value">{network.number_of_edges()}</div>
    </div>
    <div class="stat-card">
        <div class="label">Pos. Politica Media</div>
        <div class="value">{np.mean([a.profile.political_position for a in agents if a.profile.political_position]):.1f}/10</div>
    </div>
    <div class="stat-card">
        <div class="label">Polarizacion Final</div>
        <div class="value">{np.std(final_opinions):.2f}</div>
    </div>
    <div class="stat-card">
        <div class="label">Pasos</div>
        <div class="value">{N_STEPS}</div>
    </div>
</div>

<div class="legend">
    <span>Eje izquierda-derecha:</span>
    <div class="legend-item"><div class="legend-dot" style="background:#2166ac"></div> Izquierda (-1)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#f7f7f7"></div> Centro (0)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#b2182b"></div> Derecha (+1)</div>
</div>

<div class="dashboard">
    <div class="panel">
        <div class="panel-title">Red de Agentes (click en un nodo para ver detalle)</div>
        <div id="network-plot"></div>
    </div>
    <div class="panel">
        <div class="panel-title">Distribucion de Opiniones - Estado Final</div>
        <div id="dist-plot"></div>
    </div>
    <div class="panel full-width">
        <div class="panel-title">Evolucion Temporal de Opiniones</div>
        <div id="evolution-plot"></div>
    </div>
    <div class="panel">
        <div class="panel-title">Opiniones por NSE</div>
        <div id="gse-plot"></div>
    </div>
    <div class="panel">
        <div class="panel-title">Opiniones por Grupo Etario</div>
        <div id="age-plot"></div>
    </div>
</div>

<div class="agent-detail" id="agent-detail">
    <span class="close" onclick="document.getElementById('agent-detail').style.display='none'">&times;</span>
    <h3 id="detail-title"></h3>
    <div id="detail-content"></div>
</div>

<script>
const plotBg = '#1a1a2e';
const paperBg = '#1a1a2e';
const gridColor = '#0f3460';
const textColor = '#a0a0a0';
const accentColor = '#e94560';

const layoutBase = {{
    paper_bgcolor: paperBg,
    plot_bgcolor: plotBg,
    font: {{ color: textColor, family: 'Segoe UI' }},
    margin: {{ l: 50, r: 20, t: 30, b: 50 }},
    xaxis: {{ gridcolor: gridColor, zerolinecolor: gridColor }},
    yaxis: {{ gridcolor: gridColor, zerolinecolor: gridColor }},
}};

// ─── RED DE AGENTES ───────────────────────────────────
const edgeTrace = {{
    x: {json.dumps(edge_x)},
    y: {json.dumps(edge_y)},
    mode: 'lines',
    line: {{ width: 0.4, color: '#333' }},
    hoverinfo: 'none',
    type: 'scatter',
}};

const nodeColors = {json.dumps(final_opinions)};
const agentData = {json.dumps(final_snap, ensure_ascii=False)};

const nodeTrace = {{
    x: {json.dumps(node_x)},
    y: {json.dumps(node_y)},
    mode: 'markers',
    marker: {{
        size: 9,
        color: nodeColors,
        colorscale: 'RdBu',
        reversescale: true,
        cmin: -1, cmax: 1,
        colorbar: {{ title: 'Izq-Der', tickvals: [-1, 0, 1], ticktext: ['Izq', 'Centro', 'Der'], len: 0.5 }},
        line: {{ width: 0.5, color: '#555' }},
    }},
    text: agentData.map(a => a.id),
    customdata: agentData,
    hovertemplate: '<b>%{{text}}</b><br>%{{customdata.sex}}, %{{customdata.age}} anos<br>%{{customdata.region}} | %{{customdata.gse}}<extra></extra>',
    type: 'scatter',
}};

Plotly.newPlot('network-plot', [edgeTrace, nodeTrace], {{
    ...layoutBase,
    xaxis: {{ visible: false }},
    yaxis: {{ visible: false }},
    showlegend: false,
    height: 400,
    margin: {{ l: 10, r: 10, t: 10, b: 10 }},
}}, {{ responsive: true }});

document.getElementById('network-plot').on('plotly_click', function(data) {{
    if (data.points[0].curveNumber === 1) {{
        const agent = data.points[0].customdata;
        const detail = document.getElementById('agent-detail');
        document.getElementById('detail-title').textContent = agent.id;
        const polStr = agent.pol ? (agent.pol + '/10') : 'N/D';
        let opsHtml = '';
        for (const [k, v] of Object.entries(agent.opinions)) {{
            const pct = ((v + 1) / 2 * 100).toFixed(0);
            const barColor = v < 0 ? '#2166ac' : '#b2182b';
            opsHtml += '<div style="margin:4px 0"><span style="display:inline-block;width:100px;font-size:11px">' + k + '</span>' +
                '<div style="display:inline-block;width:150px;height:14px;background:#0f1117;border-radius:3px;vertical-align:middle">' +
                '<div style="width:' + pct + '%;height:100%;background:' + barColor + ';border-radius:3px"></div></div>' +
                ' <span style="font-size:11px">' + v.toFixed(2) + '</span></div>';
        }}
        document.getElementById('detail-content').innerHTML =
            '<p>' + agent.sex + ', ' + agent.age + ' anos</p>' +
            '<p>' + agent.region + ' | ' + agent.gse + '</p>' +
            '<p>Pos. politica: ' + polStr + '</p>' +
            '<p>Religion: ' + agent.religion + '</p>' +
            '<hr style="border-color:#0f3460;margin:10px 0">' +
            '<p style="font-weight:bold;margin-bottom:5px">Opiniones:</p>' + opsHtml;
        detail.style.display = 'block';
    }}
}});

// ─── DISTRIBUCIONES ───────────────────────────────────
const topics = ['eje_izq_der', 'pensiones', 'seguridad', 'inmigracion'];
const topicNames = ['Eje Izq-Der', 'Pensiones', 'Seguridad', 'Inmigracion'];
const colors = ['#e94560', '#00d2ff', '#ffd700', '#00ff88'];

const distTraces = topics.map((t, i) => ({{
    x: agentData.map(a => a.opinions[t] || 0),
    type: 'histogram',
    name: topicNames[i],
    marker: {{ color: colors[i], opacity: 0.7 }},
    xbins: {{ start: -1, end: 1, size: 0.1 }},
}}));

Plotly.newPlot('dist-plot', distTraces, {{
    ...layoutBase,
    barmode: 'overlay',
    xaxis: {{ ...layoutBase.xaxis, title: 'Opinion (-1 a +1)', range: [-1.1, 1.1] }},
    yaxis: {{ ...layoutBase.yaxis, title: 'Frecuencia' }},
    height: 400,
    legend: {{ x: 0, y: 1, bgcolor: 'rgba(0,0,0,0.3)' }},
}}, {{ responsive: true }});

// ─── EVOLUCION TEMPORAL ───────────────────────────────
const historyData = {json.dumps(history)};

const evoTraces = topics.map((t, i) => {{
    const means = historyData[t].mean;
    const stds = historyData[t].std;
    const steps = means.map((_, j) => j);
    return [
        {{
            x: steps,
            y: means.map((m, j) => m + stds[j]),
            mode: 'lines', line: {{ width: 0 }},
            showlegend: false, type: 'scatter',
        }},
        {{
            x: steps,
            y: means.map((m, j) => m - stds[j]),
            mode: 'lines', line: {{ width: 0 }},
            fill: 'tonexty',
            fillcolor: colors[i] + '22',
            showlegend: false, type: 'scatter',
        }},
        {{
            x: steps, y: means,
            mode: 'lines',
            name: topicNames[i],
            line: {{ color: colors[i], width: 2.5 }},
            type: 'scatter',
        }},
    ];
}}).flat();

Plotly.newPlot('evolution-plot', evoTraces, {{
    ...layoutBase,
    xaxis: {{ ...layoutBase.xaxis, title: 'Step' }},
    yaxis: {{ ...layoutBase.yaxis, title: 'Opinion media', range: [-0.6, 0.6] }},
    height: 350,
    legend: {{ orientation: 'h', y: 1.15, bgcolor: 'rgba(0,0,0,0.3)' }},
}}, {{ responsive: true }});

// ─── POR NSE ──────────────────────────────────────────
const gseGroups = ['ABC1', 'C2', 'C3', 'D', 'E'];
const gseData = gseGroups.map(g => agentData.filter(a => a.gse === g).map(a => a.opinions.eje_izq_der || 0));

Plotly.newPlot('gse-plot', gseGroups.map((g, i) => ({{
    y: gseData[i],
    type: 'box',
    name: g,
    marker: {{ color: colors[i % colors.length] }},
    boxmean: true,
}})), {{
    ...layoutBase,
    yaxis: {{ ...layoutBase.yaxis, title: 'Eje Izq-Der', range: [-1.1, 1.1] }},
    height: 380,
    showlegend: false,
}}, {{ responsive: true }});

// ─── POR EDAD ─────────────────────────────────────────
const ageGroups = ['18-30', '31-45', '46-60', '61+'];
const ageBins = [
    a => a.age <= 30,
    a => a.age > 30 && a.age <= 45,
    a => a.age > 45 && a.age <= 60,
    a => a.age > 60,
];
const ageData = ageBins.map(fn => agentData.filter(fn).map(a => a.opinions.eje_izq_der || 0));

Plotly.newPlot('age-plot', ageGroups.map((g, i) => ({{
    y: ageData[i],
    type: 'box',
    name: g,
    marker: {{ color: colors[i % colors.length] }},
    boxmean: true,
}})), {{
    ...layoutBase,
    yaxis: {{ ...layoutBase.yaxis, title: 'Eje Izq-Der', range: [-1.1, 1.1] }},
    height: 380,
    showlegend: false,
}}, {{ responsive: true }});

</script>
</body>
</html>"""

output_path = Path("data/processed/dashboard.html")
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(html_content, encoding="utf-8")
print(f"\nDashboard generado: {output_path.resolve()}")
print("Abriendo en el navegador...")

import webbrowser
webbrowser.open(str(output_path.resolve()))
