"""Genera datos JSON + HTML interactivo para compartir."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import networkx as nx

from src.data.loader import load_cep
from src.data.profiler import build_profiles
from src.agents.abm_agent import ABMAgent
from src.world.topology import build_network

N_AGENTS = 120

print("Cargando datos CEP 95...")
df = load_cep("data/raw/encuesta_95/bases/cep95.csv")
profiles = build_profiles(df, max_agents=N_AGENTS)

agents = []
rng = np.random.default_rng(42)
for p in profiles:
    a = ABMAgent(p, epsilon=0.4, mu=0.35, noise_sigma=0.015)
    agents.append(a)

network = build_network(agents, "small_world", k=6, p=0.12, seed=42)
pos = nx.spring_layout(network, seed=42, k=2.0/np.sqrt(N_AGENTS), iterations=80)

# Normalizar posiciones de red
raw_x = np.array([pos[i][0] for i in range(N_AGENTS)])
raw_y = np.array([pos[i][1] for i in range(N_AGENTS)])
net_x = ((raw_x - raw_x.min()) / (raw_x.max() - raw_x.min()) * 1.6 - 0.8).tolist()
net_y = ((raw_y - raw_y.min()) / (raw_y.max() - raw_y.min()) * 0.35 + 0.58).tolist()

# Aristas
edges_list = [[int(a), int(b)] for a, b in network.edges()]

# Perfiles de agentes
agent_data = []
for i, p in enumerate(profiles):
    base_opinion = 0.0
    if p.political_position is not None:
        base_opinion = (p.political_position - 5) / 5

    gse_order = {"ABC1": 1, "C2": 2, "C3": 3, "D": 4, "E": 5}
    gse_num = gse_order.get(p.gse, 3)

    # Openness
    openness = 0.5
    if p.age < 30: openness += 0.15
    elif p.age > 60: openness -= 0.15
    edu = str(p.education or "")
    if "universitaria" in edu.lower() or "postgrado" in edu.lower(): openness += 0.1
    if p.political_interest and p.political_interest > 5: openness -= 0.1
    openness = max(0.1, min(0.9, openness))

    agent_data.append({
        "id": i,
        "age": p.age,
        "sex": p.sex or "?",
        "region": p.region or "?",
        "gse": p.gse or "?",
        "gse_num": gse_num,
        "education": p.education or "?",
        "religion": p.religion or "N/D",
        "pol_position": p.political_position,
        "base_opinion": round(base_opinion, 3),
        "openness": round(openness, 3),
        "net_x": round(net_x[i], 4),
        "net_y": round(net_y[i], 4),
    })

data = {
    "agents": agent_data,
    "edges": edges_list,
    "n_agents": N_AGENTS,
}

print(f"  {N_AGENTS} agentes, {len(edges_list)} aristas")
print("Generando HTML...")

data_json = json.dumps(data, ensure_ascii=False)

html = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>CEP Agent World - Simulacion Interactiva</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0d1117; color:#c9d1d9; font-family:'Segoe UI',system-ui,sans-serif; overflow-x:hidden; }
.header { background:linear-gradient(135deg,#161b22,#0d1117); padding:18px 30px; border-bottom:1px solid #30363d; display:flex; justify-content:space-between; align-items:center; }
.header h1 { font-size:20px; color:#58a6ff; }
.header .subtitle { color:#8b949e; font-size:13px; }
.controls { display:flex; gap:8px; padding:14px 30px; background:#161b22; border-bottom:1px solid #30363d; flex-wrap:wrap; align-items:center; }
.controls label { color:#8b949e; font-size:12px; margin-right:4px; }
.btn { background:#21262d; color:#c9d1d9; border:1px solid #30363d; padding:8px 16px; border-radius:6px; cursor:pointer; font-size:13px; transition:all 0.15s; }
.btn:hover { background:#30363d; border-color:#58a6ff; }
.btn.active { background:#1f6feb; border-color:#1f6feb; color:white; }
.btn.play { background:#238636; border-color:#2ea043; min-width:90px; }
.btn.play:hover { background:#2ea043; }
.btn.reset { background:#da3633; border-color:#f85149; }
.btn.reset:hover { background:#f85149; }
select { background:#21262d; color:#c9d1d9; border:1px solid #30363d; padding:6px 10px; border-radius:6px; font-size:13px; }
input[type=range] { width:100px; accent-color:#58a6ff; }
.main { display:flex; height:calc(100vh - 120px); }
.canvas-area { flex:1; position:relative; }
canvas { width:100%; height:100%; }
.sidebar { width:280px; background:#161b22; border-left:1px solid #30363d; padding:16px; overflow-y:auto; }
.sidebar h3 { color:#58a6ff; font-size:14px; margin-bottom:10px; }
.stat { margin-bottom:12px; }
.stat .label { font-size:11px; color:#8b949e; text-transform:uppercase; }
.stat .value { font-size:20px; font-weight:bold; }
.stat .value.blue { color:#58a6ff; }
.stat .value.red { color:#f85149; }
.stat .value.gray { color:#8b949e; }
.agent-card { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:12px; margin-top:12px; display:none; }
.agent-card h4 { color:#58a6ff; margin-bottom:6px; }
.agent-card p { font-size:12px; color:#8b949e; margin:2px 0; }
.bar-container { display:flex; align-items:center; gap:6px; margin:4px 0; }
.bar-bg { flex:1; height:10px; background:#21262d; border-radius:5px; overflow:hidden; }
.bar-fill { height:100%; border-radius:5px; transition:width 0.3s; }
.bar-label { font-size:10px; color:#8b949e; width:70px; }
.bar-val { font-size:10px; color:#c9d1d9; width:35px; text-align:right; }
.phase-label { position:absolute; top:10px; left:50%; transform:translateX(-50%); color:#8b949e; font-size:13px; background:rgba(13,17,23,0.8); padding:4px 14px; border-radius:12px; pointer-events:none; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>CEP Agent World</h1>
    <div class="subtitle">Simulacion basada en Encuesta CEP N95 (2025) &mdash; 120 agentes reales</div>
  </div>
</div>

<div class="controls">
  <label>Pregunta:</label>
  <select id="topic-select">
    <option value="pensiones" data-q="Deberian eliminarse las AFP?" data-izq="Si, reparto" data-der="No, AFP">Pensiones (AFP vs Reparto)</option>
    <option value="seguridad" data-q="Mano dura contra la delincuencia?" data-izq="No, prevencion" data-der="Si, mano dura">Seguridad publica</option>
    <option value="inmigracion" data-q="Deberia restringirse la inmigracion?" data-izq="No, apertura" data-der="Si, restringir">Inmigracion</option>
    <option value="gobierno" data-q="Como evalua al gobierno?" data-izq="Mal" data-der="Bien">Evaluacion gobierno</option>
    <option value="desigualdad" data-q="Debe el Estado reducir la desigualdad?" data-izq="No, mercado" data-der="Si, Estado">Desigualdad</option>
    <option value="educacion" data-q="Educacion gratuita para todos?" data-izq="No" data-der="Si, gratuita">Educacion</option>
    <option value="medioambiente" data-q="Proteger medioambiente vs crecimiento?" data-izq="Crecimiento" data-der="Medioambiente">Medio ambiente</option>
  </select>

  <button class="btn play" id="btn-play" onclick="togglePlay()">Iniciar</button>
  <button class="btn reset" onclick="resetSim()">Reiniciar</button>

  <label style="margin-left:16px">Velocidad:</label>
  <input type="range" id="speed" min="1" max="10" value="5">

  <label style="margin-left:16px">Tolerancia (epsilon):</label>
  <input type="range" id="epsilon" min="10" max="80" value="40" style="width:80px">
  <span id="eps-val" style="font-size:12px;color:#8b949e">0.40</span>

  <label style="margin-left:16px">Influencia (mu):</label>
  <input type="range" id="mu" min="10" max="80" value="35" style="width:80px">
  <span id="mu-val" style="font-size:12px;color:#8b949e">0.35</span>
</div>

<div class="main">
  <div class="canvas-area">
    <canvas id="canvas"></canvas>
    <div class="phase-label" id="phase-label">Selecciona una pregunta y presiona Iniciar</div>
  </div>
  <div class="sidebar">
    <h3>Estadisticas</h3>
    <div class="stat"><div class="label">Paso</div><div class="value gray" id="s-step">0</div></div>
    <div class="stat"><div class="label">A favor (izq)</div><div class="value blue" id="s-izq">0</div></div>
    <div class="stat"><div class="label">Indecisos</div><div class="value gray" id="s-cen">0</div></div>
    <div class="stat"><div class="label">En contra (der)</div><div class="value red" id="s-der">0</div></div>
    <div class="stat"><div class="label">Media</div><div class="value" id="s-mean" style="color:#c9d1d9">0.00</div></div>
    <div class="stat"><div class="label">Polarizacion (std)</div><div class="value" id="s-std" style="color:#c9d1d9">0.00</div></div>

    <h3 style="margin-top:20px">Agente seleccionado</h3>
    <p style="font-size:11px;color:#484f58">Click en un punto para ver detalle</p>
    <div class="agent-card" id="agent-card">
      <h4 id="ac-name"></h4>
      <p id="ac-demo"></p>
      <p id="ac-pol"></p>
      <p id="ac-rel"></p>
      <div style="margin-top:8px" id="ac-opinion"></div>
    </div>
  </div>
</div>

<script>
const DATA = """ + data_json + r""";

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// State
let agents = [];
let edges = DATA.edges;
let phase = 'idle'; // idle, dropping, simulating
let dropProgress = []; // per agent: 0=in network, 1=landed
let opinions = [];
let simStep = 0;
let animFrame = 0;
let playing = false;
let selectedAgent = -1;

const N = DATA.n_agents;
const N_BINS = 13;
const binEdges = [];
for(let i=0;i<=N_BINS;i++) binEdges.push(-1 + i*2/N_BINS);
const binCenters = [];
for(let i=0;i<N_BINS;i++) binCenters.push((binEdges[i]+binEdges[i+1])/2);

let currentQ = '', labelIzq = '', labelDer = '';

function resize() {
  canvas.width = canvas.parentElement.clientWidth * devicePixelRatio;
  canvas.height = canvas.parentElement.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
}
resize();
window.addEventListener('resize', () => { resize(); draw(); });

// Init agents
function initAgents(topicKey) {
  const sel = document.getElementById('topic-select');
  const opt = sel.options[sel.selectedIndex];
  currentQ = opt.dataset.q;
  labelIzq = opt.dataset.izq;
  labelDer = opt.dataset.der;

  agents = DATA.agents.map(a => ({...a}));
  opinions = new Float64Array(N);
  dropProgress = new Float64Array(N); // all 0

  const rng = mulberry32(42 + topicKey.length * 7);
  for(let i=0; i<N; i++) {
    let base = agents[i].base_opinion;
    let noise = (rng()-0.5) * 0.6;
    opinions[i] = Math.max(-1, Math.min(1, base + noise));
  }

  // Drop order
  const order = [...Array(N).keys()];
  for(let i=N-1;i>0;i--) { const j=Math.floor(rng()*i); [order[i],order[j]]=[order[j],order[i]]; }
  agents.forEach((a,i) => { a.dropOrder = order.indexOf(i); });

  simStep = 0;
  animFrame = 0;
  phase = 'idle';
  selectedAgent = -1;
  document.getElementById('agent-card').style.display='none';
  updateStats();
}

function mulberry32(seed) {
  return function() {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
}

// Stacked positions
function computeStacked(ops) {
  const bins = new Int32Array(N_BINS);
  const xs = new Float64Array(N);
  const ys = new Float64Array(N);
  const dotH = 0.042;
  for(let i=0; i<N; i++) {
    let b = Math.floor((ops[i]+1)/2*N_BINS);
    b = Math.max(0, Math.min(N_BINS-1, b));
    xs[i] = binCenters[b];
    ys[i] = bins[b] * dotH + dotH/2 + 0.02;
    bins[b]++;
  }
  return {xs, ys};
}

// Homophily
function homophily(i, j) {
  let s=0, n=0;
  if(agents[i].region === agents[j].region) s+=1; n++;
  s += 1 - Math.abs(agents[i].gse_num - agents[j].gse_num)/4; n++;
  s += Math.max(0, 1 - Math.abs(agents[i].age - agents[j].age)/40); n++;
  return s/n;
}

// ABM step
function simStepFn() {
  const eps = parseInt(document.getElementById('epsilon').value)/100;
  const mu = parseInt(document.getElementById('mu').value)/100;
  const rng = mulberry32(simStep * 1000 + 7);

  const order = [...Array(N).keys()];
  for(let i=N-1;i>0;i--) { const j=Math.floor(rng()*i); [order[i],order[j]]=[order[j],order[i]]; }

  for(const i of order) {
    // Get neighbors
    const neighbors = [];
    for(const [a,b] of edges) {
      if(a===i) neighbors.push(b);
      else if(b===i) neighbors.push(a);
    }
    if(neighbors.length === 0) continue;

    const j = neighbors[Math.floor(rng()*neighbors.length)];
    const delta = Math.abs(opinions[i] - opinions[j]);

    if(delta < eps) {
      const h = homophily(i, j);
      const effMu = mu * agents[i].openness * h;
      opinions[i] += effMu * (opinions[j] - opinions[i]);
      opinions[i] += (rng()-0.5) * 0.02;
      opinions[i] = Math.max(-1, Math.min(1, opinions[i]));
    }
  }
  simStep++;
}

// Easing
function ease(t) { return t*t*(3-2*t); }

// Canvas coords
function toCanvasX(x) { return (x + 1.2) / 2.4 * (canvas.width/devicePixelRatio); }
function toCanvasY(y) { return (1.05 - y) / 1.2 * (canvas.height/devicePixelRatio); }

// Color
function opColor(v) {
  // -1=blue, 0=gray, 1=red
  if(v < 0) {
    const t = -v;
    const r = Math.round(41 + (213-41)*(1-t));
    const g = Math.round(128 + (219-128)*(1-t));
    const b_ = Math.round(255 + (219-255)*(1-t));
    return `rgb(${r},${g},${b_})`;
  } else {
    const t = v;
    const r = Math.round(213 + (248-213)*t);
    const g = Math.round(219 + (81-219)*t);
    const b_ = Math.round(219 + (73-219)*t);
    return `rgb(${r},${g},${b_})`;
  }
}

function draw() {
  const W = canvas.width / devicePixelRatio;
  const H = canvas.height / devicePixelRatio;
  ctx.clearRect(0,0,W,H);

  const stacked = computeStacked(opinions);
  const ballsPerFrame = 3;

  // Draw edges (only for agents still in network)
  if(phase === 'idle' || phase === 'dropping') {
    ctx.strokeStyle = 'rgba(48,54,61,0.5)';
    ctx.lineWidth = 0.5;
    for(const [a,b] of edges) {
      if(dropProgress[a] < 0.05 && dropProgress[b] < 0.05) {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(agents[a].net_x), toCanvasY(agents[a].net_y));
        ctx.lineTo(toCanvasX(agents[b].net_x), toCanvasY(agents[b].net_y));
        ctx.stroke();
      }
    }
  }

  // Base line
  const baseY = toCanvasY(0.02);
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toCanvasX(-1.1), baseY);
  ctx.lineTo(toCanvasX(1.1), baseY);
  ctx.stroke();

  // Center line
  ctx.strokeStyle = 'rgba(48,54,61,0.3)';
  ctx.setLineDash([4,4]);
  ctx.beginPath();
  ctx.moveTo(toCanvasX(0), baseY);
  ctx.lineTo(toCanvasX(0), toCanvasY(0.95));
  ctx.stroke();
  ctx.setLineDash([]);

  // Draw agents
  for(let i=0; i<N; i++) {
    let x, y;
    const dp = dropProgress[i];

    if(dp <= 0) {
      x = agents[i].net_x;
      y = agents[i].net_y;
    } else if(dp >= 1) {
      x = stacked.xs[i];
      y = stacked.ys[i];
    } else {
      const t = ease(dp);
      x = agents[i].net_x + (stacked.xs[i] - agents[i].net_x) * t;
      y = agents[i].net_y + (stacked.ys[i] - agents[i].net_y) * t;
    }

    const cx = toCanvasX(x);
    const cy = toCanvasY(y);
    const r = selectedAgent === i ? 8 : 5.5;

    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.fillStyle = opColor(opinions[i]);
    ctx.fill();
    ctx.strokeStyle = selectedAgent === i ? '#fff' : '#222';
    ctx.lineWidth = selectedAgent === i ? 2 : 0.5;
    ctx.stroke();
  }

  // Labels
  ctx.fillStyle = '#58a6ff';
  ctx.font = 'bold 13px Segoe UI';
  ctx.textAlign = 'center';
  ctx.fillText(labelIzq, toCanvasX(-0.85), baseY + 18);
  ctx.fillStyle = '#f85149';
  ctx.fillText(labelDer, toCanvasX(0.85), baseY + 18);

  // Question
  ctx.fillStyle = '#e6edf3';
  ctx.font = 'bold 16px Segoe UI';
  ctx.textAlign = 'center';
  ctx.fillText(currentQ, W/2, 28);

  // Counts
  const nIzq = Array.from(opinions).filter(o => o < -0.2).length;
  const nDer = Array.from(opinions).filter(o => o > 0.2).length;
  const nCen = N - nIzq - nDer;

  ctx.font = 'bold 20px Segoe UI';
  ctx.fillStyle = '#58a6ff';
  ctx.fillText(nIzq, toCanvasX(-0.85), baseY + 42);
  ctx.fillStyle = '#484f58';
  ctx.fillText(nCen, toCanvasX(0), baseY + 42);
  ctx.fillStyle = '#f85149';
  ctx.fillText(nDer, toCanvasX(0.85), baseY + 42);
}

function updateStats() {
  const ops = Array.from(opinions);
  const nIzq = ops.filter(o => o < -0.2).length;
  const nDer = ops.filter(o => o > 0.2).length;
  const nCen = N - nIzq - nDer;
  const mean = ops.reduce((a,b)=>a+b,0)/N;
  const std = Math.sqrt(ops.reduce((a,b)=>a+(b-mean)**2,0)/N);

  document.getElementById('s-step').textContent = simStep;
  document.getElementById('s-izq').textContent = nIzq;
  document.getElementById('s-cen').textContent = nCen;
  document.getElementById('s-der').textContent = nDer;
  document.getElementById('s-mean').textContent = mean.toFixed(3);
  document.getElementById('s-std').textContent = std.toFixed(3);
}

// Animation loop
let lastTime = 0;
function animate(time) {
  if(!playing) return;

  const speed = parseInt(document.getElementById('speed').value);
  const interval = 180 - speed * 15;

  if(time - lastTime < interval) { requestAnimationFrame(animate); return; }
  lastTime = time;

  if(phase === 'dropping') {
    // Drop a few more balls
    let anyDropping = false;
    for(let i=0; i<N; i++) {
      const startFrame = Math.floor(agents[i].dropOrder / 3);
      const elapsed = animFrame - startFrame;
      if(elapsed < 0) { anyDropping = true; continue; }
      if(elapsed >= 10) { dropProgress[i] = 1; continue; }
      dropProgress[i] = elapsed / 10;
      anyDropping = true;
    }
    animFrame++;

    if(!anyDropping) {
      // All landed, start simulating
      phase = 'simulating';
      document.getElementById('phase-label').textContent = 'Interaccion entre agentes...';
    } else {
      document.getElementById('phase-label').textContent = 'Distribuyendo opiniones...';
    }
  } else if(phase === 'simulating') {
    simStepFn();
    document.getElementById('phase-label').textContent = `Interaccion ${simStep}/80`;
    if(simStep >= 80) {
      playing = false;
      document.getElementById('btn-play').textContent = 'Finalizado';
      document.getElementById('phase-label').textContent = 'Simulacion terminada';
    }
  }

  updateStats();
  draw();
  requestAnimationFrame(animate);
}

function togglePlay() {
  if(phase === 'idle') {
    phase = 'dropping';
    animFrame = 0;
  }
  playing = !playing;
  document.getElementById('btn-play').textContent = playing ? 'Pausar' : 'Continuar';
  if(playing) requestAnimationFrame(animate);
}

function resetSim() {
  playing = false;
  const sel = document.getElementById('topic-select');
  initAgents(sel.value);
  document.getElementById('btn-play').textContent = 'Iniciar';
  document.getElementById('phase-label').textContent = 'Selecciona una pregunta y presiona Iniciar';
  draw();
}

// Topic change
document.getElementById('topic-select').addEventListener('change', resetSim);

// Epsilon/mu display
document.getElementById('epsilon').addEventListener('input', e => {
  document.getElementById('eps-val').textContent = (e.target.value/100).toFixed(2);
});
document.getElementById('mu').addEventListener('input', e => {
  document.getElementById('mu-val').textContent = (e.target.value/100).toFixed(2);
});

// Click on agent
canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const stacked = computeStacked(opinions);
  let closest = -1, minD = 15;

  for(let i=0; i<N; i++) {
    let x, y;
    const dp = dropProgress[i];
    if(dp <= 0) { x = agents[i].net_x; y = agents[i].net_y; }
    else if(dp >= 1) { x = stacked.xs[i]; y = stacked.ys[i]; }
    else { const t=ease(dp); x=agents[i].net_x+(stacked.xs[i]-agents[i].net_x)*t; y=agents[i].net_y+(stacked.ys[i]-agents[i].net_y)*t; }

    const cx = toCanvasX(x), cy = toCanvasY(y);
    const d = Math.sqrt((mx-cx)**2 + (my-cy)**2);
    if(d < minD) { minD = d; closest = i; }
  }

  selectedAgent = closest;
  if(closest >= 0) {
    const a = agents[closest];
    const card = document.getElementById('agent-card');
    card.style.display = 'block';
    document.getElementById('ac-name').textContent = `Agente ${a.id}`;
    document.getElementById('ac-demo').textContent = `${a.sex}, ${a.age} anos, ${a.region}`;
    document.getElementById('ac-pol').textContent = `NSE: ${a.gse} | Educacion: ${a.education}`;
    document.getElementById('ac-rel').textContent = `Religion: ${a.religion} | Pos.pol: ${a.pol_position !== null ? a.pol_position+'/10' : 'N/D'}`;

    const op = opinions[closest];
    const pct = ((op+1)/2*100).toFixed(0);
    const barColor = op < 0 ? '#58a6ff' : '#f85149';
    document.getElementById('ac-opinion').innerHTML =
      `<div class="bar-container">
        <span class="bar-label">Opinion:</span>
        <div class="bar-bg"><div class="bar-fill" style="width:${pct}%;background:${barColor}"></div></div>
        <span class="bar-val">${op.toFixed(2)}</span>
      </div>
      <p style="font-size:11px;color:#484f58;margin-top:4px">Apertura al cambio: ${a.openness.toFixed(2)}</p>`;
  } else {
    document.getElementById('agent-card').style.display = 'none';
  }
  draw();
});

// Init
initAgents('pensiones');
draw();
</script>
</body>
</html>"""

output = Path("data/processed/simulacion_interactiva.html")
output.write_text(html, encoding="utf-8")
print(f"HTML: {output.resolve()} ({output.stat().st_size/1024:.0f} KB)")

import webbrowser
webbrowser.open(str(output.resolve()))
