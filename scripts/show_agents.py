"""Muestra los agentes generados a partir de la CEP 95."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
from collections import Counter
from src.data.loader import load_cep
from src.data.profiler import build_profiles

import warnings
warnings.filterwarnings("ignore")

df = load_cep("data/raw/encuesta_95/bases/cep95.csv")
profiles = build_profiles(df)

print(f"Total agentes generados: {len(profiles)}")
print()

# --- SEXO ---
sexos = Counter(p.sex for p in profiles)
print("=== SEXO ===")
for k, v in sexos.most_common():
    print(f"  {k}: {v} ({v/len(profiles)*100:.1f}%)")

# --- REGION ---
print("\n=== REGION (top 10) ===")
regiones = Counter(p.region for p in profiles)
for k, v in regiones.most_common(10):
    print(f"  {k}: {v}")

# --- NSE ---
print("\n=== NSE ===")
gses = Counter(p.gse for p in profiles)
for k, v in sorted(gses.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v} ({v/len(profiles)*100:.1f}%)")

# --- EDAD ---
edades = [p.age for p in profiles]
print(f"\n=== EDAD ===")
print(f"  Min: {min(edades)}, Max: {max(edades)}, Media: {sum(edades)/len(edades):.1f}")
rangos = Counter()
for a in edades:
    if a <= 30: rangos["18-30"] += 1
    elif a <= 45: rangos["31-45"] += 1
    elif a <= 60: rangos["46-60"] += 1
    else: rangos["61+"] += 1
for k in ["18-30", "31-45", "46-60", "61+"]:
    print(f"  {k}: {rangos[k]} ({rangos[k]/len(profiles)*100:.1f}%)")

# --- RELIGION ---
print("\n=== RELIGION ===")
rels = Counter(p.religion for p in profiles if p.religion)
for k, v in rels.most_common():
    print(f"  {k}: {v} ({v/len(profiles)*100:.1f}%)")

# --- POSICION POLITICA ---
pols = [p.political_position for p in profiles if p.political_position is not None]
print(f"\n=== POSICION POLITICA (0=izq, 10=der) ===")
print(f"  Con posicion: {len(pols)}/{len(profiles)}")
if pols:
    print(f"  Media: {sum(pols)/len(pols):.1f}")
    pol_rangos = Counter()
    for p in pols:
        if p <= 3: pol_rangos["Izquierda (0-3)"] += 1
        elif p <= 5: pol_rangos["Centro (4-5)"] += 1
        elif p <= 7: pol_rangos["Centro-der (6-7)"] += 1
        else: pol_rangos["Derecha (8-10)"] += 1
    for k in ["Izquierda (0-3)", "Centro (4-5)", "Centro-der (6-7)", "Derecha (8-10)"]:
        print(f"  {k}: {pol_rangos[k]} ({pol_rangos[k]/len(pols)*100:.1f}%)")

# --- DEMOCRACIA ---
print("\n=== PREFERENCIA DEMOCRATICA ===")
demos = Counter(int(p.pref_democracia) for p in profiles if p.pref_democracia is not None)
demo_labels = {1: "Democracia siempre preferible", 2: "Autoritarismo a veces", 3: "Da lo mismo"}
for k in [1, 2, 3]:
    v = demos.get(k, 0)
    total_d = sum(demos.values())
    print(f"  {demo_labels[k]}: {v} ({v/total_d*100:.1f}%)")

# --- EJEMPLOS ---
print("\n" + "=" * 60)
print("=== 10 AGENTES DE EJEMPLO ===")
print("=" * 60)
random.seed(42)
muestra = random.sample(profiles, 10)
for p in muestra:
    pol = f"{p.political_position:.0f}/10" if p.political_position else "N/D"
    rel = p.religion or "Sin religion"
    print(f"\n  [{p.agent_id}]")
    print(f"  {p.sex}, {p.age} anos, {p.region} ({p.urban_rural})")
    print(f"  Educacion: {p.education} | NSE: {p.gse} | Religion: {rel}")
    print(f"  Posicion politica: {pol} | Interes politico: {p.political_interest or 'N/D'}/7")

    if p.eval_gobierno is not None:
        eval_text = "Muy bien" if p.eval_gobierno >= 0.75 else "Bien" if p.eval_gobierno >= 0.5 else "Mal" if p.eval_gobierno >= 0.25 else "Muy mal"
        print(f"  Eval gobierno: {eval_text} ({p.eval_gobierno:.2f})")

    if p.pref_democracia is not None:
        dt = {1: "Siempre democracia", 2: "A veces autoritarismo", 3: "Da lo mismo"}
        print(f"  Democracia: {dt.get(int(p.pref_democracia), '?')}")

    if p.trust:
        sorted_trust = sorted(p.trust.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_trust[:3]
        bot2 = sorted_trust[-2:]
        top_str = ", ".join(f"{k} ({v:.0%})" for k, v in top3)
        bot_str = ", ".join(f"{k} ({v:.0%})" for k, v in bot2)
        print(f"  Mas confia: {top_str}")
        print(f"  Menos confia: {bot_str}")

    opinions = p.current_opinions
    if opinions:
        op_str = ", ".join(f"{k}={v:+.2f}" for k, v in opinions.items())
        print(f"  Opiniones iniciales: {op_str}")
