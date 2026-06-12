"""Genera la silueta de Chile y posiciones de agentes dentro de cada región.

Lee el geojson de comunas, une y suaviza las geometrías (recortando islas
oceánicas y Antártica), y muestrea para cada agente del dashboard un punto
dentro del polígono real de su región. Emite:
  - data/processed/chile_outline.js  -> const CHILE={polys, regions, pts, aspect}
  - data/processed/chile_preview.png -> vista previa para control visual

Coordenadas: u = norte-sur (0=Arica, 1=extremo sur), v = este-oeste
(0=Andes, 1=costa), con `aspect` = alto/ancho real (corrección cos(lat)).
"""
import io
import json
import math
import random
import re
from pathlib import Path

from shapely.geometry import shape, box, Point
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
GEOJSON = Path(r"G:\Mi unidad\CEP\0_Meta\0.2_Nucleo\_origen_08_MAPAS\chile_comunas.geojson")
HTML = ROOT / "data" / "processed" / "simulacion_interactiva.html"
OUT = ROOT / "data" / "processed" / "chile_outline.js"
PREVIEW = ROOT / "data" / "processed" / "chile_preview.png"

CLIP = box(-76.5, -56.2, -66.0, -17.4)  # sin Rapa Nui, J. Fernández ni Antártica
LAT_N, LAT_S = -17.45, -56.05
LON_E, LON_W = -66.4, -75.8
COS_MID = math.cos(math.radians(36.0))
ASPECT = ((LON_E - LON_W) * COS_MID) / (LAT_N - LAT_S)

COD2NAME = {
    15: "Arica y Parinacota", 1: "Tarapacá", 2: "Antofagasta", 3: "Atacama",
    4: "Coquimbo", 5: "Valparaíso", 13: "Metropolitana", 6: "O'Higgins",
    7: "Maule", 16: "Ñuble", 8: "Biobío", 9: "Araucanía", 14: "Los Ríos",
    10: "Los Lagos", 11: "Aysén", 12: "Magallanes",
}


def uv(lon: float, lat: float) -> tuple[float, float]:
    return (
        round((LAT_N - lat) / (LAT_N - LAT_S), 4),
        round((LON_E - lon) / (LON_E - LON_W), 4),
    )


def counts_from_html() -> dict[str, int]:
    html = io.open(HTML, encoding="utf-8").read()
    d = json.loads(re.search(r"const D=(\{.*?\});", html, re.S).group(1))
    counts: dict[str, int] = {}
    for a in d["a"]:
        counts[a["r"]] = counts.get(a["r"], 0) + 1
    return counts


def sample_in(geom, n: int, rng: random.Random) -> list[tuple[float, float]]:
    """n puntos uniformes dentro de geom, con separación mínima adaptativa."""
    minx, miny, maxx, maxy = geom.bounds
    d_min = math.sqrt(geom.area / max(n, 1)) * 0.55
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < n * 4000:
        tries += 1
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not geom.contains(Point(x, y)):
            continue
        if any((x - px) ** 2 + ((y - py) * COS_MID) ** 2 < d_min**2 for px, py in pts):
            if tries % 800 == 0:
                d_min *= 0.85  # relajar si la región está muy llena
            continue
        pts.append((x, y))
    while len(pts) < n:  # último recurso: duplicar con jitter
        x, y = pts[rng.randrange(len(pts))]
        pts.append((x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02)))
    return pts


def main() -> None:
    gj = json.loads(GEOJSON.read_text(encoding="utf-8"))

    geoms, por_region = [], {}
    for f in gj["features"]:
        g = shape(f["geometry"]).buffer(0).intersection(CLIP)
        if g.is_empty:
            continue
        geoms.append(g)
        cod = int(f["properties"]["codregion"])
        if cod in COD2NAME:
            por_region.setdefault(cod, []).append(g)

    # Suavizado: cierra fiordos y une el archipiélago austral a la costa
    todo = unary_union(geoms)
    suave = todo.buffer(0.09).buffer(-0.07).simplify(0.035, preserve_topology=True)
    polys = sorted(
        suave.geoms if suave.geom_type == "MultiPolygon" else [suave],
        key=lambda p: p.area, reverse=True,
    )
    polys = [p for p in polys if p.area > 0.25]

    rings = [[list(uv(lon, lat)) for lon, lat in p.exterior.coords] for p in polys]

    regiones_geom = {COD2NAME[c]: unary_union(gs) for c, gs in por_region.items()}
    regions = {
        name: round((LAT_N - g.centroid.y) / (LAT_N - LAT_S), 4)
        for name, g in regiones_geom.items()
    }

    counts = counts_from_html()
    rng = random.Random(42)
    pts: dict[str, list[list[float]]] = {}
    for name, n in counts.items():
        g = regiones_geom.get(name)
        if g is None:
            continue
        # restringe el muestreo a la silueta suavizada para no caer "al mar"
        gi = g.intersection(suave)
        if gi.is_empty:
            gi = g
        pts[name] = [list(uv(x, y)) for x, y in sample_in(gi, n, rng)]
        print(f"  {name}: {n} puntos")

    data = {"aspect": round(ASPECT, 4), "polys": rings, "regions": regions, "pts": pts}
    js = "const CHILE=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";"
    OUT.write_text(js, encoding="utf-8")
    print(f"{len(rings)} polígonos, {sum(len(r) for r in rings)} pts de contorno, "
          f"{len(js)/1024:.0f} KB")

    # Vista previa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 16 * ASPECT * 1.4))
    for ring in rings:
        xs = [p[0] for p in ring]
        ys = [p[1] * ASPECT for p in ring]
        ax.fill(xs, ys, color="#eef1f3", edgecolor="#c5cdd2", lw=0.8, zorder=1)
    for name, ps in pts.items():
        ax.scatter([p[0] for p in ps], [p[1] * ASPECT for p in ps], s=4, zorder=2)
    for name, u in regions.items():
        ax.text(u, ASPECT * 1.12, name.split()[-1][:7], ha="center", fontsize=7, color="#888")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(PREVIEW, dpi=90, bbox_inches="tight")
    print(f"preview -> {PREVIEW}")


if __name__ == "__main__":
    main()
