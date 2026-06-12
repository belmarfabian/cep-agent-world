"""Genera la silueta simplificada de Chile continental (+ islas mayores) como JS.

Lee el geojson de comunas, une las geometrías, recorta islas oceánicas y
Antártica, simplifica y emite `data/processed/chile_outline.js` con:
  - polys: lista de anillos [[u, v], ...] con u = posición norte-sur (0=Arica,
    1=extremo sur) y v = posición oeste-este normalizada (0=Andes, 1=costa).
  - regions: centroide norte-sur (u) por región, con los nombres de la encuesta.
"""
import json
from pathlib import Path

from shapely.geometry import shape, box
from shapely.ops import unary_union

GEOJSON = Path(r"G:\Mi unidad\CEP\0_Meta\0.2_Nucleo\_origen_08_MAPAS\chile_comunas.geojson")
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "chile_outline.js"

# Continente + Chiloé + Tierra del Fuego; fuera Rapa Nui, Juan Fernández y Antártica
CLIP = box(-76.5, -56.2, -66.0, -17.4)

LAT_N, LAT_S = -17.45, -56.05

# codregion -> nombre usado en la encuesta CEP
COD2NAME = {
    15: "Arica y Parinacota", 1: "Tarapacá", 2: "Antofagasta", 3: "Atacama",
    4: "Coquimbo", 5: "Valparaíso", 13: "Metropolitana", 6: "O'Higgins",
    7: "Maule", 16: "Ñuble", 8: "Biobío", 9: "Araucanía", 14: "Los Ríos",
    10: "Los Lagos", 11: "Aysén", 12: "Magallanes",
}


def u_of_lat(lat: float) -> float:
    return (LAT_N - lat) / (LAT_N - LAT_S)


def main() -> None:
    gj = json.loads(GEOJSON.read_text(encoding="utf-8"))

    geoms, por_region = [], {}
    for f in gj["features"]:
        g = shape(f["geometry"]).buffer(0).intersection(CLIP)
        if g.is_empty:
            continue
        geoms.append(g)
        cod = int(f["properties"]["codregion"])
        por_region.setdefault(cod, []).append(g)

    todo = unary_union(geoms).simplify(0.045, preserve_topology=True)
    polys = sorted(
        todo.geoms if todo.geom_type == "MultiPolygon" else [todo],
        key=lambda p: p.area, reverse=True,
    )
    polys = [p for p in polys if p.area > 0.15]  # continente, Chiloé, T. del Fuego

    lons = [x for p in polys for x, _ in p.exterior.coords]
    lon_e, lon_w = max(lons), min(lons)  # este (Andes), oeste (costa)

    rings = []
    for p in polys:
        ring = [
            [round(u_of_lat(lat), 4), round((lon_e - lon) / (lon_e - lon_w), 4)]
            for lon, lat in p.exterior.coords
        ]
        rings.append(ring)

    regions = {}
    for cod, gs in por_region.items():
        if cod not in COD2NAME:
            continue
        c = unary_union(gs).centroid
        regions[COD2NAME[cod]] = round(u_of_lat(c.y), 4)

    js = "const CHILE=" + json.dumps(
        {"polys": rings, "regions": regions}, ensure_ascii=False, separators=(",", ":")
    ) + ";"
    OUT.write_text(js, encoding="utf-8")
    npts = sum(len(r) for r in rings)
    print(f"{len(rings)} polígonos, {npts} puntos, {len(js)/1024:.1f} KB -> {OUT}")
    for name, u in sorted(regions.items(), key=lambda kv: kv[1]):
        print(f"  {name}: u={u}")


if __name__ == "__main__":
    main()
