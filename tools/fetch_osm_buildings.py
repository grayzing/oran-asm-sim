"""Download OpenStreetMap building footprints (with heights) for a bbox via the
Overpass API and save them as GeoJSON for ``geometry.buildings_geojson``.

    python tools/fetch_osm_buildings.py --out data/dtla_buildings.geojson
    python tools/fetch_osm_buildings.py --bbox 34.036 -118.270 34.062 -118.230 --out my.geojson

Heights: OSM ``height`` (m) or ``building:levels`` x 3.5 m are kept as properties;
buildings without either get ``geometry.building_default_height_m`` in the simulator.
Alternative higher-quality source for LA: LA County "Countywide Building Outlines"
(LARIAC) export as GeoJSON, with ``building_height_field="HEIGHT"`` and
``building_height_scale=0.3048`` (feet -> metres).

Only the Python standard library is used. Please respect the Overpass usage policy.
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

DTLA_BBOX = (34.0360, -118.2700, 34.0620, -118.2300)
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]


def fetch(bbox, timeout=180):
    s, w, n, e = bbox
    q = f"""[out:json][timeout:{timeout}];
(way["building"]({s},{w},{n},{e});
 relation["building"]["type"="multipolygon"]({s},{w},{n},{e}););
out geom;"""
    data = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "oran_asm_sim/0.2"})
            with urllib.request.urlopen(req, timeout=timeout + 30) as r:
                return json.load(r)
        except Exception as ex:  # try the next mirror
            last = ex
            print(f"{url}: {ex}", file=sys.stderr)
    raise RuntimeError(f"all Overpass endpoints failed: {last}")


def to_geojson(osm):
    feats = []
    for el in osm.get("elements", []):
        tags = el.get("tags", {}) or {}
        props = {k: tags[k] for k in ("height", "building:levels", "building", "name") if k in tags}
        props["osm_id"] = f"{el['type']}/{el['id']}"
        rings = []
        if el["type"] == "way" and "geometry" in el:
            rings = [[[g["lon"], g["lat"]] for g in el["geometry"]]]
        elif el["type"] == "relation":
            rings = [[[g["lon"], g["lat"]] for g in m["geometry"]]
                     for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m]
        for ring in rings:
            if len(ring) >= 4:
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                feats.append({"type": "Feature", "properties": props,
                              "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, default=DTLA_BBOX, metavar=("S", "W", "N", "E"))
    ap.add_argument("--out", default="data/dtla_buildings.geojson")
    a = ap.parse_args()
    gj = to_geojson(fetch(a.bbox))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(gj, f)
    n = len(gj["features"])
    nh = sum(1 for f in gj["features"] if "height" in f["properties"] or "building:levels" in f["properties"])
    print(f"wrote {n} footprints ({nh} with height/levels) -> {a.out}")


if __name__ == "__main__":
    main()
