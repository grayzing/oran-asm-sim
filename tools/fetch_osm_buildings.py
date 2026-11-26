"""Download OpenStreetMap building footprints (with heights) for a bbox and save
them as GeoJSON for ``geometry.buildings_geojson``.

    python tools/fetch_osm_buildings.py --out data/dtla_buildings.geojson
    python tools/fetch_osm_buildings.py --bbox 34.036 -118.270 34.062 -118.230 --out my.geojson

Sources (``--source``):
  api       (default) the main OSM editing API v0.6 ``/map`` call. It is far more
            reliable than the public Overpass mirrors. Large bboxes are tiled
            automatically: the API caps each request at 0.25 deg^2 / 50k nodes.
  overpass  the public Overpass mirrors (the old behaviour).
  auto      try ``api`` first, then fall back to ``overpass``.

Heights: OSM ``height`` (m) or ``building:levels`` x 3.5 m are kept as properties;
buildings without either get ``geometry.building_default_height_m`` in the simulator.
Alternative higher-quality source for LA: LA County "Countywide Building Outlines"
(LARIAC) export as GeoJSON, with ``building_height_field="HEIGHT"`` and
``building_height_scale=0.3048`` (feet -> metres).

Only the Python standard library is used. Please respect the OSM API / Overpass
usage policies (keep bboxes modest; this is not a bulk-download tool).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# (S, W, N, E) in degrees
DTLA_BBOX = (33.970067, -117.331924, 33.975550, -117.323470)
USER_AGENT = "oran_asm_sim/0.2 (building footprint fetcher)"
OSM_API = "https://api.openstreetmap.org/api/0.6/map"
OSM_API_MAX_AREA = 0.25          # deg^2, hard server limit
OSM_API_TILE = 0.05              # deg; keeps tiles well under the 50k-node cap in dense cities
OVERPASS_ENDPOINTS = ["https://overpass-api.de/api/interpreter",
                      "https://overpass.kumi.systems/api/interpreter",
                      "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
KEEP_TAGS = ("height", "building:levels", "building", "name")


def _http(url, data=None, timeout=120, retries=3):
    """GET/POST with retry + exponential backoff on 429/5xx/network errors.
    Re-raises HTTPError for other 4xx so callers can react (e.g. split the tile)."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as ex:
            if ex.code != 429 and ex.code < 500:
                raise
            err = ex
        except (urllib.error.URLError, TimeoutError, ConnectionError) as ex:
            err = ex
        if attempt + 1 < retries:
            wait = 2 ** attempt * 2
            print(f"  {url.split('?')[0]}: {err}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise err


# ----------------------------------------------------------------------------- OSM API v0.6
def _tiles(bbox, step):
    s, w, n, e = bbox
    lat = s
    while lat < n:
        lon = w
        lat2 = min(lat + step, n)
        while lon < e:
            lon2 = min(lon + step, e)
            yield (lat, lon, lat2, lon2)
            lon = lon2
        lat = lat2


def _fetch_api_tile(bbox, nodes, ways, rels, depth=0):
    s, w, n, e = bbox
    url = f"{OSM_API}?bbox={w},{s},{e},{n}"
    try:
        raw = _http(url)
    except urllib.error.HTTPError as ex:
        # 400 = too many nodes / area too large -> split into quadrants
        if ex.code == 400 and depth < 6:
            print(f"  tile {bbox} too dense ({ex.code}); splitting", file=sys.stderr)
            mlat, mlon = (s + n) / 2, (w + e) / 2
            for sub in ((s, w, mlat, mlon), (s, mlon, mlat, e), (mlat, w, n, mlon), (mlat, mlon, n, e)):
                _fetch_api_tile(sub, nodes, ways, rels, depth + 1)
            return
        raise
    root = ET.fromstring(raw)
    for el in root:
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("lon")), float(el.get("lat")))
        elif el.tag == "way":
            ways[el.get("id")] = {"nds": [nd.get("ref") for nd in el.iter("nd")],
                                  "tags": {t.get("k"): t.get("v") for t in el.iter("tag")}}
        elif el.tag == "relation":
            rels[el.get("id")] = {"members": [(m.get("type"), m.get("ref"), m.get("role"))
                                              for m in el.iter("member")],
                                  "tags": {t.get("k"): t.get("v") for t in el.iter("tag")}}


def _stitch(segments):
    """Join open way segments (lists of coords) into closed rings."""
    rings, segs = [], [list(s) for s in segments if len(s) >= 2]
    while segs:
        ring = segs.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, sg in enumerate(segs):
                if sg[0] == ring[-1]:
                    ring += sg[1:]
                elif sg[-1] == ring[-1]:
                    ring += sg[-2::-1]
                elif sg[-1] == ring[0]:
                    ring = sg[:-1] + ring
                elif sg[0] == ring[0]:
                    ring = sg[:0:-1] + ring
                else:
                    continue
                segs.pop(i)
                changed = True
                break
        if ring[0] == ring[-1]:
            rings.append(ring)
    return rings


def fetch_api(bbox):
    s, w, n, e = bbox
    tiles = list(_tiles(bbox, OSM_API_TILE)) if (n - s) * (e - w) > OSM_API_TILE ** 2 else [bbox]
    nodes, ways, rels = {}, {}, {}
    for i, t in enumerate(tiles):
        if len(tiles) > 1:
            print(f"  OSM API tile {i + 1}/{len(tiles)}", file=sys.stderr)
        _fetch_api_tile(t, nodes, ways, rels)

    def coords(way_id):
        way = ways.get(way_id)
        if way is None or any(r not in nodes for r in way["nds"]):
            return None  # way (or some of its nodes) lies outside the downloaded area
        return [list(nodes[r]) for r in way["nds"]]

    out = []
    for wid, way in ways.items():
        if "building" in way["tags"]:
            c = coords(wid)
            if c:
                out.append(("way", wid, way["tags"], [c]))
    for rid, rel in rels.items():
        if "building" in rel["tags"] and rel["tags"].get("type") == "multipolygon":
            segs = [coords(ref) for typ, ref, role in rel["members"] if typ == "way" and role == "outer"]
            rings = _stitch([c for c in segs if c])
            if rings:
                out.append(("relation", rid, rel["tags"], rings))
    return out


# ----------------------------------------------------------------------------- Overpass
def fetch_overpass(bbox, timeout=180):
    s, w, n, e = bbox
    q = f"""[out:json][timeout:{timeout}];
(way["building"]({s},{w},{n},{e});
 relation["building"]["type"="multipolygon"]({s},{w},{n},{e}););
out geom;"""
    data = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for url in OVERPASS_ENDPOINTS:
        try:
            osm = json.loads(_http(url, data=data, timeout=timeout + 30, retries=2))
            break
        except Exception as ex:  # try the next mirror
            last = ex
            print(f"  {url}: {ex}", file=sys.stderr)
    else:
        raise RuntimeError(f"all Overpass endpoints failed: {last}")

    out = []
    for el in osm.get("elements", []):
        if el["type"] == "way" and "geometry" in el:
            rings = [[[g["lon"], g["lat"]] for g in el["geometry"]]]
        elif el["type"] == "relation":
            rings = _stitch([[[g["lon"], g["lat"]] for g in m["geometry"]]
                             for m in el.get("members", []) if m.get("role") == "outer" and "geometry" in m])
        else:
            continue
        out.append((el["type"], str(el["id"]), el.get("tags", {}) or {}, rings))
    return out


# ----------------------------------------------------------------------------- GeoJSON
def to_geojson(buildings):
    feats = []
    for typ, oid, tags, rings in buildings:
        props = {k: tags[k] for k in KEEP_TAGS if k in tags}
        props["osm_id"] = f"{typ}/{oid}"
        for ring in rings:
            if len(ring) >= 4:
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                feats.append({"type": "Feature", "properties": props,
                              "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


def check_bbox(bbox):
    s, w, n, e = bbox
    if not (-90 <= s < n <= 90 and -180 <= w < e <= 180):
        sys.exit(f"invalid bbox {bbox}: expected S W N E (lat, lon, lat, lon) with S<N and W<E")
    if (n - s) * (e - w) > 4 * OSM_API_MAX_AREA:
        print(f"warning: bbox covers {(n - s) * (e - w):.2f} deg^2; this will take many requests",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, default=DTLA_BBOX, metavar=("S", "W", "N", "E"))
    ap.add_argument("--out", default="data/dtla_buildings.geojson")
    ap.add_argument("--source", choices=("api", "overpass", "auto"), default="api")
    a = ap.parse_args()
    check_bbox(a.bbox)

    if a.source == "overpass":
        buildings = fetch_overpass(a.bbox)
    else:
        try:
            buildings = fetch_api(a.bbox)
        except Exception as ex:
            if a.source != "auto":
                raise
            print(f"OSM API failed ({ex}); falling back to Overpass", file=sys.stderr)
            buildings = fetch_overpass(a.bbox)

    gj = to_geojson(buildings)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(gj, f)
    n = len(gj["features"])
    nh = sum(1 for f in gj["features"] if "height" in f["properties"] or "building:levels" in f["properties"])
    print(f"wrote {n} footprints ({nh} with height/levels) -> {a.out}")


if __name__ == "__main__":
    main()
