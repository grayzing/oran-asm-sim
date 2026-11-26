"""GIS geometry: bounding box -> local ENU frame, hexagonal tri-sector layout,
points of interest (TVC communities) and optional building raster for LOS.

Only numpy is required (no shapely / pyproj). The projection is a local
equirectangular (ENU tangent-plane) approximation, accurate to < 0.1 % over a
few kilometres, which is what a downtown study area needs.
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import GeometryConfig, HetNetConfig, RadioConfig

EARTH_R = 6_371_008.8


class LocalProjection:
    """WGS-84 lat/lon <-> local metres (x east, y north) around (lat0, lon0)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self._kx = EARTH_R * np.cos(np.radians(lat0)) * np.pi / 180.0
        self._ky = EARTH_R * np.pi / 180.0

    def to_xy(self, lat, lon):
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        return np.stack([(lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky], -1)

    def to_latlon(self, xy):
        xy = np.asarray(xy, float)
        return np.stack([self.lat0 + xy[..., 1] / self._ky, self.lon0 + xy[..., 0] / self._kx], -1)


# --------------------------------------------------------------------------- #
# Points of interest (approximate WGS-84 coordinates). Used as CMP communities
# in the TVC model ([HSU07] Sec. V: "fix the locations of multiple communities
# serving as known points-of-interest").
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class POI:
    name: str
    lat: float
    lon: float
    tags: Tuple[str, ...]
    weight: float = 1.0


DTLA_POIS: Tuple[POI, ...] = (
    POI("Pershing Square", 34.0488, -118.2518, ("food", "work"), 1.0),
    POI("Grand Central Market", 34.0508, -118.2489, ("food",), 1.5),
    POI("Bradbury Building", 34.0505, -118.2479, ("work",), 0.6),
    POI("Walt Disney Concert Hall", 34.0553, -118.2498, ("night",), 1.2),
    POI("The Broad", 34.0544, -118.2506, ("night", "food"), 0.8),
    POI("LA City Hall", 34.0537, -118.2427, ("work",), 1.5),
    POI("Grand Park", 34.0560, -118.2454, ("food",), 0.8),
    POI("Union Station", 34.0562, -118.2365, ("work", "night"), 1.2),
    POI("Little Tokyo", 34.0490, -118.2397, ("food", "night"), 1.4),
    POI("7th St / Metro Center", 34.0486, -118.2587, ("work",), 1.5),
    POI("Wilshire Grand Center", 34.0502, -118.2603, ("work",), 1.5),
    POI("U.S. Bank Tower", 34.0511, -118.2543, ("work",), 1.3),
    POI("Crypto.com Arena", 34.0430, -118.2673, ("night",), 2.0),
    POI("L.A. LIVE", 34.0447, -118.2658, ("night", "food"), 1.5),
    POI("Orpheum Theatre", 34.0447, -118.2566, ("night",), 0.7),
    POI("Fashion District", 34.0405, -118.2535, ("work", "food"), 0.9),
    POI("Arts District", 34.0400, -118.2330, ("food", "night"), 1.0),
)


# --------------------------------------------------------------------------- #
@dataclass
class Deployment:
    proj: LocalProjection
    area_min: np.ndarray          # (2,) metres
    area_max: np.ndarray          # (2,) metres
    site_xy: np.ndarray           # (n_sites, 2)
    sector_site: np.ndarray       # (n_sectors,) site index
    sector_az_deg: np.ndarray     # (n_sectors,)
    poi_xy: np.ndarray            # (n_poi, 2)
    poi_names: List[str]
    poi_tags: List[Tuple[str, ...]]
    poi_weights: np.ndarray
    buildings: Optional["BuildingRaster"] = None
    area_polygon_xy: Optional[np.ndarray] = None
    site_h: Optional[np.ndarray] = None        # [n_sites] antenna height above ground
    site_tilt_deg: Optional[np.ndarray] = None # [n_sites] downtilt
    site_vis: Optional[np.ndarray] = None      # [n_sites, n_cells] bool viewsheds (LOS to h_ut)
    placement: Optional[object] = None
    buildings_source: str = "none"
    site_tier: Optional[np.ndarray] = None     # [n_sites] tier index (HetNet: 0 = macro, 1 = mmWave)
    tier_names: Tuple[str, ...] = ("cell",)

    def __post_init__(self):
        if self.site_tier is None:
            self.site_tier = np.zeros(len(self.site_xy), int)

    @property
    def n_sites(self) -> int:
        return len(self.site_xy)

    @property
    def n_sectors(self) -> int:
        return len(self.sector_site)

    @property
    def sector_tier(self) -> np.ndarray:
        return self.site_tier[self.sector_site]

    @property
    def n_tiers(self) -> int:
        return len(self.tier_names)

    @property
    def sector_xy(self) -> np.ndarray:
        return self.site_xy[self.sector_site]

    @property
    def area_size(self) -> np.ndarray:
        return self.area_max - self.area_min

    def site_latlon(self) -> np.ndarray:
        return self.proj.to_latlon(self.site_xy)


# --------------------------------------------------------------------------- #
# Geo helpers: GeoJSON / Overpass JSON building import
# --------------------------------------------------------------------------- #
def _load_geojson(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _iter_polygons(gj: dict):
    """Yield (exterior_ring [N,2] lon/lat, properties) for every polygon.

    Accepts GeoJSON (Feature / FeatureCollection / bare geometry) and raw
    Overpass API JSON (``[out:json]; ... out geom;``)."""
    if "elements" in gj:                                   # Overpass API output
        for el in gj["elements"]:
            tags = el.get("tags", {}) or {}
            if el.get("type") == "way" and "geometry" in el:
                ring = [(g["lon"], g["lat"]) for g in el["geometry"]]
                if len(ring) >= 3:
                    yield np.asarray(ring, float), tags
            elif el.get("type") == "relation":
                for m in el.get("members", []):
                    if m.get("role") == "outer" and "geometry" in m and len(m["geometry"]) >= 3:
                        yield np.asarray([(g["lon"], g["lat"]) for g in m["geometry"]], float), tags
        return
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    for ft in feats:
        geom = ft.get("geometry", ft)
        if geom is None:
            continue
        props = ft.get("properties", {}) or {}
        if geom["type"] == "Polygon":
            yield np.asarray(geom["coordinates"][0], float)[:, :2], props
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                yield np.asarray(poly[0], float)[:, :2], props


def _parse_height(props: dict, default_h: float, scale: float, field: Optional[str]) -> float:
    keys = [field] if field else ["height", "HEIGHT", "building:height", "Height", "HGT"]
    for k in keys:
        v = props.get(k)
        if v is None:
            continue
        try:
            sv = str(v).strip().lower()
            feet = sv.endswith("'") or sv.endswith("ft")
            num = float(sv.replace("m", "").replace("ft", "").replace("'", "").strip())
            if num > 0:
                return num * (0.3048 if feet else scale)
        except ValueError:
            continue
    lv = props.get("building:levels")
    try:
        if lv is not None and float(lv) > 0:
            return float(lv) * 3.5
    except ValueError:
        pass
    return default_h


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting. pts [N,2], poly [M,2] -> bool [N]."""
    x, y = pts[:, 0:1], pts[:, 1:2]
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    cond = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
    return np.logical_and(cond, x < xint).sum(1) % 2 == 1


class BuildingRaster:
    """Building-height grid ("digital surface model") over the study area.

    Cell (r, c) covers x in [ox + c*res, ox + (c+1)*res), y likewise, with the
    origin at the area's south-west corner. ``h == 0`` marks outdoor cells (the
    service area of [MASRUR25]); every indoor cell knows its nearest outdoor
    cell, used for O2I (LOS of the facade + indoor distance)."""

    def __init__(self, h: np.ndarray, origin, res: float, n_buildings: int = 0):
        self.h = np.asarray(h, np.float32)
        self.origin = np.asarray(origin, float)
        self.res = float(res)
        self.shape = self.h.shape
        self.n_buildings = n_buildings
        self.outdoor = self.h <= 0
        if self.outdoor.all() or not self.outdoor.any():
            self.near_r, self.near_c = np.indices(self.shape)
            self.near_d = np.zeros(self.shape, np.float32)
        else:
            from scipy.ndimage import distance_transform_edt
            d, (rr, cc) = distance_transform_edt(~self.outdoor, return_indices=True)
            self.near_r, self.near_c = rr.astype(np.int32), cc.astype(np.int32)
            self.near_d = (d * self.res).astype(np.float32)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str, proj: "LocalProjection", area_min, area_max, res=5.0,
                  default_h=20.0, height_scale=1.0, height_field=None):
        origin = np.asarray(area_min, float)
        size = np.asarray(area_max, float) - origin
        shape = (int(np.ceil(size[1] / res)), int(np.ceil(size[0] / res)))
        h = np.zeros(shape, np.float32)
        n = 0
        for ring, props in _iter_polygons(_load_geojson(path)):
            xy = proj.to_xy(ring[:, 1], ring[:, 0])
            lo = np.floor((xy.min(0) - origin) / res).astype(int)
            hi = np.ceil((xy.max(0) - origin) / res).astype(int)
            c0, r0 = max(lo[0], 0), max(lo[1], 0)
            c1, r1 = min(hi[0], shape[1]), min(hi[1], shape[0])
            if c1 <= c0 or r1 <= r0:
                continue
            gr, gc = np.mgrid[r0:r1, c0:c1]
            ctr = np.stack([origin[0] + (gc + 0.5) * res, origin[1] + (gr + 0.5) * res], -1).reshape(-1, 2)
            inside = points_in_polygon(ctr, xy).reshape(gr.shape)
            if not inside.any():
                continue
            hb = _parse_height(props, default_h, height_scale, height_field)
            sub = h[r0:r1, c0:c1]
            sub[inside] = np.maximum(sub[inside], hb)
            n += 1
        return cls(h, origin, res, n)

    # ------------------------------------------------------------------ #
    def cell_rc(self, xy: np.ndarray):
        ij = np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)
        c = np.clip(ij[..., 0], 0, self.shape[1] - 1)
        r = np.clip(ij[..., 1], 0, self.shape[0] - 1)
        return r, c

    def height_at(self, xy: np.ndarray) -> np.ndarray:
        r, c = self.cell_rc(xy)
        return self.h[r, c]

    def is_indoor(self, xy: np.ndarray) -> np.ndarray:
        return self.height_at(xy) > 0

    def indoor_distance(self, xy: np.ndarray) -> np.ndarray:
        r, c = self.cell_rc(xy)
        return self.near_d[r, c]

    def outdoor_flat_index(self, xy: np.ndarray) -> np.ndarray:
        """Flat index of the cell itself if outdoor, else of its nearest outdoor cell."""
        r, c = self.cell_rc(xy)
        return self.near_r[r, c].astype(np.int64) * self.shape[1] + self.near_c[r, c]

    def cell_centers(self) -> np.ndarray:
        gr, gc = np.indices(self.shape)
        return np.stack([self.origin[0] + (gc + 0.5) * self.res, self.origin[1] + (gr + 0.5) * self.res], -1)

    def los_raytrace(self, ue_xy, site_xy, h_ut, h_bs, n_samples: int = 64) -> np.ndarray:
        """Brute-force ray test (reference for the cached viewsheds). -> bool [U,S]."""
        h_bs = np.broadcast_to(np.asarray(h_bs, float), (len(site_xy),))
        t = np.linspace(0.0, 1.0, n_samples + 2)[1:-1]
        p = ue_xy[:, None, None, :] + (site_xy[None, :, None, :] - ue_xy[:, None, None, :]) * t[None, None, :, None]
        ray_h = h_ut + (h_bs[None, :, None] - h_ut) * t[None, None, :]
        return np.all(self.height_at(p) < ray_h, axis=-1)


def make_synthetic_buildings(bbox, path: str, seed: int = 0, block_m: float = 110.0,
                             street_m: float = 22.0, rotation_deg: float = 36.0,
                             core_latlon=(34.0505, -118.2550)) -> str:
    """Write a *synthetic* DTLA-like building layer (GeoJSON) - NOT real data.

    Rotated street grid (DTLA's grid runs ~36 deg off north), 1-4 lots per
    block, lognormal heights with a high-rise core (Bunker Hill / Financial
    District). Used only when no real building file is supplied."""
    rng = np.random.default_rng(seed)
    s, w, n, e = bbox
    proj = LocalProjection((s + n) / 2, (w + e) / 2)
    lo, hi = proj.to_xy(s, w), proj.to_xy(n, e)
    core = proj.to_xy(*core_latlon)
    a = np.radians(rotation_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    span = np.hypot(*(hi - lo))
    pitch = block_m + street_m
    k = int(np.ceil(span / pitch)) + 1
    feats = []
    for i in range(-k, k):
        for j in range(-k, k):
            bx, by = i * pitch, j * pitch
            n_lots = rng.integers(1, 5)
            cuts = np.sort(rng.uniform(0.25, 0.75, n_lots - 1)) if n_lots > 1 else np.array([])
            edges = np.concatenate([[0.0], cuts, [1.0]]) * block_m
            for l in range(n_lots):
                if rng.random() < 0.12:            # plazas / parking lots
                    continue
                x0, x1 = bx + edges[l] + 3, bx + edges[l + 1] - 3
                y0, y1 = by + 3, by + block_m - 3
                if rng.random() < 0.4:             # setbacks
                    y0 += rng.uniform(0, 20)
                    y1 -= rng.uniform(0, 20)
                corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]) @ R.T
                ctr = corners[:4].mean(0)
                if np.any(ctr < lo) or np.any(ctr > hi):
                    continue
                dcore = np.hypot(*(ctr - core))
                hmed = 14.0 + 110.0 * np.exp(-(dcore / 450.0) ** 2)
                hgt = float(np.clip(rng.lognormal(np.log(hmed), 0.45), 6.0, 300.0))
                ll = proj.to_latlon(corners)
                feats.append({"type": "Feature", "properties": {"height": round(hgt, 1), "synthetic": True},
                              "geometry": {"type": "Polygon",
                                           "coordinates": [[[float(q[1]), float(q[0])] for q in ll]]}})
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return path


# --------------------------------------------------------------------------- #
# Hexagonal layout
# --------------------------------------------------------------------------- #
def hex_ring_sites(n_rings: int, isd: float, rot_deg: float = 0.0) -> np.ndarray:
    pts = []
    for q in range(-n_rings, n_rings + 1):
        for r in range(-n_rings, n_rings + 1):
            if abs(q + r) <= n_rings:
                pts.append((isd * (q + r / 2.0), isd * r * np.sqrt(3) / 2.0))
    pts = np.asarray(pts)
    order = np.lexsort((np.arctan2(pts[:, 1], pts[:, 0]), np.round(np.hypot(*pts.T), 3)))
    pts = pts[order]                         # centre site first, then ring by ring
    return _rotate(pts, rot_deg)


def hex_fill_sites(area_min, area_max, isd: float, center, rot_deg=0.0, margin=0.0,
                   polygon: Optional[np.ndarray] = None) -> np.ndarray:
    span = np.max(np.asarray(area_max) - np.asarray(area_min))
    n = int(np.ceil(span / isd)) + 2
    pts = _rotate(hex_ring_sites(n, isd), rot_deg) + np.asarray(center)
    lo, hi = np.asarray(area_min) + margin, np.asarray(area_max) - margin
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    if polygon is not None:
        keep &= points_in_polygon(pts, polygon)
    return pts[keep]


def _rotate(p: np.ndarray, deg: float) -> np.ndarray:
    if deg == 0:
        return p
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return p @ R.T


def _file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_building_raster(geo: GeometryConfig, proj, area_min, area_max, bbox):
    """Rasterise the building layer once and cache it (keyed by file hash + grid)."""
    from .placement import default_cache_dir
    cache_dir = geo.cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    path, source = geo.buildings_geojson, "file"
    if path is None:
        path = os.path.join(cache_dir, f"synthetic_dtla_{geo.synthetic_seed}_"
                            f"{hashlib.sha1(json.dumps(list(bbox)).encode()).hexdigest()[:8]}.geojson")
        if not os.path.exists(path):
            make_synthetic_buildings(bbox, path, seed=geo.synthetic_seed)
        source = "synthetic"
        warnings.warn("geometry.buildings_geojson not set: using a SYNTHETIC DTLA-like city. Run "
                      "tools/fetch_osm_buildings.py to import real OpenStreetMap buildings.", stacklevel=3)
    key = hashlib.sha1(json.dumps([_file_sha1(path), list(map(float, area_min)), list(map(float, area_max)),
                                   geo.building_raster_res_m, geo.building_default_height_m,
                                   geo.building_height_scale, geo.building_height_field]).encode()).hexdigest()[:20]
    cpath = os.path.join(cache_dir, f"raster_{key}.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        return BuildingRaster(z["h"], z["origin"], float(z["res"]), int(z["n"])), source, path
    r = BuildingRaster.from_file(path, proj, area_min, area_max, geo.building_raster_res_m,
                                 geo.building_default_height_m, geo.building_height_scale,
                                 geo.building_height_field)
    np.savez_compressed(cpath, h=r.h, origin=r.origin, res=r.res, n=r.n_buildings)
    return r, source, path


def macro_sites(het: HetNetConfig, proj, area_min, area_max, center, n_small: int,
                polygon: Optional[np.ndarray] = None) -> np.ndarray:
    """Macro O-RU positions [M, 2]. Explicit ``macro_latlons`` win; otherwise
    M = ``n_macro`` or max(1, round(n_small / mmwave_per_macro)). One macro sits at
    ``center``; several are spread evenly over the study area as the generators of
    a centroidal Voronoi tessellation (Lloyd's algorithm on an area grid), which
    gives each macro an umbrella cell of about equal area."""
    if het.macro_latlons:
        ll = np.asarray(het.macro_latlons, float).reshape(-1, 2)
        return proj.to_xy(ll[:, 0], ll[:, 1]).reshape(-1, 2)
    m = het.n_macro if het.n_macro is not None else int(np.floor(n_small / het.mmwave_per_macro + 0.5))
    m = max(int(m), 1)
    if m == 1:
        return np.asarray(center, float).reshape(1, 2)
    lo, hi = np.asarray(area_min, float), np.asarray(area_max, float)
    step = float(np.sqrt(np.prod(hi - lo) / 20000.0))            # ~20k grid points
    gx = np.arange(lo[0] + step / 2, hi[0], step)
    gy = np.arange(lo[1] + step / 2, hi[1], step)
    pts = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    if polygon is not None:
        inside = points_in_polygon(pts, polygon)
        pts = pts[inside] if inside.any() else pts
    rng = np.random.default_rng(het.macro_layout_seed)
    # k-means++ initialisation, then Lloyd iterations
    c = [pts[rng.integers(len(pts))]]
    for _ in range(m - 1):
        d2 = np.min(((pts[:, None, :] - np.asarray(c)[None]) ** 2).sum(-1), 1)
        c.append(pts[rng.choice(len(pts), p=d2 / d2.sum())])
    c = np.asarray(c)
    for _ in range(100):
        lab = np.argmin(((pts[:, None, :] - c[None]) ** 2).sum(-1), 1)
        new = np.stack([pts[lab == k].mean(0) if np.any(lab == k) else c[k] for k in range(m)])
        if np.allclose(new, c, atol=0.1):
            c = new
            break
        c = new
    return c


def build_deployment(geo: GeometryConfig, radio: Optional[RadioConfig] = None,
                     pois: Sequence[POI] = DTLA_POIS, hetnet: Optional[HetNetConfig] = None) -> Deployment:
    """Study area, buildings and sites. With an enabled ``hetnet`` the macro O-RUs are
    the first sites (tier 0, see :func:`macro_sites`) and the ``geo.layout`` sites
    (minus any within ``macro_exclusion_m`` of a macro) are the small-cell O-RUs (tier 1)."""
    radio = radio or RadioConfig()
    het = hetnet is not None and hetnet.enabled
    tiers = [hetnet.macro, hetnet.small] if het else []
    small_radio = tiers[1].to_radio(radio) if het else radio
    s, w, n, e = geo.bbox
    poly_xy = None
    if geo.area_geojson:
        ring, _ = next(_iter_polygons(_load_geojson(geo.area_geojson)))
        s, n = ring[:, 1].min(), ring[:, 1].max()
        w, e = ring[:, 0].min(), ring[:, 0].max()
    lat0, lon0 = (s + n) / 2.0, (w + e) / 2.0
    proj = LocalProjection(lat0, lon0)
    area_min = proj.to_xy(s, w)
    area_max = proj.to_xy(n, e)
    if geo.area_geojson:
        poly_xy = proj.to_xy(ring[:, 1], ring[:, 0])

    need_bld = geo.layout == "viewshed" or radio.los_model == "geometry" or geo.buildings_geojson
    bld, source = None, "none"
    if need_bld:
        bld, source, _ = load_building_raster(geo, proj, area_min, area_max, (s, w, n, e))

    center = proj.to_xy(*geo.center_latlon) if geo.center_latlon else np.zeros(2)
    placement = None
    az_off = None
    if geo.layout == "viewshed":
        from .placement import place_sites
        placement = place_sites(bld, geo, small_radio)
        sites, site_h, az_off = placement.site_xy, placement.site_h, placement.az_offset
    elif geo.layout == "rings":
        sites = hex_ring_sites(geo.n_rings, geo.isd_m, geo.lattice_rotation_deg) + center
    elif geo.layout == "fill":
        sites = hex_fill_sites(area_min, area_max, geo.isd_m, center, geo.lattice_rotation_deg,
                               geo.fill_margin_m, poly_xy)
    else:
        raise ValueError(f"unknown layout {geo.layout}")
    if len(sites) == 0:
        raise ValueError("no sites inside the study area")
    if geo.layout != "viewshed":
        site_h = np.full(len(sites), small_radio.h_bs_m)
        az_off = np.zeros(len(sites))
        if bld is not None:   # a hex site inside a footprint sits on that roof (+ mast) if taller
            site_h = np.maximum(site_h, bld.height_at(sites) + geo.mast_height_m)
    site_h = np.asarray(site_h, float)
    keep = np.ones(len(sites), bool)
    site_tier = np.zeros(len(sites), int)

    n_macro = 0
    if het:
        mxy = macro_sites(hetnet, proj, area_min, area_max, center, len(sites), poly_xy)
        n_macro = len(mxy)
        dmin = np.hypot(*(sites[:, None, :] - mxy[None, :, :]).transpose(2, 0, 1)).min(1)
        keep = dmin >= hetnet.macro_exclusion_m
        mh = np.full(n_macro, hetnet.macro_height_m)
        if bld is not None:
            mh = np.maximum(mh, bld.height_at(mxy) + geo.mast_height_m)
        sites = np.vstack([mxy, sites[keep]])
        site_h = np.concatenate([mh, site_h[keep]])
        az_off = np.concatenate([np.zeros(n_macro), az_off[keep]])
        site_tier = np.concatenate([np.zeros(n_macro, int), np.ones(int(keep.sum()), int)])
    n_sites = len(sites)

    site_vis = None
    if bld is not None and radio.los_model == "geometry":
        from .placement import site_viewsheds
        if placement is not None:
            site_vis = placement.vis[keep]
            if het:
                mvis = site_viewsheds(bld, sites[:n_macro], site_h[:n_macro], radio.h_ut_m,
                                      max(geo.viewshed_radius_m, 1500.0), geo.cache_dir)
                site_vis = np.vstack([mvis, site_vis])
        else:
            site_vis = site_viewsheds(bld, sites, site_h, radio.h_ut_m, max(geo.viewshed_radius_m, 1500.0),
                                      geo.cache_dir)

    if geo.tilt_ground_m:
        tilt = np.degrees(np.arctan2(site_h - radio.h_ut_m, geo.tilt_ground_m))
    else:
        tilt = np.full(n_sites, radio.downtilt_deg)
    for t, tc in enumerate(tiers):        # a tier with a fixed downtilt overrides the ground-aim rule
        if tc.downtilt_deg is not None:
            tilt[site_tier == t] = tc.downtilt_deg

    sector_site = np.repeat(np.arange(n_sites), 3)
    sector_az = (np.tile(np.asarray(geo.sector_azimuths_deg, float), n_sites) + np.repeat(az_off, 3)) % 360.0

    in_box = [p for p in pois if s <= p.lat <= n and w <= p.lon <= e]
    poi_xy = proj.to_xy([p.lat for p in in_box], [p.lon for p in in_box]).reshape(-1, 2)

    return Deployment(proj=proj, area_min=area_min, area_max=area_max, site_xy=sites,
                      sector_site=sector_site, sector_az_deg=sector_az, poi_xy=poi_xy,
                      poi_names=[p.name for p in in_box], poi_tags=[p.tags for p in in_box],
                      poi_weights=np.asarray([p.weight for p in in_box], float),
                      buildings=bld, area_polygon_xy=poly_xy, site_h=site_h,
                      site_tilt_deg=tilt, site_vis=site_vis, placement=placement, buildings_source=source,
                      site_tier=site_tier, tier_names=tuple(t.name for t in tiers) or ("cell",))


def bbox_geojson(bbox, name="Downtown Los Angeles") -> dict:
    s, w, n, e = bbox
    return {"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"name": name},
        "geometry": {"type": "Polygon",
                     "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}]}
