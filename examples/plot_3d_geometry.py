"""Publication figure: 3-D view of the imported building geometry with the
rooftop base-station sites and a UE snapshot from the simulator.

    python examples/plot_3d_geometry.py --buildings data/dtla_buildings.geojson --out results
    python examples/plot_3d_geometry.py --hour 18 --extent -800 800 -600 600 --zexag 1.5
    python examples/plot_3d_geometry.py --buildings data/ucr_buildings.geojson \
        --bbox 33.970067 -117.331924 33.975550 -117.323470
    python examples/plot_3d_geometry.py --synthetic            # preview without OSM data

Buildings are extruded from the exact GeoJSON footprints (same height parsing as
the simulator, so the heights match the LOS raster). Sites and sector azimuths
come from the same deployment as the simulation study (``--preset``), and the UE
positions are a TVC-mobility snapshot after ``--warmup`` simulated seconds at
``--hour``.

Rendering notes (matplotlib's 3-D painter's algorithm):
  * walls facing away from the camera are culled, so the plot is meant for a
    fixed ``--elev/--azim`` and not for interactive rotation;
  * UEs and sites are drawn above the buildings. UEs hidden from the camera by
    a building (incl. indoor UEs) are found by ray-marching the height raster
    and drawn faded (``--occluded fade``), dropped (``hide``) or as is (``show``);
  * ``--zexag`` exaggerates heights; the factor is printed so it can go in the caption.

Outputs <out>/<name>.pdf (vector text, rasterised geometry, TrueType fonts) and .png.
"""
import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from oran_asm_sim.config import preset  # noqa: E402
from oran_asm_sim.geometry import _iter_polygons, _load_geojson, _parse_height, load_building_raster  # noqa: E402
from oran_asm_sim.simulator import ORANSystemSimulator  # noqa: E402

LIGHT_AZ, LIGHT_EL = 315.0, 45.0           # cartographic convention: light from the NW
C_SITE, C_UE, C_CAR, C_OUT = "#c1121f", "#1d4e89", "#e09f3e", "#000000"


# --------------------------------------------------------------------------- geometry
def clip_ring(ring, x0, x1, y0, y1):
    """Sutherland-Hodgman clip of a polygon ring [N,2] to an axis-aligned box."""
    def clip(pts, inside, inter):
        out = []
        for i in range(len(pts)):
            p, q = pts[i - 1], pts[i]
            if inside(q):
                if not inside(p):
                    out.append(inter(p, q))
                out.append(q)
            elif inside(p):
                out.append(inter(p, q))
        return out

    def ix(p, q, x):
        t = (x - p[0]) / (q[0] - p[0])
        return (x, p[1] + t * (q[1] - p[1]))

    def iy(p, q, y):
        t = (y - p[1]) / (q[1] - p[1])
        return (p[0] + t * (q[0] - p[0]), y)

    pts = [tuple(p) for p in ring]
    for inside, inter in ((lambda p: p[0] >= x0, lambda p, q: ix(p, q, x0)),
                          (lambda p: p[0] <= x1, lambda p, q: ix(p, q, x1)),
                          (lambda p: p[1] >= y0, lambda p, q: iy(p, q, y0)),
                          (lambda p: p[1] <= y1, lambda p, q: iy(p, q, y1))):
        if not pts:
            break
        pts = clip(pts, inside, inter)
    return np.asarray(pts, float)


def load_footprints(path, dep, geo, extent):
    """[(ring_xy [N,2] counter-clockwise, height_m)] clipped to ``extent``."""
    x0, x1, y0, y1 = extent
    out = []
    for ring, props in _iter_polygons(_load_geojson(path)):
        xy = dep.proj.to_xy(ring[:, 1], ring[:, 0]).reshape(-1, 2)
        if np.allclose(xy[0], xy[-1]):
            xy = xy[:-1]
        if len(xy) < 3 or xy[:, 0].max() < x0 or xy[:, 0].min() > x1 or xy[:, 1].max() < y0 or xy[:, 1].min() > y1:
            continue
        xy = clip_ring(xy, x0, x1, y0, y1)
        if len(xy) < 3:
            continue
        area2 = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
        if abs(area2) < 2.0:                       # < 1 m^2 sliver
            continue
        if area2 < 0:
            xy = xy[::-1]
        h = _parse_height(props, geo.building_default_height_m, geo.building_height_scale,
                          geo.building_height_field)
        out.append((xy, float(h)))
    return out


def building_faces(footprints, view_az, cmap, norm):
    """Roof + camera-facing wall polygons with Lambertian shading."""
    la, le = np.radians(LIGHT_AZ), np.radians(LIGHT_EL)
    light_h = np.array([np.cos(la), np.sin(la)]) * np.cos(le)
    view_h = np.array([np.cos(np.radians(view_az)), np.sin(np.radians(view_az))])
    amb, dif = 0.55, 0.45
    faces, colors = [], []
    for xy, h in footprints:
        base = np.asarray(cmap(norm(h)))
        roof = np.column_stack([xy, np.full(len(xy), h)])
        nxt = np.roll(xy, -1, axis=0)
        e = nxt - xy
        n = np.column_stack([e[:, 1], -e[:, 0]])      # outward normal for a CCW ring
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-6
        n[ok] /= ln[ok, None]
        for i in np.where(ok & (n @ view_h > 0))[0]:
            a, b = xy[i], nxt[i]
            faces.append([(a[0], a[1], 0.0), (b[0], b[1], 0.0), (b[0], b[1], h), (a[0], a[1], h)])
            s = amb + dif * max(0.0, float(n[i] @ light_h))
            colors.append(np.r_[base[:3] * s, 1.0])
        faces.append(roof)
        colors.append(np.r_[base[:3] * (amb + dif * np.sin(le)), 1.0])
    return faces, colors


def occluded(pts, raster, view_az, view_el, zexag, step=2.5):
    """True where the straight line from a 3-D point towards the (orthographic) camera
    passes through a building of the height raster (indoor points count as occluded)."""
    d = np.array([np.cos(np.radians(view_az)), np.sin(np.radians(view_az))])
    # with heights exaggerated by zexag, a display slope tan(el) is a data slope tan(el)/zexag
    rise = np.tan(np.radians(view_el)) / zexag
    hmax = float(raster.h.max())
    occ = raster.height_at(pts[:, :2]) > pts[:, 2]
    s = step
    while True:
        z = pts[:, 2] + s * rise
        if z.min() > hmax:
            break
        occ |= raster.height_at(pts[:, :2] + s * d) > z
        s += step
    return occ


# --------------------------------------------------------------------------- figure
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buildings", default="data/dtla_buildings.geojson", help="OSM GeoJSON (tools/fetch_osm_buildings.py)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("S", "W", "N", "E"),
                    help="study area in degrees (default: the preset's bbox, Downtown LA); must match "
                         "the area the --buildings file was fetched for")
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic city instead (preview only)")
    ap.add_argument("--preset", default="dtla_full")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", default="geometry_3d")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hour", type=float, default=12.0, help="time of day of the UE snapshot")
    ap.add_argument("--warmup", type=float, default=60.0, help="simulated seconds before the snapshot")
    ap.add_argument("--extent", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="crop in local metres (default: whole study area)")
    ap.add_argument("--elev", type=float, default=32.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--zexag", type=float, default=2.0, help="vertical exaggeration")
    ap.add_argument("--persp", action="store_true", help="perspective instead of orthographic projection")
    ap.add_argument("--zoom", type=float, default=1.35, help="3-D axes zoom (crop the empty margins)")
    ap.add_argument("--zaxis", action="store_true", help="draw a height axis (default: colour bar only)")
    ap.add_argument("--occluded", choices=("fade", "hide", "show"), default="fade")
    ap.add_argument("--links", action="store_true", help="draw UE -> serving-site links")
    ap.add_argument("--pois", action="store_true", help="label the TVC community POIs")
    ap.add_argument("--sector-len", type=float, default=70.0, help="length of the sector boresight ticks [m]")
    ap.add_argument("--width", type=float, default=7.16, help="figure width [in] (IEEE: 3.5 / 7.16)")
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--font", default="serif")
    ap.add_argument("--fontsize", type=float, default=8.0)
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()

    geo_over = {}
    if not a.synthetic:
        if not os.path.exists(a.buildings):
            sys.exit(f"{a.buildings} not found. Run tools/fetch_osm_buildings.py first, or pass --synthetic "
                     "for a preview with the synthetic city.")
        geo_over["buildings_geojson"] = a.buildings
    if a.bbox:
        geo_over["bbox"] = tuple(a.bbox)
    cfg = preset(a.preset, geometry=geo_over, env={"seed": a.seed, "warmup_s": 0.0})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if a.synthetic else "default")
        sim = ORANSystemSimulator(cfg, seed=a.seed)
    dep, geo = sim.dep, cfg.geometry
    if dep.buildings is None:
        sys.exit("this preset has no building layer")
    raster, source, path = load_building_raster(geo, dep.proj, dep.area_min, dep.area_max, geo.bbox)
    print(f"buildings: {path} ({source})")

    # UE snapshot
    sim.reset(a.seed, a.hour)
    n_steps = int(round(a.warmup / cfg.env.decision_interval_s))
    for _ in range(n_steps):
        sim.step(np.zeros(sim.S, int))
    ue_xy = sim.mob.xy.copy()
    indoor = dep.buildings.is_indoor(ue_xy)
    car = sim.mob.in_car & ~indoor
    serv = sim.serving.copy()

    ext = a.extent or (dep.area_min[0], dep.area_max[0], dep.area_min[1], dep.area_max[1])
    x0, x1, y0, y1 = ext
    inb = lambda p: (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)  # noqa: E731

    fps = load_footprints(path, dep, geo, ext)
    hmax = max(h for _, h in fps)
    print(f"{len(fps)} footprints in view (max height {hmax:.0f} m), {dep.n_sites} sites, {sim.U} UEs")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    plt.rcParams.update({"font.family": a.font, "font.size": a.fontsize, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.labelsize": a.fontsize, "legend.fontsize": a.fontsize - 1,
                         "xtick.labelsize": a.fontsize - 1, "ytick.labelsize": a.fontsize - 1})
    cmap = LinearSegmentedColormap.from_list("bld", ["#eceae4", "#a9b4bf", "#4f5d6e"])
    hnorm = Normalize(0, np.ceil(hmax / 25) * 25)

    W = a.width
    H = a.height or W * 0.5
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.set_proj_type("persp", focal_length=0.6) if a.persp else ax.set_proj_type("ortho")
    ax.view_init(elev=a.elev, azim=a.azim)

    # ground (study area)
    ground = Poly3DCollection([[(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)]],
                              facecolor="#f7f7f5", edgecolor="0.55", lw=0.4, zorder=0)
    ax.add_collection3d(ground)

    faces, colors = building_faces(fps, a.azim, cmap, hnorm)
    bc = Poly3DCollection(faces, facecolors=colors, edgecolors=(0.2, 0.2, 0.2, 0.35), linewidths=0.08,
                          zsort="average", zorder=1)
    bc.set_rasterized(True)
    ax.add_collection3d(bc)

    # sites: mast (roof -> antenna), antenna marker, sector boresights with downtilt
    s_in = inb(dep.site_xy)
    roof = dep.buildings.height_at(dep.site_xy)
    sxyz = np.column_stack([dep.site_xy, dep.site_h])
    masts = [[(x, y, r), (x, y, h)] for (x, y, h), r in zip(sxyz[s_in], roof[s_in])]
    ax.add_collection3d(Line3DCollection(masts, colors="k", linewidths=0.7, zorder=4))
    ticks = []
    for s in np.where(np.repeat(s_in, 3))[0]:
        i = dep.sector_site[s]
        az = np.radians(dep.sector_az_deg[s])
        L = a.sector_len
        drop = L * np.tan(np.radians(dep.site_tilt_deg[i]))
        p = sxyz[i]
        ticks.append([tuple(p), (p[0] + L * np.cos(az), p[1] + L * np.sin(az), p[2] - drop)])
    ax.add_collection3d(Line3DCollection(ticks, colors=C_SITE, linewidths=0.9, zorder=5))
    ax.scatter(*sxyz[s_in].T, marker="^", s=22, c=C_SITE, edgecolors="k", linewidths=0.4,
               depthshade=False, zorder=6)

    # UEs
    u_in = inb(ue_xy)
    uxyz = np.column_stack([ue_xy, np.full(sim.U, cfg.radio.h_ut_m)])
    occ = occluded(uxyz, dep.buildings, a.azim, a.elev, a.zexag) if a.occluded != "show" else np.zeros(sim.U, bool)
    if a.occluded == "hide":
        u_in &= ~occ
    if a.links:
        m = u_in & (serv >= 0)
        m[m] = s_in[dep.sector_site[serv[m]]]          # only links to sites inside the view
        tgt = sxyz[dep.sector_site[serv[m]]]
        ax.add_collection3d(Line3DCollection(np.stack([uxyz[m], tgt], 1), colors=(0.11, 0.30, 0.54, 0.18),
                                             linewidths=0.3, zorder=2))
    groups = [("outdoor UE", u_in & ~indoor & ~car & (serv >= 0), dict(marker="o", c=C_UE)),
              ("in-car UE", u_in & car & (serv >= 0), dict(marker="s", c=C_CAR)),
              ("indoor UE", u_in & indoor & (serv >= 0), dict(marker="o", facecolors="none", edgecolors=C_UE)),
              ("UE in outage", u_in & (serv < 0), dict(marker="x", c=C_OUT))]
    for _, m, kw in groups:
        for vis, alpha in ((~occ, 0.95), (occ, 0.25)):
            mm = m & vis
            if mm.any():
                ax.scatter(*uxyz[mm].T, s=4, linewidths=0.5, alpha=alpha, depthshade=False, zorder=3, **kw)

    if a.pois:
        p_in = inb(dep.poi_xy)
        for (x, y), name in zip(dep.poi_xy[p_in], np.asarray(dep.poi_names)[p_in]):
            ax.text(x, y, 2.0, name, fontsize=a.fontsize - 2, zorder=7, ha="center")

    for art in list(ax.collections) + list(ax.texts):   # 3-D artists are clipped to the axes box otherwise
        art.set_clip_on(False)

    # axes: true horizontal proportions, exaggerated heights, km ticks
    zt = hnorm.vmax
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(0, zt)
    ax.set_box_aspect((x1 - x0, y1 - y0, zt * a.zexag), zoom=a.zoom)
    km = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    ax.xaxis.set_major_formatter(km)
    ax.yaxis.set_major_formatter(km)
    ax.set_xlabel("x east [km]", labelpad=-2)
    ax.set_ylabel("y north [km]", labelpad=-2)
    ax.tick_params(pad=-1)
    if a.zaxis:
        ax.set_zlabel("height [m]", labelpad=-2)
        ax.set_zticks(np.arange(0, zt + 1, 50 if zt > 100 else 25))
    else:                                   # heights are read from the colour bar
        ax.set_zticks([])
        ax.zaxis.line.set_color((1, 1, 1, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis._axinfo["grid"]["linewidth"] = 0.0

    handles = [Line2D([], [], marker="^", ls="", mfc=C_SITE, mec="k", mew=0.4, ms=6, label="BS site (antenna)"),
               Line2D([], [], c=C_SITE, lw=1, label="sector boresight")]
    ue_handles = {"outdoor UE": dict(marker="o", mfc=C_UE, mec=C_UE),
                  "in-car UE": dict(marker="s", mfc=C_CAR, mec=C_CAR),
                  "indoor UE": dict(marker="o", mfc="none", mec=C_UE),
                  "UE in outage": dict(marker="x", c=C_OUT)}
    for label, m, _ in groups:
        if m.any():
            handles.append(Line2D([], [], ls="", ms=3, label=label, **ue_handles[label]))
    if a.links:
        handles.append(Line2D([], [], c=(0.11, 0.30, 0.54, 0.5), lw=0.6, label="serving link"))
    if a.occluded == "fade":
        handles.append(Line2D([], [], marker="o", ls="", mfc=C_UE, mec=C_UE, ms=3, alpha=0.25,
                              label="hidden by buildings"))
    fig.subplots_adjust(left=0.0, right=0.9, bottom=0.08, top=1.0)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 0.0), frameon=False,
               ncol=min(len(handles), 4 if len(handles) > 6 else 6) if W > 5 else 3,
               handletextpad=0.3, columnspacing=1.2)
    sm = matplotlib.cm.ScalarMappable(norm=hnorm, cmap=cmap)
    cax = fig.add_axes([0.87, 0.3, 0.013, 0.45])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("building height [m]")
    cb.outline.set_linewidth(0.4)

    os.makedirs(a.out, exist_ok=True)
    for ext_ in ("pdf", "png"):
        f = os.path.join(a.out, f"{a.name}.{ext_}")
        fig.savefig(f, dpi=a.dpi, bbox_inches="tight", pad_inches=0.02)
        print("saved", f)
    n_vis = int(u_in.sum())
    print(f"caption facts: {len(fps)} buildings ({source}), {int(s_in.sum())} sites / {int(3 * s_in.sum())} sectors, "
          f"{n_vis} UEs shown at {a.hour:05.2f} h after {a.warmup:g} s of TVC mobility, "
          f"{100 * indoor[u_in].mean():.0f} % indoor; heights exaggerated {a.zexag:g}x; "
          f"{'perspective' if a.persp else 'orthographic'} view elev={a.elev:g} azim={a.azim:g}.")


if __name__ == "__main__":
    main()
