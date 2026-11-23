#!/usr/bin/env python3
"""Map of the scan trajectory, used to document the pose data and to check its scale.

Reads every pose file (3x4 world-to-camera matrix [R | t]), computes the camera centres C = -R^T t and writes:
  pose_map.pdf / pose_map.png   top-down view and side view of the trajectory, coloured by deck level
  pose_stats.json               extents, deck levels and their spacing, frames per deck, frames outside the hull

Example
  python src/pose_map.py --db-poses data/latest/poses \
      --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json --out runs_final/pose_map

Notes
- The vertical axis is taken from --up-axis (default 2). The two horizontal axes are rotated so that the first
  one follows the longest direction of the trajectory (the length of the ship).
- Deck levels are the peaks of the camera-height histogram. Use --decks N to force a number of levels.
- A frame is reported as "outside the hull" if it lies more than --hull-half-width from the centre line.
  This is a simple geometric rule. Look at a few of the reported images before stating it in the report.
- --json is optional. If given, the annotated scan images (gt_image_ids) are drawn as black markers.
"""
import argparse, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def num(path):
    m = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return int(m[-1]) if m else None


def load_centres(folder):
    files = sorted((f for f in glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True) if num(f) is not None), key=num)
    ids, C, F, bad = [], [], [], 0
    for f in files:
        try:
            T = np.loadtxt(f)
            R, t = T[:3, :3], T[:3, 3]
            if not np.isfinite(T[:3]).all(): raise ValueError
        except Exception:
            bad += 1; continue
        ids.append(num(f)); C.append(-R.T @ t); F.append(R[2])
    if not ids: raise SystemExit(f"no readable pose files in {folder}")
    if bad: print(f"[poses] {bad} files could not be read and were skipped")
    return np.array(ids), np.array(C), np.array(F)


def deck_levels(h, n_decks, min_gap, min_share):
    """Peaks of the height histogram (5 cm bins, smoothed), lowest first."""
    if n_decks:
        from sklearn.cluster import KMeans
        return np.sort(KMeans(n_decks, n_init=10, random_state=0).fit(h[:, None]).cluster_centers_[:, 0])
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    edges = np.arange(h.min() - 0.5, h.max() + 0.55, 0.05)
    hist, _ = np.histogram(h, edges)
    smooth = gaussian_filter1d(hist.astype(float), 3)                       # sigma 15 cm
    peaks, _ = find_peaks(smooth, distance=int(min_gap / 0.05), prominence=smooth.max() * 0.08)
    centres = (edges[:-1] + edges[1:]) / 2
    levels = centres[peaks]
    lab = np.abs(h[:, None] - levels[None]).argmin(1)
    keep = [k for k in range(len(levels)) if (lab == k).mean() >= min_share]   # drop tiny peaks
    return levels[keep]


def gt_ids(path):
    items = json.load(open(path))
    if isinstance(items, dict): items = items.get("items", list(items.values()))
    out = set()
    for it in items:
        for x in it.get("gt_image_ids", []) or []:
            try: out.add(int(x))
            except (TypeError, ValueError): pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-poses", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--json", help="IHM json: annotated scan images are drawn as markers")
    ap.add_argument("--up-axis", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--decks", type=int, default=0, help="force this number of deck levels (default: detect)")
    ap.add_argument("--deck-min-gap", type=float, default=1.2, help="smallest height difference between two decks")
    ap.add_argument("--deck-min-share", type=float, default=0.02, help="a level needs at least this share of the frames")
    ap.add_argument("--hull-half-width", type=float, default=6.0, help="frames further than this from the centre line count as outside the hull")
    ap.add_argument("--unit", default="m", help="unit written on the axes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ids, C, F = load_centres(a.db_poses)
    print(f"[poses] {len(ids)} poses, ids {ids.min()}..{ids.max()}")
    det = np.linalg.det(np.loadtxt(sorted(glob.glob(os.path.join(a.db_poses, '**', '*.txt'), recursive=True))[0])[:3, :3])

    # ---- ship frame: along (longest horizontal direction), across, height
    hor = [i for i in range(3) if i != a.up_axis]
    H = C[:, hor]; mid = np.median(H, axis=0)
    _, _, V = np.linalg.svd(H - mid, full_matrices=False)
    along, across = (H - mid) @ V[0], (H - mid) @ V[1]
    height = C[:, a.up_axis]
    angle = float(np.degrees(np.arctan2(abs(V[0][1]), abs(V[0][0]))))

    # ---- outside the hull, decks
    centre_line = np.median(across)
    outside = np.abs(across - centre_line) > a.hull_half_width
    inside = ~outside
    levels = deck_levels(height[inside], a.decks, a.deck_min_gap, a.deck_min_share)
    deck = np.abs(height[:, None] - levels[None]).argmin(1)

    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    ext = lambda x: float(np.ptp(x))
    rob = lambda x: float(np.percentile(x, 99) - np.percentile(x, 1))
    stats = dict(
        n_poses=int(len(ids)), id_min=int(ids.min()), id_max=int(ids.max()), det_R_first=float(det),
        ship_axis_angle_to_first_horizontal_axis_deg=round(angle, 1),
        median_step_between_frames=round(float(np.median(step)), 4), path_length=round(float(step[step < 1.0].sum()), 1),
        extent_all=dict(along=round(ext(along), 2), across=round(ext(across), 2), height=round(ext(height), 2)),
        extent_inside_hull=dict(along=round(ext(along[inside]), 2), across=round(ext(across[inside]), 2), height=round(ext(height[inside]), 2)),
        extent_inside_hull_1_99=dict(along=round(rob(along[inside]), 2), across=round(rob(across[inside]), 2), height=round(rob(height[inside]), 2)),
        deck_levels=[round(float(x), 2) for x in levels], deck_spacing=[round(float(x), 2) for x in np.diff(levels)],
        frames_per_deck=[int((deck[inside] == k).sum()) for k in range(len(levels))],
        outside_hull=dict(n=int(outside.sum()), share=round(float(outside.mean()), 4), hull_half_width=a.hull_half_width,
                          id_min=int(ids[outside].min()) if outside.any() else None, id_max=int(ids[outside].max()) if outside.any() else None),
    )

    marks = None
    if a.json:
        g = gt_ids(a.json); marks = np.isin(ids, list(g))
        stats["annotated_images"] = dict(listed=len(g), with_pose=int(marks.sum()), outside_hull=int((marks & outside).sum()),
                                         per_deck=[int((marks & inside & (deck == k)).sum()) for k in range(len(levels))])
    json.dump(stats, open(os.path.join(a.out, "pose_stats.json"), "w"), indent=2)

    # ---- figure
    cols = ["#014693", "#3CBFBE", "#ED8013", "#B01B81", "#7A8F00", "#575757", "#F7D019", "#8C564B"]
    u = a.unit
    th = lambda n: f"{int(n):,}".replace(",", " ")
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[ext(across) + 2, ext(height) + 2], hspace=0.08))
    for k in range(len(levels)):
        m = inside & (deck == k)
        lab = f"Level {k + 1} ({levels[k]:+.1f} {u}, {th(m.sum())} frames)"
        ax[0].scatter(along[m], across[m], s=0.6, color=cols[k % len(cols)], rasterized=True, linewidths=0)
        ax[1].scatter(along[m], height[m], s=0.6, color=cols[k % len(cols)], rasterized=True, linewidths=0)
        ax[1].scatter([], [], s=18, color=cols[k % len(cols)], label=lab)
    if outside.any():
        ax[0].scatter(along[outside], across[outside], s=0.6, color="#B0B0B0", rasterized=True, linewidths=0)
        ax[1].scatter(along[outside], height[outside], s=0.6, color="#B0B0B0", rasterized=True, linewidths=0)
        ax[1].scatter([], [], s=18, color="#B0B0B0", label=f"Outside the hull ({th(outside.sum())} frames)")
    if marks is not None and marks.any():
        for x, y in ((ax[0], across), (ax[1], height)):
            x.scatter(along[marks], y[marks], s=14, marker="x", color="black", linewidths=0.8, zorder=5)
        ax[1].scatter([], [], s=18, marker="x", color="black", linewidths=0.8, label=f"Annotated scan image ({marks.sum()})")
    ax[0].set_aspect("equal"); ax[1].set_aspect("equal")
    ax[0].set_ylabel(f"Across ({u})"); ax[1].set_ylabel(f"Height ({u})"); ax[1].set_xlabel(f"Along the ship ({u})")
    ax[0].set_title("Top-down view", fontsize=9, loc="left"); ax[1].set_title("Side view", fontsize=9, loc="left")
    for x in ax:
        x.grid(True, color="#E5E5E5", linewidth=0.5); x.set_axisbelow(True); x.tick_params(labelsize=8)
        for s in ("top", "right"): x.spines[s].set_visible(False)
    fig.legend(*ax[1].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"pose_map.{e}"), dpi=300, bbox_inches="tight")

    # ---- report
    s = stats
    print(f"[check] det R = {s['det_R_first']:.6f} (a rotation has 1), median step between frames {s['median_step_between_frames']}")
    print(f"[extent] all frames:        along {s['extent_all']['along']}, across {s['extent_all']['across']}, height {s['extent_all']['height']}")
    print(f"[extent] inside the hull:   along {s['extent_inside_hull']['along']}, across {s['extent_inside_hull']['across']}, height {s['extent_inside_hull']['height']}")
    print(f"[decks] {len(levels)} levels at {s['deck_levels']}, spacing {s['deck_spacing']}, frames {s['frames_per_deck']}")
    o = s["outside_hull"]
    print(f"[hull] {o['n']} frames ({100 * o['share']:.1f}%) lie more than {a.hull_half_width} from the centre line" + (f", ids {o['id_min']}..{o['id_max']}" if o["n"] else ""))
    if marks is not None:
        m = s["annotated_images"]; print(f"[ihm] {m['with_pose']}/{m['listed']} annotated images have a pose, per level {m['per_deck']}, outside the hull {m['outside_hull']}")
    print(f"saved {os.path.join(a.out, 'pose_map.pdf')}, pose_map.png and pose_stats.json")


if __name__ == "__main__":
    main()
