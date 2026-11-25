"""Map of the scan trajectory, to document the pose data and check its scale.

    python main.py pose-map --db-poses data/latest/poses \\
        --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json --out runs/pose_map

Files written to --out
    pose_map.png                 top-down and side view of the camera positions, coloured by deck level
    pose_stats.json              extents, deck levels and their spacing, frames per deck, frames outside the hull

Notes
- The vertical axis is --up-axis. The horizontal axes are rotated so that the first follows the longest
  direction of the trajectory (the length of the ship).
- Deck levels are the peaks of the camera-height histogram. --decks N forces N levels (k-means).
- A frame counts as outside the hull if it lies more than --hull-half-width from the centre line. This is
  a simple geometric rule: look at a few of the reported images before stating it in the report.
- With --json, the annotated scan images (gt_image_ids) are drawn as black crosses.
"""
import glob
import json
import os

import numpy as np

from .dataset import load_items
from .trajectory import add_deck_arguments, camera_centres, decks_from_args, load_pose_matrices

HELP = "map of the scan trajectory by deck, with the scale check (pose_stats.json)"

LEVEL_COLOURS = ["#014693", "#3CBFBE", "#ED8013", "#B01B81", "#7A8F00", "#575757", "#F7D019", "#8C564B"]
OUTSIDE_COLOUR = "#B0B0B0"
MAX_STEP = 1.0                    # steps longer than this are jumps between passes, not walking


def add_arguments(parser):
    parser.add_argument("--db-poses", required=True, help="folder with the pose files")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--json", help="IHM json: annotated scan images are drawn as markers")
    add_deck_arguments(parser)
    parser.add_argument("--unit", default="m", help="unit written on the axes")


def run(args):
    os.makedirs(args.out, exist_ok=True)
    ids, matrices = load_pose_matrices(args.db_poses)
    centres = camera_centres(matrices)
    print(f"[poses] {len(ids)} poses, ids {ids.min()}..{ids.max()}")
    first_file = sorted(glob.glob(os.path.join(args.db_poses, "**", "*.txt"), recursive=True))[0]
    det = np.linalg.det(np.loadtxt(first_file)[:3, :3])

    d = decks_from_args(centres, args)
    along, across, height, angle = d["along"], d["across"], d["height"], d["angle"]
    outside, levels, deck = d["outside"], d["levels"], d["deck"]
    inside = ~outside

    stats = statistics(ids, centres, along, across, height, angle, det, levels, deck, outside, args.hull_half_width)
    annotated = None

    if args.json:
        listed = annotated_ids(args.json)
        annotated = np.isin(ids, list(listed))
        stats["annotated_images"] = dict(
            listed=len(listed), with_pose=int(annotated.sum()), outside_hull=int((annotated & outside).sum()),
            per_deck=[int((annotated & inside & (deck == k)).sum()) for k in range(len(levels))])
        
    with open(os.path.join(args.out, "pose_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    plot(along, across, height, levels, deck, outside, annotated, args.unit, args.out)
    print_summary(stats, args.hull_half_width, annotated is not None)
    print(f"saved {os.path.join(args.out, 'pose_map.png')} and pose_stats.json")


def annotated_ids(json_path):
    """All gt_image_ids in the IHM json, as a set of ints."""
    ids = set()
    for item in load_items(json_path):
        for x in item.get("gt_image_ids", []) or []:
            try:
                ids.add(int(x))
            except (TypeError, ValueError):
                pass

    return ids


def statistics(ids, centres, along, across, height, angle, det, levels, deck, outside, hull_half_width):
    inside = ~outside
    step = np.linalg.norm(np.diff(centres, axis=0), axis=1)

    def extent(x):
        return float(np.ptp(x))

    def robust_extent(x):
        return float(np.percentile(x, 99) - np.percentile(x, 1))

    def extents(mask, measure):
        return dict(along=round(measure(along[mask]), 2), across=round(measure(across[mask]), 2), height=round(measure(height[mask]), 2))

    everything = np.ones(len(ids), bool)

    return dict(
        n_poses=int(len(ids)), id_min=int(ids.min()), id_max=int(ids.max()), det_R_first=float(det),
        ship_axis_angle_to_first_horizontal_axis_deg=round(angle, 1),
        median_step_between_frames=round(float(np.median(step)), 4),
        path_length=round(float(step[step < MAX_STEP].sum()), 1),
        extent_all=extents(everything, extent),
        extent_inside_hull=extents(inside, extent),
        extent_inside_hull_1_99=extents(inside, robust_extent),
        deck_levels=[round(float(x), 2) for x in levels],
        deck_spacing=[round(float(x), 2) for x in np.diff(levels)],
        frames_per_deck=[int((deck[inside] == k).sum()) for k in range(len(levels))],
        outside_hull=dict(n=int(outside.sum()), share=round(float(outside.mean()), 4), hull_half_width=hull_half_width,
                          id_min=int(ids[outside].min()) if outside.any() else None,
                          id_max=int(ids[outside].max()) if outside.any() else None),
    )


def plot(along, across, height, levels, deck, outside, annotated, unit, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def thousands(n):
        return f"{int(n):,}".replace(",", " ")

    def extent(x):
        return float(np.ptp(x))

    inside = ~outside
    fig, (top, side) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                                    gridspec_kw=dict(height_ratios=[extent(across) + 2, extent(height) + 2], hspace=0.08))
    dots = dict(s=0.6, rasterized=True, linewidths=0)

    for k in range(len(levels)):
        mask = inside & (deck == k)
        colour = LEVEL_COLOURS[k % len(LEVEL_COLOURS)]
        top.scatter(along[mask], across[mask], color=colour, **dots)
        side.scatter(along[mask], height[mask], color=colour, **dots)
        side.scatter([], [], s=18, color=colour, label=f"Level {k + 1} ({levels[k]:+.1f} {unit}, {thousands(mask.sum())} frames)")

    if outside.any():
        top.scatter(along[outside], across[outside], color=OUTSIDE_COLOUR, **dots)
        side.scatter(along[outside], height[outside], color=OUTSIDE_COLOUR, **dots)
        side.scatter([], [], s=18, color=OUTSIDE_COLOUR, label=f"Outside the hull ({thousands(outside.sum())} frames)")

    if annotated is not None and annotated.any():
        for axis, y in ((top, across), (side, height)):
            axis.scatter(along[annotated], y[annotated], s=14, marker="x", color="black", linewidths=0.8, zorder=5)

        side.scatter([], [], s=18, marker="x", color="black", linewidths=0.8, label=f"Annotated scan image ({annotated.sum()})")

    top.set_aspect("equal")
    side.set_aspect("equal")
    top.set_ylabel(f"Across ({unit})")
    side.set_ylabel(f"Height ({unit})")
    side.set_xlabel(f"Along the ship ({unit})")
    top.set_title("Top-down view", fontsize=9, loc="left")
    side.set_title("Side view", fontsize=9, loc="left")

    for axis in (top, side):
        axis.grid(True, color="#E5E5E5", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)

        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)

    fig.legend(*side.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(os.path.join(out_dir, "pose_map.png"), dpi=300, bbox_inches="tight")


def print_summary(stats, hull_half_width, with_annotations):
    s = stats
    print(f"[check] det R = {s['det_R_first']:.6f} (a rotation has 1), median step between frames {s['median_step_between_frames']}")

    for label, key in (("all frames:       ", "extent_all"), ("inside the hull:  ", "extent_inside_hull")):
        e = s[key]
        print(f"[extent] {label} along {e['along']}, across {e['across']}, height {e['height']}")

    print(f"[decks] {len(s['deck_levels'])} levels at {s['deck_levels']}, spacing {s['deck_spacing']}, frames {s['frames_per_deck']}")
    o = s["outside_hull"]
    print(f"[hull] {o['n']} frames ({100 * o['share']:.1f}%) lie more than {hull_half_width} from the centre line"
          + (f", ids {o['id_min']}..{o['id_max']}" if o["n"] else ""))
    
    if with_annotations:
        m = s["annotated_images"]
        print(f"[ihm] {m['with_pose']}/{m['listed']} annotated images have a pose, per level {m['per_deck']}, outside the hull {m['outside_hull']}")
