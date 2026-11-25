"""Sanity check of the pose files: convention, scale and deck levels.

    python main.py pose-check --db-poses data/latest/poses --out runs/pose_check

Checks
    rotation      det R of every pose should be +1 (a proper rotation, no scaling or mirroring)
    convention    the camera path is smooth only under the right reading of [R | t]: the median step
                  between frames is computed both as world-to-camera (centre -R^T t, what the code assumes)
                  and as camera-to-world (centre t), and the smaller one wins
    scale         extent of the path along its principal axes, also without the outer 1% at each end,
                  to compare with the size of the ship
    decks         histogram of the camera height, with the deck levels that pose-map detects

Files written to --out
    pose_check.png     top-down view along the principal axes (coloured by frame order) and the height histogram
    pose_check.json    the numbers above
"""
import json
import os

import numpy as np

from .trajectory import add_deck_arguments, camera_centres, decks_from_args, load_pose_matrices, median_step, principal_axes

HELP = "sanity check of the poses: convention, scale and deck levels (pose_check.png)"

HISTOGRAM_BINS = 200


def add_arguments(parser):
    parser.add_argument("--db-poses", required=True, help="folder with the pose files")
    parser.add_argument("--out", required=True, help="output folder")
    add_deck_arguments(parser)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    ids, matrices = load_pose_matrices(args.db_poses)
    centres = camera_centres(matrices)
    print(f"[poses] {len(ids)} poses, ids {ids.min()}..{ids.max()}")

    stats = dict(n_poses=int(len(ids)))
    stats.update(rotation_check(matrices))
    stats.update(convention_check(matrices, centres))
    projected, axes = principal_axes(centres)
    stats.update(scale_check(projected, axes))
    decks = decks_from_args(centres, args)
    stats["deck_levels"] = [round(float(x), 2) for x in decks["levels"]]
    stats["deck_spacing"] = [round(float(x), 2) for x in np.diff(decks["levels"])]

    with open(os.path.join(args.out, "pose_check.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print_summary(stats)
    plot(projected, centres[:, args.up_axis], decks["levels"], args.out)
    print(f"saved {os.path.join(args.out, 'pose_check.png')} and pose_check.json")


def rotation_check(matrices):
    det = np.linalg.det(matrices[:, :, :3])

    return dict(det_R_min=round(float(det.min()), 6), det_R_max=round(float(det.max()), 6))


def convention_check(matrices, centres):
    """Median step between frames under both readings of the pose matrix."""
    world_to_camera = median_step(centres)              # centre = -R^T t
    camera_to_world = median_step(matrices[:, :, 3])    # centre = t

    return dict(median_step_world_to_camera=round(world_to_camera, 4),
                median_step_camera_to_world=round(camera_to_world, 4),
                convention="world-to-camera" if world_to_camera < camera_to_world else "camera-to-world")


def scale_check(projected, axes):
    def rounded(values):
        return [round(float(v), 2) for v in values]

    return dict(extent_principal_axes=rounded(np.ptp(projected, axis=0)),
                extent_principal_axes_1_99=rounded(np.percentile(projected, 99, axis=0) - np.percentile(projected, 1, axis=0)),
                longest_axis_direction=[round(float(v), 4) for v in axes[0]])


def print_summary(s):
    print(f"[rotation] det R between {s['det_R_min']} and {s['det_R_max']} (a rotation has 1)")
    print(f"[convention] median step between frames: {s['median_step_world_to_camera']} if world-to-camera, "
          f"{s['median_step_camera_to_world']} if camera-to-world -> {s['convention']}")
    
    if s["convention"] != "world-to-camera":
        print("[convention] WARNING: the code assumes world-to-camera poses. Check the pose format before using the results.")

    print(f"[scale] extent along the principal axes: {s['extent_principal_axes']}, "
          f"without the outer 1%: {s['extent_principal_axes_1_99']}, longest axis {s['longest_axis_direction']}")
    print(f"[decks] {len(s['deck_levels'])} levels at {s['deck_levels']}, spacing {s['deck_spacing']}")


def plot(projected, height, levels, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, hist) = plt.subplots(1, 2, figsize=(14, 5))
    points = top.scatter(projected[:, 0], projected[:, 1], s=1, c=np.arange(len(projected)))
    fig.colorbar(points, ax=top, label="frame order", shrink=0.5)
    top.set_aspect("equal")
    top.set_title("Top-down, along the principal axes")
    top.set_xlabel("first principal axis (m)")
    top.set_ylabel("second principal axis (m)")

    hist.hist(height, bins=HISTOGRAM_BINS, color="#014693")
    for k, level in enumerate(levels):
        hist.axvline(level, color="#ED8013", linestyle="--", linewidth=1, label="deck levels (pose-map)" if k == 0 else None)

    if len(levels):
        hist.legend()
        
    hist.set_title("Camera height")
    hist.set_xlabel("height (m)")
    hist.set_ylabel("frames")
    fig.savefig(os.path.join(out_dir, "pose_check.png"), dpi=150, bbox_inches="tight")
