"""The scan trajectory as a whole: loading all poses, the ship's axes, deck levels and frames outside the hull.

Used by the pose-map and pose-check commands. Distances are in meters.
"""
import glob
import os

import numpy as np

from .files import file_id
from .poses import camera_from_matrix

HISTOGRAM_BIN = 0.05              # meters
HISTOGRAM_SMOOTHING = 3           # bins (sigma 15 cm)


def add_deck_arguments(parser):
    """Arguments for the ship frame, the hull and the deck detection, with the same defaults everywhere."""
    parser.add_argument("--up-axis", type=int, default=2, choices=[0, 1, 2], help="vertical world axis (default 2)")
    parser.add_argument("--decks", type=int, default=0, help="force this number of deck levels (default: detect)")
    parser.add_argument("--deck-min-gap", type=float, default=1.2, help="smallest height difference between two decks")
    parser.add_argument("--deck-min-share", type=float, default=0.02, help="a level needs at least this share of the frames")
    parser.add_argument("--hull-half-width", type=float, default=6.0, help="frames further than this from the centre line count as outside the hull")


def load_pose_matrices(folder):
    """Frame ids and 3x4 pose matrices [R | t] of all readable pose files, sorted by frame id.

    Files that cannot be read, have the wrong shape or contain non-finite values are skipped.
    """
    files = sorted((f for f in glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True) if file_id(f) is not None), key=file_id)
    ids, matrices, unreadable = [], [], 0

    for path in files:
        try:
            matrix = np.loadtxt(path)
            if not np.isfinite(matrix[:3]).all():
                raise ValueError
            
            camera_from_matrix(matrix)                    # raises if the shape is wrong
        except Exception:
            unreadable += 1
            continue

        ids.append(file_id(path))
        matrices.append(matrix[:3, :4])

    if not ids:
        raise SystemExit(f"no readable pose files in {folder}")
    
    if unreadable:
        print(f"[poses] {unreadable} files could not be read and were skipped")

    return np.array(ids), np.stack(matrices)


def camera_centres(matrices):
    """Camera centres (N, 3) in world coordinates, assuming world-to-camera matrices."""
    return np.array([camera_from_matrix(m)[0] for m in matrices])


def median_step(points):
    """Median distance between consecutive points (frames)."""
    return float(np.median(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def principal_axes(points):
    """Points expressed along their own principal axes (longest first), and those axes as rows."""
    centred = points - points.mean(0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)

    return centred @ axes.T, axes


def ship_frame(centres, up_axis):
    """Coordinates along the ship, across it and in height, and the angle of the ship axis to the first world axis.

    The horizontal axes are rotated so that the first follows the longest direction of the trajectory.
    """
    horizontal = centres[:, [i for i in range(3) if i != up_axis]]
    middle = np.median(horizontal, axis=0)
    _, _, axes = np.linalg.svd(horizontal - middle, full_matrices=False)
    along, across = (horizontal - middle) @ axes[0], (horizontal - middle) @ axes[1]
    angle = float(np.degrees(np.arctan2(abs(axes[0][1]), abs(axes[0][0]))))

    return along, across, centres[:, up_axis], angle


def outside_hull(across, hull_half_width):
    """Frames further than hull_half_width from the centre line (the median across-ship position)."""
    return np.abs(across - np.median(across)) > hull_half_width


def deck_levels(height, n_decks, min_gap, min_share):
    """Deck heights, lowest first: peaks of the smoothed height histogram, or k-means with n_decks levels."""
    if n_decks:
        from sklearn.cluster import KMeans
        return np.sort(KMeans(n_decks, n_init=10, random_state=0).fit(height[:, None]).cluster_centers_[:, 0])
    
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    edges = np.arange(height.min() - 0.5, height.max() + 0.55, HISTOGRAM_BIN)
    counts, _ = np.histogram(height, edges)
    smooth = gaussian_filter1d(counts.astype(float), HISTOGRAM_SMOOTHING)
    peaks, _ = find_peaks(smooth, distance=int(min_gap / HISTOGRAM_BIN), prominence=smooth.max() * 0.08)
    levels = ((edges[:-1] + edges[1:]) / 2)[peaks]
    nearest = np.abs(height[:, None] - levels[None]).argmin(1)

    return levels[[k for k in range(len(levels)) if (nearest == k).mean() >= min_share]]     # drop tiny peaks


def decks_from_args(centres, args):
    """Ship frame, hull and deck levels with the settings of add_deck_arguments.

    Deck levels are found on the frames inside the hull. Returns a dict with along, across, height,
    angle, outside (bool per frame), levels and deck (level index per frame).
    """
    along, across, height, angle = ship_frame(centres, args.up_axis)
    outside = outside_hull(across, args.hull_half_width)
    levels = deck_levels(height[~outside], args.decks, args.deck_min_gap, args.deck_min_share)
    deck = np.abs(height[:, None] - levels[None]).argmin(1)
    
    return dict(along=along, across=across, height=height, angle=angle, outside=outside, levels=levels, deck=deck)
