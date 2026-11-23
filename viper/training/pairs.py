"""Which scan frames show the same place, derived from the camera poses (numpy only).

Two frames are a POSITIVE pair if their cameras are within the positive radius and their viewing directions
within the maximum angle. They are a NEGATIVE pair if the cameras are further apart than the negative radius.
Pairs in between are ignored, since they may or may not show the same content.
"""
import time

import numpy as np

from viper.utils.files import file_id, list_images
from viper.utils.poses import load_poses


def load_posed_frames(img_dir, pose_dir):
    """Scan images that have a pose, with camera centres and viewing directions as float32 (N, 3) arrays."""
    images = [p for p in list_images(img_dir) if file_id(p) is not None]
    centres, forwards = load_poses(pose_dir, [file_id(p) for p in images])
    has_pose = ~np.isnan(centres).any(1)

    return ([p for p, ok in zip(images, has_pose) if ok],
            centres[has_pose].astype(np.float32), forwards[has_pose].astype(np.float32))


def angles_deg(forwards_a, forwards_b):
    """Angles in degrees between every row of forwards_a and every row of forwards_b."""
    return np.degrees(np.arccos(np.clip(forwards_a @ forwards_b.T, -1, 1)))


def positives_of(pos, fwd, radius, max_angle, allowed, min_dist=0.0, min_angle=0.0, min_gap=0):
    """For every frame, the allowed frames that show the same place (the frame itself excluded).

    With min_dist, min_angle or min_gap set ("informative pairs"), a positive must also differ from the anchor
    in at least one way: the camera moved at least min_dist, the view turned at least min_angle degrees, or the
    two frames are at least min_gap frames apart in the capture (another pass through the same spot).
    This drops near-duplicate neighbouring video frames.
    """
    from scipy.spatial import cKDTree

    informative = min_dist > 0 or min_angle > 0 or min_gap > 0
    positives = []
    for i, near in enumerate(cKDTree(pos).query_ball_point(pos, radius)):
        near = np.array([j for j in near if j != i and allowed[j]], int)
        if len(near):
            angle = angles_deg(fwd[i:i + 1], fwd[near])[0]
            keep = angle <= max_angle

            if informative:
                different = np.zeros(len(near), bool)
                if min_dist > 0:
                    different |= np.linalg.norm(pos[near] - pos[i], axis=1) >= min_dist

                if min_angle > 0:
                    different |= angle >= min_angle

                if min_gap > 0:
                    different |= np.abs(near - i) >= min_gap

                keep &= different

            near = near[keep]

        positives.append(near)

    return positives


def keyframes(pos, fwd, min_dist, min_angle):
    """Keep a frame only when the camera has moved min_dist or turned min_angle since the last kept frame."""
    keep = np.zeros(len(pos), bool)
    last = None

    for i in range(len(pos)):
        if (last is None or np.linalg.norm(pos[i] - pos[last]) >= min_dist
                or angles_deg(fwd[i:i + 1], fwd[last:last + 1])[0, 0] >= min_angle):
            keep[i] = True
            last = i
    
    return keep


def make_split(pos, n_blocks, block_len, radius):
    """Validation frames are evenly spaced blocks of consecutive frames.

    Training frames are all others, except those within radius of a validation frame, so no validation
    place is seen during training. Returns two boolean masks: (validation, training).
    """
    from scipy.spatial import cKDTree

    n = len(pos)
    validation = np.zeros(n, bool)
    for start in np.linspace(0, n - block_len, n_blocks).astype(int):
        validation[start:start + block_len] = True
    
    near = np.zeros(n, bool)
    for close in cKDTree(pos).query_ball_point(pos[validation], radius):
        near[close] = True
    
    return validation, ~(validation | near)


def mine_hard_negatives(desc, pos, neg_radius, allowed, rows=None, keep=10, candidates=200, chunk=512):
    """For each frame in rows (default: all), the `keep` allowed frames that the baseline model finds most
    similar although they are negatives (cameras further apart than neg_radius).

    These are the look-alike places the baseline confuses. Returns an (n, keep) int32 array, -1 where none.
    """
    desc = np.asarray(desc, np.float32)
    n = len(desc)
    hard = np.full((n, keep), -1, np.int32)
    rows = np.arange(n) if rows is None else np.asarray(rows)
    candidates = min(candidates, n - 1)
    start_time = time.time()

    for start in range(0, len(rows), chunk):
        ids = rows[start:start + chunk]
        sim = desc[ids] @ desc.T
        top = np.argpartition(-sim, candidates, axis=1)[:, :candidates]

        for r, i in enumerate(ids):
            ranked = top[r][np.argsort(-sim[r, top[r]])]
            ranked = ranked[allowed[ranked] & (np.linalg.norm(pos[ranked] - pos[i], axis=1) > neg_radius)][:keep]
            hard[i, :len(ranked)] = ranked
        
        if (start // chunk) % 8 == 0:
            print(f"  [mining] {min(start + chunk, len(rows))}/{len(rows)} frames, {time.time() - start_time:.0f}s", flush=True)
    
    return hard
