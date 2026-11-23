"""Camera poses of the scan.

Each pose file holds a 3x4 matrix [R | t] that maps world coordinates to camera coordinates
(x_camera = R x_world + t). Distances are in meters.
"""
import glob
import os

import numpy as np

from .files import file_id


def load_pose(path):
    """Camera centre and viewing direction of one pose file, both in world coordinates."""
    return camera_from_matrix(np.loadtxt(path))


def camera_from_matrix(matrix):
    """Camera centre and viewing direction from a 3x4 world-to-camera matrix [R | t].

    The centre is C = -R^T t. The viewing direction is the camera's z axis, which is the third row of R.
    """
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    centre = rotation.T @ -translation
    forward = rotation[2]

    return centre, forward


def load_poses(pose_dir, ids):
    """Centres and viewing directions for the given frame ids, as two (N, 3) arrays.

    Rows are NaN for frames without a readable pose file.
    """
    pose_files = {
        file_id(path): path
        for path in glob.glob(os.path.join(pose_dir, "**", "*.txt"), recursive=True)
        if file_id(path) is not None
    }

    centres = np.full((len(ids), 3), np.nan)
    forwards = np.full((len(ids), 3), np.nan)
    
    for row, frame in enumerate(ids):
        if frame not in pose_files:
            continue
        try:
            centres[row], forwards[row] = load_pose(pose_files[frame])
        except Exception:
            pass                      # an unreadable file counts as a missing pose
        
    return centres, forwards


def viewing_angle(forward_a, forward_b):
    """Angle in degrees between two unit viewing directions."""
    return np.degrees(np.arccos(np.clip(forward_a @ forward_b, -1, 1)))
