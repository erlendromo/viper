"""Step 1: combine all pose .txt files into a camera cloud (positions + view frustums).
Usage: python poses_to_cloud.py DATASET_DIR [OUT_DIR]
Writes cameras.ply (camera centres), frustums.ply (line set) and cameras.html (interactive).
"""
import sys, glob, os
import numpy as np

W, H = 1280, 720
K = np.array([
    [636.55603027, 0, 630.45141602],
    [0, 635.93658447, 372.948698],
    [0, 0, 1.0]
])

def load_pose(fn):
    T = np.loadtxt(fn)                      # 3x4 (or 4x4) world->camera
    R_wc = T[:3, :3]
    t = T[:3, 3]
    R_cw = R_wc.T
    return R_cw @ -t, R_cw                  # camera centre, camera->world rotation

def frustum(c, R_cw, depth=0.3):
    corners_px = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    rays = (np.linalg.inv(K) @ np.c_[corners_px, np.ones(4)].T).T
    pts = [c] + [c + R_cw @ (r * depth) for r in rays]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    return np.array(pts), np.array(lines)

def write_ply(fn, pts, rgb):
    with open(fn, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in pts: f.write(f"{p[0]} {p[1]} {p[2]} {rgb[0]} {rgb[1]} {rgb[2]}\n")

def main(root, out="."):
    os.makedirs(out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(root, "**", "*.txt"), recursive=True))
    files = [f for f in files if os.path.basename(f) not in ("cameras.txt", "images.txt", "points3D.txt")]
    centres, all_pts, all_lines, names = [], [], [], []
    for f in files:
        try:
            c, R = load_pose(f)
        except Exception as e:
            print("skip", f, e); continue
        centres.append(c); names.append(os.path.basename(f))
        p, l = frustum(c, R)
        all_lines.append(l + 5 * len(all_pts))
        all_pts.append(p)
    centres = np.array(centres)
    print(f"{len(centres)} poses; extent min {centres.min(0).round(2)} max {centres.max(0).round(2)}")

    write_ply(os.path.join(out, "cameras.ply"), centres, [255, 0, 0])
    np.save(os.path.join(out, "frustum_pts.npy"), np.vstack(all_pts))
    np.save(os.path.join(out, "frustum_lines.npy"), np.vstack(all_lines))

    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter3d(x=centres[:, 0], y=centres[:, 1], z=centres[:, 2], mode="markers+lines",
                                 marker=dict(size=2, color=np.arange(len(centres)), colorscale="Viridis"),
                                 line=dict(width=1), text=names))
    P, L = np.vstack(all_pts), np.vstack(all_lines)
    xs, ys, zs = [], [], []
    for i, j in L:
        xs += [P[i, 0], P[j, 0], None]; ys += [P[i, 1], P[j, 1], None]; zs += [P[i, 2], P[j, 2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(width=1, color="royalblue"), name="frustums"))
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.write_html(os.path.join(out, "cameras.html"))
    np.save(os.path.join(out, "camera_centres.npy"), centres)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out_cameras")
