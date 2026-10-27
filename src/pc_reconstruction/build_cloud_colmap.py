"""Step 2: build a real 3D point cloud from images + known poses with COLMAP (point_triangulator).
Requires: colmap on PATH (apt/conda install colmap), pip install numpy scipy open3d
Usage: python build_cloud_colmap.py DATASET_DIR WORK_DIR [--matcher exhaustive|sequential]
Images (.png) and poses (.txt) must share the same stem, e.g. 00012.png / 00012.txt.
Output: WORK_DIR/sparse_tri/ (COLMAP model) and WORK_DIR/cloud.ply (colored points).
"""
import sys, os, glob, shutil, subprocess, argparse
import numpy as np
from scipy.spatial.transform import Rotation as Rot

W, H = 1280, 720
FX, FY, CX, CY = 636.55603027, 635.93658447, 630.45141602, 372.948698

def run(*a):
    print(">", " ".join(map(str, a))); subprocess.run(list(map(str, a)), check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("work")
    ap.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    a = ap.parse_args()

    img_dir = os.path.join(a.work, "images"); os.makedirs(img_dir, exist_ok=True)
    model = os.path.join(a.work, "sparse_init"); os.makedirs(model, exist_ok=True)
    db = os.path.join(a.work, "database.db")

    stems = []
    for p in sorted(glob.glob(os.path.join(a.root, "**", "*.png"), recursive=True)):
        s = os.path.splitext(p)[0]
        if os.path.exists(s + ".txt"):
            dst = os.path.join(img_dir, os.path.basename(p))
            if not os.path.exists(dst): shutil.copy(p, dst)
            stems.append((os.path.basename(p), s + ".txt"))
    print(len(stems), "image/pose pairs")

    # 1. features + matching
    run("colmap", "feature_extractor", "--database_path", db, "--image_path", img_dir,
        "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", 1,
        "--ImageReader.camera_params", f"{FX},{FY},{CX},{CY}")
    run("colmap", f"{a.matcher}_matcher", "--database_path", db)

    # 2. write a COLMAP text model with the KNOWN poses (image ids must match database order)
    with open(os.path.join(model, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {W} {H} {FX} {FY} {CX} {CY}\n")
    open(os.path.join(model, "points3D.txt"), "w").close()
    names_sorted = sorted(n for n, _ in stems)          # feature_extractor assigns ids in sorted order
    pose_of = dict(stems)
    with open(os.path.join(model, "images.txt"), "w") as f:
        for i, n in enumerate(names_sorted, 1):
            T = np.loadtxt(pose_of[n]); R, t = T[:3, :3], T[:3, 3]      # world -> camera
            qx, qy, qz, qw = Rot.from_matrix(R).as_quat()
            f.write(f"{i} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {n}\n\n")

    # 3. triangulate points with fixed poses
    out = os.path.join(a.work, "sparse_tri"); os.makedirs(out, exist_ok=True)
    run("colmap", "point_triangulator", "--database_path", db, "--image_path", img_dir,
        "--input_path", model, "--output_path", out)
    run("colmap", "model_converter", "--input_path", out, "--output_path",
        os.path.join(a.work, "cloud.ply"), "--output_type", "PLY")
    print("done ->", os.path.join(a.work, "cloud.ply"))

if __name__ == "__main__":
    main()
