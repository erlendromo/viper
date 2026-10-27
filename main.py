"""Zero-shot AnyLoc (DINOv2 value-facet + VLAD) on the Hestmanden scan dataset.

  python anyloc_eval.py DATASET_DIR --out run1 [--model dinov2_vitg14 --layer 31 --clusters 32]

Pairs  <stem>.png  with  <stem>.txt  (world->camera pose). Queries = every Nth image, database = the rest.
Ground truth: a database image is a positive if its camera is within --pos-radius (pose units) of the query
AND its viewing direction is within --pos-angle degrees. Queries with no positive are skipped.
Outputs in --out: descriptors.npy, results.json (Recall@K), failures.json (hard negatives for Task 2.1).
"""
import argparse, glob, json, os
import numpy as np
import torch, cv2
from sklearn.cluster import MiniBatchKMeans

# ---------------- poses / ground truth (numpy only) ----------------
def load_pose(fn):
    T = np.loadtxt(fn)
    R_wc, t = T[:3, :3], T[:3, 3]
    return R_wc.T @ -t, R_wc[2]            # camera centre, forward direction (world frame)

def find_pairs(root):
    pairs = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True)):
        t = os.path.splitext(p)[0] + ".txt"
        if os.path.exists(t):
            pairs.append((p, t))
    return pairs

def split_ids(n, every):
    q = np.arange(0, n, every)
    d = np.setdiff1d(np.arange(n), q)
    return q, d

def ground_truth(pos, fwd, q_ids, db_ids, radius, angle_deg):
    dist = np.linalg.norm(pos[q_ids][:, None] - pos[db_ids][None], axis=2)
    cosang = np.clip(fwd[q_ids] @ fwd[db_ids].T, -1, 1)
    ok = (dist <= radius) & (np.degrees(np.arccos(cosang)) <= angle_deg)
    return [np.where(r)[0] for r in ok]    # indices into db_ids

# ---------------- VLAD (numpy / sklearn) ----------------
def fit_vocab(local_feats, k, seed=0):
    return MiniBatchKMeans(k, random_state=seed, n_init=3, batch_size=8192).fit(local_feats).cluster_centers_.astype(np.float32)

def vlad(feats, centers):
    """feats (N,D) -> intra-normalised, L2-normalised VLAD (K*D,)"""
    f = feats.astype(np.float32)
    d2 = (f**2).sum(1)[:, None] - 2 * f @ centers.T + (centers**2).sum(1)[None]
    a = d2.argmin(1)
    v = np.zeros_like(centers)
    for k in range(len(centers)):
        m = a == k
        if m.any():
            v[k] = (f[m] - centers[k]).sum(0)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    v = v.ravel()
    return v / (np.linalg.norm(v) + 1e-12)

def recall_at_k(sim, gts, ks=(1, 5, 10)):
    hits = {k: 0 for k in ks}; fails = []
    for i, gt in enumerate(gts):
        order = np.argsort(-sim[i])
        for k in ks:
            if np.isin(order[:k], gt).any(): hits[k] += 1
        if order[0] not in gt:
            fails.append(dict(query=i, top1=int(order[0]), top5=order[:5].tolist(), positives=gt[:10].tolist()))
    return {f"R@{k}": hits[k] / len(gts) for k in ks}, fails

# ---------------- DINOv2 value-facet extractor (torch) ----------------
class DinoValueFeats:
    def __init__(self, model, layer, size, device):
        self.torch = torch; self.dev = device; self.size = size; self.layer = layer
        self.m = torch.hub.load("facebookresearch/dinov2", model).to(device).eval()
        self._buf = {}
        self.m.blocks[layer].attn.qkv.register_forward_hook(lambda mod, i, o: self._buf.__setitem__("qkv", o))
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @property
    def dim(self): return self.m.embed_dim

    def __call__(self, img_paths):
        xs = []
        for p in img_paths:
            im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
            xs.append(torch.from_numpy(im).permute(2, 0, 1).float() / 255)
        x = (torch.stack(xs).to(self.dev) - self.mean) / self.std
        with torch.no_grad():
            self.m.forward_features(x)
        C = self.dim
        v = self._buf["qkv"][..., 2 * C:]          # value facet
        return v[:, 1:].float().cpu().numpy()      # drop CLS -> (B, patches, C)

def extract(ex, paths, bs=8):
    for i in range(0, len(paths), bs):
        for f in ex(paths[i:i + bs]): yield f

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--out", default="run")
    ap.add_argument("--model", default="dinov2_vitg14"); ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--clusters", type=int, default=32)
    ap.add_argument("--size", type=int, nargs=2, default=[322, 560], help="H W, multiples of 14")
    ap.add_argument("--query-every", type=int, default=5)
    ap.add_argument("--pos-radius", type=float, default=0.5); ap.add_argument("--pos-angle", type=float, default=45)
    ap.add_argument("--max-images", type=int, default=1000, help="Task 2.1 cap; 0 = all")
    ap.add_argument("--vocab-images", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pairs = find_pairs(a.root)
    if a.max_images: pairs = pairs[::max(1, len(pairs) // a.max_images)][:a.max_images]
    imgs = [p for p, _ in pairs]
    P = [load_pose(t) for _, t in pairs]; pos = np.array([p[0] for p in P]); fwd = np.array([p[1] for p in P])
    q_ids, db_ids = split_ids(len(imgs), a.query_every)
    gts = ground_truth(pos, fwd, q_ids, db_ids, a.pos_radius, a.pos_angle)
    keep = [i for i, g in enumerate(gts) if len(g)]
    print(f"{len(imgs)} images | {len(q_ids)} queries ({len(keep)} with positives) | {len(db_ids)} database")
    q_ids = q_ids[keep]; gts = [gts[i] for i in keep]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(dev)
    ex = DinoValueFeats(a.model, a.layer, tuple(a.size), dev)

    rng = np.random.default_rng(0)
    vocab_ids = rng.choice(db_ids, min(a.vocab_images, len(db_ids)), replace=False)
    local = np.concatenate(list(extract(ex, [imgs[i] for i in vocab_ids])))
    centers = fit_vocab(local[rng.choice(len(local), min(len(local), 300000), replace=False)], a.clusters)
    desc = np.stack([vlad(f, centers) for f in extract(ex, imgs)])
    np.save(os.path.join(a.out, "descriptors.npy"), desc)

    sim = desc[q_ids] @ desc[db_ids].T
    res, fails = recall_at_k(sim, gts)
    res.update(n_queries=len(q_ids), n_database=len(db_ids), model=a.model, layer=a.layer, clusters=a.clusters)
    for f in fails:
        f["query_img"] = imgs[q_ids[f["query"]]]; f["top1_img"] = imgs[db_ids[f["top1"]]]
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    json.dump(fails, open(os.path.join(a.out, "failures.json"), "w"), indent=1)
    print(res, f"| {len(fails)} failures -> failures.json")

if __name__ == "__main__":
    main()
