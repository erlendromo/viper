"""Fine-tune MegaLoc (or SALAD) on the scan images, with training pairs taken from the camera poses.

  python train_pose.py --db-imgs data/latest/imgs --db-poses data/latest/poses \\
      --baseline-run run_megaloc --out run_ft            # add --dry-run first for a 2-minute smoke test

Needs anyloc_full.py, baselines_full.py and anyloc_eval.py in the same folder. The IHM photos are never used here.

What it does
  1. Pairs from poses: two scan frames are POSITIVE if their cameras are within --pos-radius and their viewing
     directions within --pos-angle; NEGATIVE if further apart than --neg-radius; anything in between is ignored.
  2. Hard negatives (Task 2.1): for every frame, the frames the baseline model finds most similar although they are
     negatives. Mined from the cached descriptors in --baseline-run and saved to hard_negatives.npy.
  3. Training (Task 2.2 / 3.3): backbone frozen, aggregation head trained with a multi-similarity loss. Each anchor
     is a close-up crop with lighting / perspective / occlusion augmentation; its positives and hard negatives are
     full frames, as the database images are at test time.
  4. Validation on held-out scan frames (close-up crops -> scan database). The best epoch is saved to best.pth and
     the most recent epoch to last.pth (use last.pth with --epochs N for a fixed training budget without selection).
Then score it on IHM with:
  python baselines_full.py --model megaloc --weights run_ft/best.pth ...usual arguments... --out run_megaloc_ft
"""
import argparse, json, os, time
import numpy as np
import anyloc_full as F
import baselines_full as B

# ------------------------------------------------------------------ pose logic (numpy only)
def load_db(img_dir, pose_dir):
    imgs = [p for p in F.list_images(img_dir) if F.num(p) is not None]
    import glob
    pose_of = {F.num(t): t for t in glob.glob(os.path.join(pose_dir, "**", "*.txt"), recursive=True) if F.num(t) is not None}
    keep, pos, fwd = [], [], []
    for p in imgs:
        t = pose_of.get(F.num(p))
        if t is None: continue
        try: c, f = F.A.load_pose(t)
        except Exception: continue
        keep.append(p); pos.append(c); fwd.append(f)
    return keep, np.array(pos, np.float32), np.array(fwd, np.float32)

def angle_deg(fa, fb):
    return np.degrees(np.arccos(np.clip(fa @ fb.T, -1, 1)))

def positives_of(pos, fwd, radius, max_angle, allowed, min_dist=0.0, min_angle=0.0, min_gap=0):
    """list of arrays: for each frame, the allowed frames that count as the same place (self excluded).
    With min_dist / min_angle / min_gap set, a positive must also differ from the anchor in at least one way:
    camera moved >= min_dist, view turned >= min_angle degrees, or >= min_gap frames apart in the capture
    (a different pass through the same spot). This drops near-duplicate neighbouring video frames."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pos); out = []; informative = min_dist > 0 or min_angle > 0 or min_gap > 0
    for i, nb in enumerate(tree.query_ball_point(pos, radius)):
        nb = np.array([j for j in nb if j != i and allowed[j]], int)
        if len(nb):
            ang = angle_deg(fwd[i:i + 1], fwd[nb])[0]; keep = ang <= max_angle
            if informative:
                d = np.linalg.norm(pos[nb] - pos[i], axis=1)
                diff = np.zeros(len(nb), bool)
                if min_dist > 0: diff |= d >= min_dist
                if min_angle > 0: diff |= ang >= min_angle
                if min_gap > 0: diff |= np.abs(nb - i) >= min_gap
                keep &= diff
            nb = nb[keep]
        out.append(nb)
    return out

def keyframes(pos, fwd, min_dist, min_angle):
    """keep a frame only when the camera has moved >= min_dist or turned >= min_angle since the last kept frame"""
    keep = np.zeros(len(pos), bool); last = None
    for i in range(len(pos)):
        if last is None or np.linalg.norm(pos[i] - pos[last]) >= min_dist or angle_deg(fwd[i:i + 1], fwd[last:last + 1])[0, 0] >= min_angle:
            keep[i] = True; last = i
    return keep

def make_split(n, pos, n_blocks, block_len, radius, seed=0):
    """validation frames = evenly spaced blocks of consecutive frames; training excludes everything near them"""
    from scipy.spatial import cKDTree
    starts = np.linspace(0, n - block_len, n_blocks).astype(int)
    val = np.zeros(n, bool)
    for s in starts: val[s:s + block_len] = True
    near = np.zeros(n, bool)
    for nb in cKDTree(pos).query_ball_point(pos[val], radius): near[nb] = True
    return val, ~(val | near)

def mine_hard_negatives(desc, pos, neg_radius, allowed, rows=None, keep=10, cand=200, chunk=512):
    """for each frame in `rows` (default all): the `keep` most similar allowed frames further away than neg_radius.
    Returns (n, keep) int32, -1 where there is none."""
    d = np.asarray(desc, np.float32); n = len(d); out = np.full((n, keep), -1, np.int32)
    rows = np.arange(n) if rows is None else np.asarray(rows); cand = min(cand, n - 1)
    t0 = time.time()
    for s in range(0, len(rows), chunk):
        ids = rows[s:s + chunk]; sim = d[ids] @ d.T
        top = np.argpartition(-sim, cand, axis=1)[:, :cand]
        for r, i in enumerate(ids):
            c = top[r][np.argsort(-sim[r, top[r]])]
            c = c[allowed[c] & (np.linalg.norm(pos[c] - pos[i], axis=1) > neg_radius)][:keep]
            out[i, :len(c)] = c
        if (s // chunk) % 8 == 0: print(f"  [mining] {min(s + chunk, len(rows))}/{len(rows)} frames, {time.time() - t0:.0f}s", flush=True)
    return out

def recall(sim, gts, ks=(1, 5, 10)):
    order = np.argsort(-sim, axis=1)[:, :max(ks)]
    return {f"R@{k}": float(np.mean([np.isin(order[i, :k], g).any() for i, g in enumerate(gts)])) for k in ks}

# ------------------------------------------------------------------ torch part
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def to_tensor(img):
    import torch
    x = torch.from_numpy(np.asarray(img, np.uint8).copy()).permute(2, 0, 1).float() / 255
    return (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)

def build_transforms(size, crop_min):
    from torchvision import transforms as T
    query = T.Compose([                                              # anchor: imitates an inspector's close-up photo
        T.RandomResizedCrop(size, scale=(crop_min, 1.0), ratio=(0.6, 1.8), antialias=True),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.05),         # lighting
        T.RandomPerspective(distortion_scale=0.3, p=0.3),                             # viewing angle
        T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.5))], p=0.2)])
    erase = T.RandomErasing(p=0.3, scale=(0.02, 0.2))                               # occlusion (applied on the tensor)
    dbase = T.Compose([T.Resize(size, antialias=True), T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)])
    return query, erase, dbase

class Groups:
    """one item = one anchor (augmented close-up) + its positives + its hard negatives (full frames).
    Top-level and picklable, because Python 3.14 starts DataLoader workers in fresh processes."""
    def __init__(self, imgs, anchors, positives, hard, size, n_pos, n_hard, crop_min, length):
        self.imgs, self.anchors, self.positives, self.hard = imgs, anchors, positives, hard
        self.size, self.n_pos, self.n_hard, self.crop_min, self.length = size, n_pos, n_hard, crop_min, length
        self.tf = None
    def __len__(self): return self.length
    def __getitem__(self, _):
        import torch
        from PIL import Image
        if self.tf is None: self.tf = build_transforms(self.size, self.crop_min)
        query_tf, erase, db_tf = self.tf
        a = int(self.anchors[torch.randint(len(self.anchors), (1,)).item()])
        p = self.positives[a]; p = p[torch.randint(len(p), (self.n_pos,)).numpy()]
        h = self.hard[a][self.hard[a] >= 0]
        h = h[torch.randperm(len(h))[:self.n_hard].numpy()] if len(h) else np.array([], int)
        idx = [a] + np.atleast_1d(p).tolist() + np.atleast_1d(h).tolist()
        xs = [erase(to_tensor(query_tf(Image.open(self.imgs[a]).convert("RGB"))))]
        xs += [to_tensor(db_tf(Image.open(self.imgs[j]).convert("RGB"))) for j in idx[1:]]
        return torch.stack(xs), torch.tensor(idx)

def collate_groups(batch):
    import torch
    return torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

def ms_loss(emb, pos_mask, neg_mask, alpha=1.0, beta=50.0, base=0.0, eps=0.1):
    """multi-similarity loss with explicit positive / negative masks (pairs in neither mask are ignored)"""
    import torch
    S = emb @ emb.T
    min_pos = S.masked_fill(~pos_mask, 1e4).min(1, keepdim=True).values
    max_neg = S.masked_fill(~neg_mask, -1e4).max(1, keepdim=True).values
    hp = pos_mask & (S - eps < max_neg)                      # informative positives
    hn = neg_mask & (S + eps > min_pos)                      # informative negatives
    valid = hp.any(1) & hn.any(1)
    if not valid.any(): return None
    zero = torch.zeros(len(S), 1, device=S.device, dtype=S.dtype)
    lp = torch.logsumexp(torch.cat([zero, (-alpha * (S - base)).masked_fill(~hp, float("-inf"))], 1), 1) / alpha
    ln = torch.logsumexp(torch.cat([zero, (beta * (S - base)).masked_fill(~hn, float("-inf"))], 1), 1) / beta
    return (lp + ln)[valid].mean()

def select_trainable(model, unfreeze_blocks, head_prefixes):
    """freeze everything, then unfreeze the aggregation head (+ optionally the last backbone blocks)"""
    import re
    names = [n for n, _ in model.named_parameters()]
    is_head = (lambda n: any(n.startswith(p) for p in head_prefixes)) if head_prefixes else (lambda n: "backbone" not in n)
    blocks = sorted({int(m.group(1)) for n in names for m in [re.search(r"blocks\.(\d+)\.", n)] if m and not is_head(n)})
    last = set(blocks[-unfreeze_blocks:]) if unfreeze_blocks > 0 else set()
    summary = {}
    for n, p in model.named_parameters():
        m = re.search(r"blocks\.(\d+)\.", n)
        p.requires_grad = bool(is_head(n) or (m and not is_head(n) and int(m.group(1)) in last))
        key = n.split(".")[0]; s = summary.setdefault(key, [0, 0]); s[0] += p.numel(); s[1] += p.numel() if p.requires_grad else 0
    print("[model] parameters by top-level module (total / trainable):")
    for k, (t, tr) in summary.items(): print(f"    {k:<20}{t / 1e6:8.2f}M {tr / 1e6:8.2f}M")
    n_tr = sum(v[1] for v in summary.values()); n_all = sum(v[0] for v in summary.values())
    if n_tr == 0: raise SystemExit("nothing is trainable: pass --head-prefixes with the module name(s) of the aggregation head listed above")
    if n_tr == n_all and unfreeze_blocks == 0:
        print("[model] WARNING: every parameter is trainable, so the backbone was not recognised by name. "
              "Pass --head-prefixes with the head's module name(s) from the list above.")
    if unfreeze_blocks > 0: print(f"[model] backbone blocks unfrozen: {sorted(last)}")
    return [p for p in model.parameters() if p.requires_grad]

def embed_paths(model, paths, size, device, bs, transform=None):
    """descriptors for image files; transform(i, PIL image) -> PIL image is optional (validation close-ups)"""
    import torch
    from PIL import Image
    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(paths), bs):
            xs = []
            for i, p in enumerate(paths[s:s + bs], start=s):
                im = Image.open(p).convert("RGB")
                if transform is not None: im = transform(i, im)
                xs.append(to_tensor(im.resize((size[1], size[0]), Image.BILINEAR)))
            d = model(torch.stack(xs).to(device))
            out.append(torch.nn.functional.normalize(d.float(), dim=1).cpu().numpy())
    return np.concatenate(out)

def val_closeup(seed, area=0.10):
    """deterministic hard close-up per validation query: small crop, brightness and contrast change, some blur"""
    from PIL import ImageEnhance, ImageFilter
    def tf(i, im):
        r = np.random.default_rng(seed + i); w, h = im.size
        s = np.sqrt(area); cw, ch = max(8, int(w * s)), max(8, int(h * s))
        x0, y0 = int(r.integers(0, w - cw + 1)), int(r.integers(0, h - ch + 1))
        im = im.crop((x0, y0, x0 + cw, y0 + ch))
        im = ImageEnhance.Brightness(im).enhance(float(r.uniform(0.5, 1.5)))
        im = ImageEnhance.Contrast(im).enhance(float(r.uniform(0.6, 1.4)))
        return im.filter(ImageFilter.GaussianBlur(float(r.uniform(0.0, 1.5))))
    return tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-imgs", required=True); ap.add_argument("--db-poses", required=True)
    ap.add_argument("--baseline-run", required=True, help="finished run of the same model (cached descriptors -> hard negatives)")
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="megaloc", choices=sorted(B.HUB))
    ap.add_argument("--size", type=int, nargs=2, default=[322, 322])
    ap.add_argument("--pos-radius", type=float, default=1.0); ap.add_argument("--pos-angle", type=float, default=45.0)
    ap.add_argument("--neg-radius", type=float, default=4.0)
    ap.add_argument("--pos-min-dist", type=float, default=0.0, help="informative positives: camera moved at least this far ...")
    ap.add_argument("--pos-min-angle", type=float, default=0.0, help="... or view turned at least this many degrees ...")
    ap.add_argument("--pos-min-gap", type=int, default=0, help="... or at least this many frames apart (another pass). 0 = off")
    ap.add_argument("--keyframe-dist", type=float, default=0.0, help="thin training frames: keep one when the camera moved this far ...")
    ap.add_argument("--keyframe-angle", type=float, default=10.0, help="... or turned this many degrees (only with --keyframe-dist > 0)")
    ap.add_argument("--anchors", type=int, default=16, help="anchors per batch")
    ap.add_argument("--n-pos", type=int, default=3); ap.add_argument("--n-hard", type=int, default=2)
    ap.add_argument("--crop-min", type=float, default=0.15, help="smallest close-up crop, as a share of the image area")
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--iters", type=int, default=300, help="batches per epoch")
    ap.add_argument("--patience", type=int, default=3); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--unfreeze-blocks", type=int, default=0, help="also train the last N backbone blocks (0 = head only)")
    ap.add_argument("--head-prefixes", nargs="*", default=None, help="override which modules count as the head")
    ap.add_argument("--val-blocks", type=int, default=20); ap.add_argument("--val-block-len", type=int, default=40)
    ap.add_argument("--val-queries", type=int, default=300); ap.add_argument("--val-db-stride", type=int, default=4)
    ap.add_argument("--val-area", type=float, default=0.10, help="size of the validation close-up, as a share of the image area "
                    "(smaller = harder; aim for an epoch-0 validation R@1 of roughly 0.4-0.8)")
    ap.add_argument("--eval-batch", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", default="mean", choices=["mean", "R@1"],
                    help="which validation score picks the best epoch: mean of R@1/R@5/R@10 (default) or R@1 alone")
    ap.add_argument("--dry-run", action="store_true", help="tiny run to check that everything works")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.dry_run: a.epochs, a.iters, a.val_queries, a.val_db_stride, a.anchors = 1, 3, 20, 8, 4

    # ---- data, split, pairs ----
    imgs, pos, fwd = load_db(a.db_imgs, a.db_poses)
    n = len(imgs); print(f"[data] {n} frames with poses")
    val, train = make_split(n, pos, a.val_blocks, a.val_block_len, a.pos_radius, a.seed)
    if a.keyframe_dist > 0:
        kf = keyframes(pos, fwd, a.keyframe_dist, a.keyframe_angle); before = int(train.sum()); train = train & kf
        print(f"[data] keyframe thinning: {before} -> {int(train.sum())} training frames")
    positives = positives_of(pos, fwd, a.pos_radius, a.pos_angle, train, a.pos_min_dist, a.pos_min_angle, a.pos_min_gap)
    anchors = np.array([i for i in np.where(train)[0] if len(positives[i])])
    print(f"[data] train frames {int(train.sum())} ({len(anchors)} usable as anchors), validation frames {int(val.sum())}, "
          f"median positives per anchor {int(np.median([len(positives[i]) for i in anchors]))}")

    names = json.load(open(os.path.join(a.baseline_run, "db_names.json")))
    row = {nm: r for r, nm in enumerate(names)}
    if any(os.path.basename(p) not in row for p in imgs): raise SystemExit("--baseline-run does not cover all frames")
    hfile = os.path.join(a.out, f"hard_negatives_kf{a.keyframe_dist}_{a.keyframe_angle}_neg{a.neg_radius}.npy")
    if os.path.exists(hfile) and len(np.load(hfile)) == n:
        hard = np.load(hfile); print("[mining] using cached hard negatives")
    else:
        desc = np.load(os.path.join(a.baseline_run, "db_desc.npy"), mmap_mode="r")
        desc = np.asarray(desc, np.float32)[[row[os.path.basename(p)] for p in imgs]]
        if a.dry_run:
            anchors = anchors[:64]; hard = mine_hard_negatives(desc, pos, a.neg_radius, train, rows=anchors)
        else:
            hard = mine_hard_negatives(desc, pos, a.neg_radius, train); np.save(hfile, hard)
        del desc
    print(f"[mining] anchors with at least one hard negative: {int((hard[anchors] >= 0).any(1).sum())}/{len(anchors)}")

    # validation set: close-up crops of held-out frames -> every Nth frame of the scan
    rng = np.random.default_rng(a.seed)
    vq = np.sort(rng.choice(np.where(val)[0], min(a.val_queries, int(val.sum())), replace=False))
    vdb = np.arange(0, n, a.val_db_stride)
    vgt = []
    for q in vq:
        ok = (np.linalg.norm(pos[vdb] - pos[q], axis=1) <= a.pos_radius) & (angle_deg(fwd[q:q + 1], fwd[vdb])[0] <= a.pos_angle) & (vdb != q)
        vgt.append(np.where(ok)[0])
    keepq = [i for i, g in enumerate(vgt) if len(g)]; vq, vgt = vq[keepq], [vgt[i] for i in keepq]
    print(f"[val] {len(vq)} close-up queries against {len(vdb)} scan frames")
    if not len(vq): raise SystemExit("no validation query has a positive in the validation database: lower --val-db-stride")

    # ---- model ----
    import torch
    torch.manual_seed(a.seed)
    device = F.pick_device()
    repo, entry = B.HUB[a.model]
    model = torch.hub.load(repo, entry).to(device).eval()      # eval mode throughout: dropout off, deterministic
    params = select_trainable(model, a.unfreeze_blocks, a.head_prefixes)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    pos_t, fwd_t = torch.tensor(pos, device=device), torch.tensor(fwd, device=device)
    cos_max = float(np.cos(np.radians(a.pos_angle)))

    ds = Groups(imgs, anchors, positives, hard, tuple(a.size), a.n_pos, a.n_hard, a.crop_min, a.iters * a.anchors)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.anchors, num_workers=a.workers, collate_fn=collate_groups, drop_last=True,
                                         persistent_workers=a.workers > 0)

    def validate():
        dbd = embed_paths(model, [imgs[i] for i in vdb], a.size, device, a.eval_batch)
        qd = embed_paths(model, [imgs[i] for i in vq], a.size, device, a.eval_batch, val_closeup(a.seed, a.val_area))
        return recall(qd @ dbd.T, vgt)

    log = []; t0 = time.time()
    score = (lambda v: float(np.mean([v["R@1"], v["R@5"], v["R@10"]]))) if a.select == "mean" else (lambda v: v["R@1"])
    v = validate(); best = score(v); bad = 0
    print(f"[epoch 0] before training: val {v}", flush=True); log.append(dict(epoch=0, val=v))
    if v["R@1"] > 0.85: print("[val] NOTE: the validation task is nearly solved before training, so it cannot show an improvement. "
                              "Stop and rerun with a smaller --val-area (e.g. half the current value).", flush=True)
    torch.save({"state_dict": model.state_dict(), "epoch": 0, "val": v}, os.path.join(a.out, "best.pth"))
    step, warm = 0, max(1, a.iters // 2)
    for ep in range(1, a.epochs + 1):
        losses, skipped = [], 0
        for x, idx in loader:
            for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (step + 1) / warm)      # linear warm-up
            x, idx = x.to(device, non_blocking=True), idx.to(device)
            emb = torch.nn.functional.normalize(model(x).float(), dim=1)
            dist = torch.cdist(pos_t[idx], pos_t[idx]); cosang = fwd_t[idx] @ fwd_t[idx].T
            eye = torch.eye(len(idx), dtype=torch.bool, device=device)
            pm = (dist <= a.pos_radius) & (cosang >= cos_max) & ~eye
            nm = dist > a.neg_radius
            loss = ms_loss(emb, pm, nm)
            if loss is None: skipped += 1; continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            losses.append(loss.item()); step += 1
            if step % 50 == 0: print(f"    step {step}: loss {np.mean(losses[-50:]):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        v = validate()
        rec = dict(epoch=ep, loss=float(np.mean(losses)) if losses else None, skipped_batches=skipped, val=v); log.append(rec)
        print(f"[epoch {ep}] loss {rec['loss']}  val {v}  score {score(v):.3f}" + ("  <- best" if score(v) > best else ""), flush=True)
        json.dump(dict(args=vars(a), log=log), open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
        torch.save({"state_dict": model.state_dict(), "epoch": ep, "val": v}, os.path.join(a.out, "last.pth"))   # always: the most recent epoch
        if score(v) > best:
            best, bad = score(v), 0
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val": v}, os.path.join(a.out, "best.pth"))
        else:
            bad += 1
            if bad >= a.patience: print(f"no improvement for {a.patience} epochs: stopping"); break
    json.dump(dict(args=vars(a), log=log), open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
    print(f"done. best validation score ({a.select}) {best:.3f}; weights in {os.path.join(a.out, 'best.pth')}")

if __name__ == "__main__":
    main()
