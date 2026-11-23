"""AnyLoc on the full Hestmanden database with the IHM queries.

  python anyloc_full.py --db-imgs DATA/imgs --db-poses DATA/poses --queries IHM --json IHM/queries_v2.json --out run_full

Needs anyloc_eval.py in the same folder (feature extractor + VLAD).
- Database images and pose files are matched by the number at the end of the file name (00012.png <-> 00012.txt).
- gt_image_ids in the JSON are matched to database images by that same number.
- Poses are optional; with them the localisation error is also reported.
- Database descriptors are written to disk as they are computed, so an interrupted run resumes where it stopped.
Outputs in --out: results.json, topk.json, failures.json, failures_per_item.json, montage/*.jpg
"""
import argparse, glob, json, os, re
import numpy as np
import anyloc_eval as A

IMG_EXT = ("png", "jpg", "jpeg")
KS = (1, 5, 10)

def num(path):
    m = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return int(m[-1]) if m else None

def list_images(root):
    return sorted(p for e in IMG_EXT for p in glob.glob(os.path.join(root, "**", f"*.{e}"), recursive=True))

def metrics(sim, gts):
    """sim (Q,N), gts list of db-index arrays -> per-query dict of arrays + top-10 order"""
    order = np.argsort(-sim, axis=1)[:, :max(KS)]
    out = {f"R@{k}": [] for k in KS}; out["mAP@10"] = []; out["MRR@10"] = []
    for i, gt in enumerate(gts):
        rel = np.isin(order[i], gt)
        for k in KS: out[f"R@{k}"].append(float(rel[:k].any()))
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        out["mAP@10"].append(float((prec * rel)[:10].sum() / min(10, len(gt))))
        out["MRR@10"].append(float(1 / (np.argmax(rel[:10]) + 1)) if rel[:10].any() else 0.0)
    return {k: np.array(v) for k, v in out.items()}, order

def summarise(per_query, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(next(iter(per_query.values()))); res = {}
    idx = rng.integers(0, n, (n_boot, n))
    for k, v in per_query.items():
        boots = v[idx].mean(1)
        res[k] = round(float(v.mean()), 4)
        res[k + "_ci95"] = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
    res["n"] = n
    return res

def loc_error(order, gts, db_pos):
    """distance from the top-1 image's camera to the nearest ground-truth image's camera"""
    d = []
    for i, gt in enumerate(gts):
        p = db_pos[order[i, 0]]; g = db_pos[gt]; g = g[~np.isnan(g).any(1)]
        if len(g) and not np.isnan(p).any(): d.append(np.linalg.norm(g - p, axis=1).min())
    d = np.array(d)
    if not len(d): return {}
    return {"loc_err_median": round(float(np.median(d)), 3), "within_1": round(float((d <= 1).mean()), 4),
            "within_2": round(float((d <= 2).mean()), 4), "within_5": round(float((d <= 5).mean()), 4), "n_with_pose": len(d)}

def montage(q_img, db_imgs, flags, out, h=240):
    import cv2
    tiles = []
    for j, p in enumerate([q_img] + db_imgs):
        im = cv2.imread(p)
        if im is None: continue
        im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        col = (255, 0, 0) if j == 0 else ((0, 200, 0) if flags[j - 1] else (0, 0, 255))   # blue query, green correct, red wrong
        tiles.append(cv2.copyMakeBorder(im, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=col))
    cv2.imwrite(out, np.hstack(tiles))

def db_descriptors(ex, db_imgs, centers, out, dim):
    """float16 memmap, resumable"""
    dfile, pfile, nfile = os.path.join(out, "db_desc.npy"), os.path.join(out, "db_done.txt"), os.path.join(out, "db_names.json")
    names = [os.path.basename(p) for p in db_imgs]
    done = 0
    if os.path.exists(dfile) and os.path.exists(pfile) and os.path.exists(nfile) and json.load(open(nfile)) == names:
        done = int(open(pfile).read().strip() or 0)
        mm = np.lib.format.open_memmap(dfile, mode="r+")
    else:
        mm = np.lib.format.open_memmap(dfile, mode="w+", dtype=np.float16, shape=(len(db_imgs), dim))
        json.dump(names, open(nfile, "w"))
    if done < len(db_imgs):
        print(f"[db] computing descriptors {done}..{len(db_imgs)}", flush=True)
        for i, f in enumerate(A.extract(ex, db_imgs[done:]), start=done):
            mm[i] = A.vlad(f, centers)
            if (i + 1) % 500 == 0 or i + 1 == len(db_imgs):
                mm.flush(); open(pfile, "w").write(str(i + 1))
    else:
        print("[db] using cached descriptors", flush=True)
    return mm

def add_common_args(ap):
    ap.add_argument("--db-imgs", required=True); ap.add_argument("--db-poses")
    ap.add_argument("--queries", required=True, help="folder holding the IHM query images")
    ap.add_argument("--json", required=True, help="IHM json (use the _v2 file)")
    ap.add_argument("--out", required=True, help="output folder; use a different one per model")
    ap.add_argument("--topk", type=int, default=5); ap.add_argument("--no-montage", action="store_true")
    ap.add_argument("--pos-radius", type=float, default=1.0, help="relaxed ground truth: max camera distance to a GT image (pose units)")
    ap.add_argument("--pos-angle", type=float, default=45.0, help="relaxed ground truth: max difference in viewing direction (degrees)")

def pick_device():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (torch {torch.__version__}, cuda {torch.version.cuda})", flush=True)
    return device

def similarity(q_desc, db_desc):
    sim = np.empty((len(q_desc), len(db_desc)), np.float32)
    for s in range(0, len(db_desc), 2000):
        sim[:, s:s + 2000] = q_desc @ np.asarray(db_desc[s:s + 2000], np.float32).T
    return sim

def load_data(a):
    os.makedirs(os.path.join(a.out, "montage"), exist_ok=True)
    # ---- database ----
    db_imgs = [p for p in list_images(a.db_imgs) if num(p) is not None]
    ids = [num(p) for p in db_imgs]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{len(ids) - len(set(ids))} database images share a file-name number; ids would be ambiguous")
    id2idx = {n: j for j, n in enumerate(ids)}
    print(f"[db] {len(db_imgs)} images, ids {min(ids)}..{max(ids)}")
    db_pos = None
    if a.db_poses:
        pose_of = {num(t): t for t in glob.glob(os.path.join(a.db_poses, "**", "*.txt"), recursive=True) if num(t) is not None}
        db_pos = np.full((len(db_imgs), 3), np.nan); db_fwd = np.full((len(db_imgs), 3), np.nan)
        for j, n in enumerate(ids):
            if n in pose_of:
                try: db_pos[j], db_fwd[j] = A.load_pose(pose_of[n])
                except Exception: pass
        print(f"[db] poses found for {int((~np.isnan(db_pos).any(1)).sum())}/{len(db_imgs)} images")

    # ---- queries + ground truth ----
    items = json.load(open(a.json))
    if isinstance(items, dict): items = items.get("items", list(items.values()))
    q_imgs, q_item, missing_img, by_search = [], [], 0, 0
    pool = list_images(a.queries) + list_images(os.path.dirname(os.path.abspath(a.json)))
    by_name = {os.path.basename(p).lower(): p for p in reversed(pool)}
    by_stem = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in reversed(pool)}
    for k, it in enumerate(items):
        for rel in it.get("image_paths", []):
            rel_n = rel.replace("\\", "/")
            p = next((c for c in (os.path.join(a.queries, rel_n), os.path.join(os.path.dirname(a.json), rel_n), rel_n) if os.path.isfile(c)), None)
            if p is None:      # fall back: same file name (any subfolder, any case), then same name with another extension
                base = os.path.basename(rel_n).lower()
                p = by_name.get(base) or by_stem.get(os.path.splitext(base)[0])
                by_search += p is not None
            if p is None:
                missing_img += 1
                if missing_img <= 5: print("  missing query image:", rel)
                continue
            q_imgs.append(p); q_item.append(k)
    if by_search: print(f"[queries] {by_search} images were located by file name in a subfolder or with another extension")
    if missing_img:
        print(f"[queries] {missing_img} listed images not found. The query folder holds {len(list_images(a.queries))} images, e.g. "
              f"{[os.path.relpath(p, a.queries) for p in list_images(a.queries)[:3]]}")
    q_item = np.array(q_item)
    item_gt, unresolved = {}, []
    for k, it in enumerate(items):
        got = []
        for x in it.get("gt_image_ids", []):
            j = id2idx.get(int(x))
            (got.append(j) if j is not None else unresolved.append(int(x)))
        item_gt[k] = np.unique(np.array(got, int))
    n_refs = sum(len(it.get("gt_image_ids", [])) for it in items)
    print(f"[queries] {len(q_imgs)} images from {len(items)} items ({missing_img} image files missing)")
    print(f"[gt] {n_refs - len(unresolved)}/{n_refs} ground-truth ids found in the database" + (f"; not found e.g. {sorted(set(unresolved))[:8]}" if unresolved else ""))
    no_gt = [items[k].get("Item", k) for k in range(len(items)) if not len(item_gt[k])]
    if no_gt: print(f"[gt] {len(no_gt)} items have no usable ground truth and are skipped: {no_gt[:10]}")
    if not q_imgs: raise SystemExit("no query images found")

    # relaxed ground truth: every database image that looks at the same spot as a listed GT image
    item_gt_rel = None
    if db_pos is not None:
        from scipy.spatial import cKDTree
        has = ~np.isnan(db_pos).any(1); hidx = np.where(has)[0]; tree = cKDTree(db_pos[has])
        item_gt_rel = {}
        for k, gt in item_gt.items():
            got = set(gt.tolist())
            for j in gt:
                if not has[j]: continue
                for m in hidx[tree.query_ball_point(db_pos[j], a.pos_radius)]:
                    if np.degrees(np.arccos(np.clip(db_fwd[m] @ db_fwd[j], -1, 1))) <= a.pos_angle: got.add(int(m))
            item_gt_rel[k] = np.array(sorted(got), int)
        sizes = [len(v) for k, v in item_gt_rel.items() if len(item_gt[k])]
        print(f"[gt] relaxed ground truth (within {a.pos_radius} units and {a.pos_angle} deg of a GT image): "
              f"median {int(np.median(sizes))} correct images per item (strict: {int(np.median([len(v) for v in item_gt.values() if len(v)]))})")

    return dict(db_imgs=db_imgs, ids=ids, db_pos=db_pos, items=items, q_imgs=q_imgs, q_item=q_item,
                item_gt=item_gt, item_gt_rel=item_gt_rel, n_refs=n_refs, unresolved=unresolved)

def evaluate(sim, D, a, info):
    db_imgs, ids, db_pos, items, q_imgs, q_item = D["db_imgs"], D["ids"], D["db_pos"], D["items"], D["q_imgs"], D["q_item"]
    item_gt, item_gt_rel, n_refs, unresolved = D["item_gt"], D["item_gt_rel"], D["n_refs"], D["unresolved"]
    results = dict(info, n_database=len(db_imgs), gt_ids_found=n_refs - len(unresolved), gt_ids_total=n_refs)
    qi = [i for i in range(len(q_imgs)) if len(item_gt[q_item[i]])]
    pq, order = metrics(sim[qi], [item_gt[q_item[i]] for i in qi])
    results["per_image"] = summarise(pq)
    if db_pos is not None: results["per_image"].update(loc_error(order, [item_gt[q_item[i]] for i in qi], db_pos))
    ik = [k for k in range(len(items)) if len(item_gt[k]) and (q_item == k).any()]
    isim = np.stack([sim[q_item == k].max(0) for k in ik])
    pi, iorder = metrics(isim, [item_gt[k] for k in ik])
    results["per_item"] = summarise(pi)
    if db_pos is not None: results["per_item"].update(loc_error(iorder, [item_gt[k] for k in ik], db_pos))
    pi_strict, pi_rel, gt_strict, top1_item = pi, None, item_gt, iorder[:, 0].copy()
    if item_gt_rel is not None:
        results["relaxed"] = {"pos_radius": a.pos_radius, "pos_angle": a.pos_angle}
        rpq, order = metrics(sim[qi], [item_gt_rel[q_item[i]] for i in qi])
        rpi, iorder = metrics(isim, [item_gt_rel[k] for k in ik])
        results["relaxed"]["per_image"] = summarise(rpq); results["relaxed"]["per_item"] = summarise(rpi)
        pi_rel = rpi
        strict_gt, item_gt, pi = item_gt, item_gt_rel, rpi          # montages and failure lists use the relaxed truth
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    meta = lambda it: {f: it.get(f) for f in ("Item", "Hazardous materials", "Location", "Name of equipment", "Component", "Sample No.", "search_query")}
    # one record per item, used by compare_runs.py for paired comparisons between models
    per_item = []
    for r, k in enumerate(ik):
        rec = dict(index=k, n_images=int((q_item == k).sum()), top1_id=ids[top1_item[r]], **meta(items[k]))
        rec["strict"] = {m: float(pi_strict[m][r]) for m in pi_strict}
        if pi_rel is not None: rec["relaxed"] = {m: float(pi_rel[m][r]) for m in pi_rel}
        if db_pos is not None:
            g = db_pos[gt_strict[k]]; g = g[~np.isnan(g).any(1)]; p = db_pos[top1_item[r]]
            if len(g) and not np.isnan(p).any(): rec["loc_err"] = round(float(np.linalg.norm(g - p, axis=1).min()), 3)
        per_item.append(rec)
    json.dump(per_item, open(os.path.join(a.out, "per_item.json"), "w"), indent=1)

    topk, fails = [], []
    for r, i in enumerate(qi):
        gt = item_gt[q_item[i]]; top = order[r, :a.topk]; ok = np.isin(top, gt)
        rec = dict(query=q_imgs[i], item=items[q_item[i]].get("Item"), gt_ids=[ids[j] for j in gt][:30],
                   top=[dict(id=ids[j], img=db_imgs[j], score=round(float(sim[i, j]), 4), correct=bool(c)) for j, c in zip(top, ok)])
        topk.append(rec)
        if not ok[0]: fails.append(rec)
        if not a.no_montage:
            montage(q_imgs[i], [db_imgs[j] for j in top], ok, os.path.join(a.out, "montage", f"{'ok' if ok[0] else 'FAIL'}_{r:04d}_{os.path.basename(q_imgs[i])}.jpg"))
    json.dump(topk, open(os.path.join(a.out, "topk.json"), "w"), indent=1)
    json.dump(fails, open(os.path.join(a.out, "failures.json"), "w"), indent=1)
    json.dump([dict(meta(items[k]), gt_ids=[ids[j] for j in item_gt[k]][:30], top_ids=[ids[j] for j in iorder[r, :a.topk]])
               for r, k in enumerate(ik) if pi["R@1"][r] == 0], open(os.path.join(a.out, "failures_per_item.json"), "w"), indent=1)

    show = lambda d: "  ".join(f"{k} {d[k]:.3f} [{d[k + '_ci95'][0]:.2f}-{d[k + '_ci95'][1]:.2f}]" for k in ("R@1", "R@5", "R@10", "mAP@10"))
    print(f"strict  per image (n={results['per_image']['n']}): {show(results['per_image'])}")
    print(f"strict  per item  (n={results['per_item']['n']}): {show(results['per_item'])}")
    if "relaxed" in results:
        print(f"relaxed per image: {show(results['relaxed']['per_image'])}")
        print(f"relaxed per item : {show(results['relaxed']['per_item'])}")
    if "loc_err_median" in results["per_item"]:
        print(f"localisation error per item: median {results['per_item']['loc_err_median']}, within 1/2/5: "
              f"{results['per_item']['within_1']:.2f}/{results['per_item']['within_2']:.2f}/{results['per_item']['within_5']:.2f} (pose units)")

def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--model", default="dinov2_vitg14"); ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--clusters", type=int, default=32); ap.add_argument("--size", type=int, nargs=2, default=[322, 560])
    ap.add_argument("--vocab-images", type=int, default=300)
    a = ap.parse_args()
    D = load_data(a); db_imgs, q_imgs = D["db_imgs"], D["q_imgs"]

    # ---- features ----
    ex = A.DinoValueFeats(a.model, a.layer, tuple(a.size), pick_device())
    cfile = os.path.join(a.out, "centers.npy")
    if os.path.exists(cfile):
        centers = np.load(cfile)
    else:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(db_imgs), min(a.vocab_images, len(db_imgs)), replace=False)
        local = np.concatenate(list(A.extract(ex, [db_imgs[i] for i in pick])))
        centers = A.fit_vocab(local[rng.choice(len(local), min(len(local), 100000), replace=False)], a.clusters)
        np.save(cfile, centers)
    db_desc = db_descriptors(ex, db_imgs, centers, a.out, centers.size)
    print("[queries] computing descriptors", flush=True)
    q_desc = np.stack([A.vlad(f, centers) for f in A.extract(ex, q_imgs)]).astype(np.float32)


    evaluate(similarity(q_desc, db_desc), D, a, {"method": "AnyLoc", "model": a.model, "layer": a.layer, "clusters": a.clusters})

if __name__ == "__main__":
    main()
