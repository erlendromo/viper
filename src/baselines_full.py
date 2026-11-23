"""Zero-shot SALAD and MegaLoc baselines on the full Hestmanden database with the IHM queries.

  python baselines_full.py --model salad   --db-imgs data/latest/imgs --db-poses data/latest/poses \\
      --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json --out run_salad
  python baselines_full.py --model megaloc  ...same arguments...  --out run_megaloc

Needs anyloc_full.py and anyloc_eval.py in the same folder: data loading, ground truth and metrics are shared,
so the numbers are directly comparable with the AnyLoc run. Use a separate --out folder per model.
Models are downloaded by torch.hub on first use:
  salad   -> torch.hub.load("serizba/salad", "dinov2_salad")          (DINOv2-B + SALAD, 8448-d)
  megaloc -> torch.hub.load("gmberton/MegaLoc", "get_trained_model")  (8448-d)
Both are evaluated at 322x322 in their papers, which is the default --size here.
"""
import argparse, json, os
import numpy as np
import anyloc_full as F

HUB = {"salad": ("serizba/salad", "dinov2_salad"), "megaloc": ("gmberton/MegaLoc", "get_trained_model")}

def make_embedder(name, size, device, fp16=False, weights=None):
    """returns embed(list_of_paths) -> float32 array (B, D), L2-normalised"""
    import torch, cv2
    repo, entry = HUB[name]
    model = torch.hub.load(repo, entry)
    if weights:                                           # fine-tuned weights written by train_pose.py
        sd = torch.load(weights, map_location="cpu"); sd = sd.get("state_dict", sd)
        model.load_state_dict(sd); print(f"[model] loaded fine-tuned weights from {weights}", flush=True)
    model = model.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    def embed(paths):
        xs = []
        for p in paths:
            im = cv2.imread(p)
            if im is None: raise SystemExit(f"cannot read image {p}")
            im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (size[1], size[0]), interpolation=cv2.INTER_AREA)
            xs.append(torch.from_numpy(im).permute(2, 0, 1).float() / 255)
        x = (torch.stack(xs).to(device) - mean) / std
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16 and device == "cuda"):
            d = model(x)
        d = torch.nn.functional.normalize(d.float(), dim=1)
        return d.cpu().numpy()
    return embed

def descriptors(embed, paths, out, tag, bs):
    """float16 memmap on disk, resumable (same scheme as anyloc_full.db_descriptors)"""
    import time
    dfile, pfile, nfile = (os.path.join(out, f"{tag}_{x}") for x in ("desc.npy", "done.txt", "names.json"))
    names = [os.path.basename(p) for p in paths]
    done, mm = 0, None
    if all(os.path.exists(f) for f in (dfile, pfile, nfile)) and json.load(open(nfile)) == names:
        done = int(open(pfile).read().strip() or 0); mm = np.lib.format.open_memmap(dfile, mode="r+")
    if done >= len(paths):
        print(f"[{tag}] using cached descriptors", flush=True); return mm
    print(f"[{tag}] computing descriptors {done}..{len(paths)}", flush=True)
    t0 = time.time()
    for i in range(done, len(paths), bs):
        d = embed(paths[i:i + bs])
        if mm is None:
            mm = np.lib.format.open_memmap(dfile, mode="w+", dtype=np.float16, shape=(len(paths), d.shape[1]))
            json.dump(names, open(nfile, "w"))
        mm[i:i + len(d)] = d
        n = i + len(d)
        if (i // bs) % 20 == 0 or n == len(paths):
            mm.flush(); open(pfile, "w").write(str(n))
            print(f"  {n}/{len(paths)} images, {time.time() - t0:.0f}s", flush=True)
    return mm

def main():
    ap = argparse.ArgumentParser()
    F.add_common_args(ap)
    ap.add_argument("--model", required=True, choices=sorted(HUB))
    ap.add_argument("--size", type=int, nargs=2, default=[322, 322], help="H W, multiples of 14")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fp16", action="store_true", help="half precision on GPU (faster, untested with these models)")
    ap.add_argument("--weights", help="fine-tuned weights from train_pose.py (best.pth); use a new --out folder")
    a = ap.parse_args()
    if a.size[0] % 14 or a.size[1] % 14: raise SystemExit("--size values must be multiples of 14")
    D = F.load_data(a)
    embed = make_embedder(a.model, a.size, F.pick_device(), a.fp16, a.weights)
    db_desc = descriptors(embed, D["db_imgs"], a.out, "db", a.batch_size)
    q_desc = np.concatenate([embed(D["q_imgs"][i:i + a.batch_size]) for i in range(0, len(D["q_imgs"]), a.batch_size)]).astype(np.float32)
    F.evaluate(F.similarity(q_desc, db_desc), D, a, {"method": a.model + ("+finetuned" if a.weights else ""), "weights": a.weights, "size": a.size, "descriptor_dim": int(db_desc.shape[1])})

if __name__ == "__main__":
    main()
