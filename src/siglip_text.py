"""Zero-shot text test with SigLIP2: does the item's search_query help find the right place?

  python siglip_text.py --db-imgs data/latest/imgs --db-poses data/latest/poses \\
      --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json \\
      --megaloc-run run_megaloc --out run_siglip

Needs anyloc_full.py, baselines_full.py and anyloc_eval.py in the same folder, and:  pip install transformers pillow
Nothing is trained. It writes three result folders inside --out, each readable by compare_runs.py:
  text_only/          search_query text  -> SigLIP2 image embeddings of the scan images
  siglip_image/       query photo        -> SigLIP2 image embeddings (image-to-image, for reference)
  megaloc_plus_text/  MegaLoc image score + weight * text score (both standardised per query); needs --megaloc-run
The fusion weight is fixed (--weight, default 1.0) and NOT tuned on the IHM items. A sweep over other weights is
printed for analysis only.
"""
import argparse, copy, json, os
import numpy as np
import anyloc_full as F
import baselines_full as B

def make_siglip(ckpt, device):
    """returns (embed_images(paths) -> (B,D), embed_texts(strings) -> (B,D)), both L2-normalised float32"""
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(ckpt).to(device).eval()
    proc = AutoProcessor.from_pretrained(ckpt)
    def feats(out):                                   # tensor in older transformers, output object in newer ones
        t = out if torch.is_tensor(out) else out.pooler_output
        return torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()
    def embed_images(paths):
        ims = [Image.open(p).convert("RGB") for p in paths]
        inp = proc(images=ims, return_tensors="pt").to(device)
        with torch.no_grad(): return feats(model.get_image_features(**inp))
    def embed_texts(texts):
        inp = proc(text=[t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad(): return feats(model.get_text_features(**inp))
    return embed_images, embed_texts

def zrow(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)

def item_scores(sim, D):
    """per-item Recall@1 / @10 with the relaxed truth (strict if no poses); used for the printed sweep only"""
    gt = D["item_gt_rel"] or D["item_gt"]; q_item = D["q_item"]
    ik = [k for k in range(len(D["items"])) if len(D["item_gt"][k]) and (q_item == k).any()]
    m, _ = F.metrics(np.stack([sim[q_item == k].max(0) for k in ik]), [gt[k] for k in ik])
    return m["R@1"].mean(), m["R@10"].mean()

def sub(a, name):
    b = copy.copy(a); b.out = os.path.join(a.out, name); os.makedirs(os.path.join(b.out, "montage"), exist_ok=True); return b

def main():
    ap = argparse.ArgumentParser()
    F.add_common_args(ap)
    ap.add_argument("--siglip", default="google/siglip2-so400m-patch14-384", help="smaller/faster: google/siglip2-base-patch16-224")
    ap.add_argument("--template", default="This is a photo of {}.", help="prompt around the search_query")
    ap.add_argument("--megaloc-run", help="folder of a finished MegaLoc run (its cached database descriptors are reused)")
    ap.add_argument("--weight", type=float, default=1.0, help="fixed weight of the text score in the fusion")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    D = F.load_data(a); items, q_item = D["items"], D["q_item"]
    device = F.pick_device()

    embed_images, embed_texts = make_siglip(a.siglip, device)
    db = B.descriptors(embed_images, D["db_imgs"], a.out, "db", a.batch_size)
    db32 = np.asarray(db, np.float32)
    texts = [a.template.format(str(it.get("search_query") or it.get("Name of equipment") or "").strip()) for it in items]
    print("[text] examples:", texts[:3], flush=True)
    t_emb = np.concatenate([embed_texts(texts[i:i + 64]) for i in range(0, len(texts), 64)]).astype(np.float32)
    q_emb = np.concatenate([embed_images(D["q_imgs"][i:i + a.batch_size]) for i in range(0, len(D["q_imgs"]), a.batch_size)]).astype(np.float32)

    sim_text = (t_emb @ db32.T)[q_item]               # one row per query photo (the text is the same for an item's photos)
    sim_simg = q_emb @ db32.T
    info = {"siglip": a.siglip, "template": a.template}
    print("\n===== text only"); F.evaluate(sim_text, D, sub(a, "text_only"), dict(info, method="siglip2-text"))
    print("\n===== SigLIP2 image-to-image"); F.evaluate(sim_simg, D, sub(a, "siglip_image"), dict(info, method="siglip2-image"))

    if not a.megaloc_run:
        print("\nno --megaloc-run given: skipping the fusion"); return
    names = json.load(open(os.path.join(a.megaloc_run, "db_names.json")))
    if names != [os.path.basename(p) for p in D["db_imgs"]]:
        raise SystemExit("the MegaLoc run was computed on a different list of database images; cannot fuse")
    del embed_images, embed_texts
    prev = json.load(open(os.path.join(a.megaloc_run, "results.json")))
    size, weights = prev.get("size", [322, 322]), prev.get("weights")      # a fine-tuned run records its weights file
    if weights: print(f"[fusion] the image model is fine-tuned: {weights}")
    mega = B.make_embedder("megaloc", size, device, False, weights)
    mq = np.concatenate([mega(D["q_imgs"][i:i + a.batch_size]) for i in range(0, len(D["q_imgs"]), a.batch_size)]).astype(np.float32)
    sim_mega = F.similarity(mq, np.load(os.path.join(a.megaloc_run, "db_desc.npy"), mmap_mode="r"))
    zi, zt = zrow(sim_mega), zrow(sim_text)
    print(f"\n===== MegaLoc + text (fixed weight {a.weight})")
    F.evaluate(zi + a.weight * zt, D, sub(a, "megaloc_plus_text"), dict(info, method="megaloc+siglip2-text", weight=a.weight, image_weights=weights))

    print("\nweight sweep, per item (analysis only, do not pick the weight from this):")
    print(f"  {'weight':>8}{'R@1':>8}{'R@10':>8}")
    for w in (0, 0.25, 0.5, 1, 2, 4):
        r1, r10 = item_scores(zi + w * zt, D); print(f"  {w:>8}{r1:>8.3f}{r10:>8.3f}" + ("   <- MegaLoc alone" if w == 0 else ""))
    r1, r10 = item_scores(zt, D); print(f"  {'text':>8}{r1:>8.3f}{r10:>8.3f}")

if __name__ == "__main__":
    main()
