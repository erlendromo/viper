"""Zero-shot text test with SigLIP2: does the item's text help find the right place?

    python main.py siglip --db-imgs data/latest/imgs --db-poses data/latest/poses \\
        --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json \\
        --megaloc-run runs/megaloc --out runs/siglip_megaloc

Nothing is trained. Three result folders are written inside --out, each readable by `compare`:
    text_only/          item prompt -> SigLIP2 embeddings of the scan images
    siglip_image/       IHM photograph -> SigLIP2 embeddings of the scan images (image to image, for reference)
    megaloc_plus_text/  standardized MegaLoc score + weight * standardized text score (needs --megaloc-run)

The SigLIP2 embeddings of the scan images are cached in --out (db_desc.npy). The MegaLoc database
descriptors are taken from --megaloc-run. If that run used fine-tuned weights (recorded in its
results.json), the photographs are embedded with the same weights. The text weight is fixed (--weight)
and never tuned on the IHM items. A sweep over other weights is printed for analysis only.
"""
import json
import os

import numpy as np

from viper.utils.cli import add_data_arguments, add_output_arguments, dataset_from_args, pick_device
from viper.utils.descriptors import cached_descriptors, encode_in_batches, similarity
from viper.utils.evaluation import evaluate

from .fusion import TEMPLATE, item_prompts, print_weight_sweep, standardize_rows
from .model import CHECKPOINT

HELP = "SigLIP2: item text alone, photograph alone, and MegaLoc + text (no training)"

TEXT_BATCH = 64
DEFAULT_MEGALOC_SIZE = [322, 322]


def add_arguments(parser):
    add_data_arguments(parser)
    add_output_arguments(parser)
    group = parser.add_argument_group("SigLIP2")
    group.add_argument("--siglip", default=CHECKPOINT, help=f"checkpoint (default {CHECKPOINT})")
    group.add_argument("--template", default=TEMPLATE, help=f'prompt around the search_query (default "{TEMPLATE}")')
    group.add_argument("--megaloc-run", help="folder of a finished MegaLoc run; enables the fusion")
    group.add_argument("--weight", type=float, default=1.0, help="fixed weight of the text score in the fusion (default 1.0)")
    group.add_argument("--batch-size", type=int, default=32, help="images per batch (default 32)")


def run(args):
    from .model import SigLIP

    data = dataset_from_args(args)
    device = pick_device()

    # ---- SigLIP2 embeddings: scan images (cached), item prompts, photographs
    siglip = SigLIP(args.siglip, device)
    db = np.asarray(cached_descriptors(siglip.embed_images, data.db_images, args.out, tag="db", batch_size=args.batch_size), np.float32)
    prompts = item_prompts(data.items, args.template)
    print("[text] examples:", prompts[:3], flush=True)

    text_emb = encode_in_batches(siglip.embed_texts, prompts, TEXT_BATCH).astype(np.float32)
    photo_emb = encode_in_batches(siglip.embed_images, data.query_images, args.batch_size).astype(np.float32)
    del siglip                                     # free the GPU memory before MegaLoc is loaded

    text_sim = (text_emb @ db.T)[data.query_item]  # one row per photograph; an item's photographs share its text
    photo_sim = photo_emb @ db.T
    info = {"siglip": args.siglip, "template": args.template}

    def evaluate_in(name, sim, extra):
        evaluate(sim, data, os.path.join(args.out, name), dict(info, **extra), top_k=args.topk, montages=not args.no_montage)

    print("\n===== text only")
    evaluate_in("text_only", text_sim, dict(method="siglip2-text"))
    print("\n===== SigLIP2 image-to-image")
    evaluate_in("siglip_image", photo_sim, dict(method="siglip2-image"))

    if not args.megaloc_run:
        print("\nno --megaloc-run given: skipping the fusion")
        return

    # ---- fusion with MegaLoc
    megaloc_sim, weights = megaloc_similarity(args, data, device)
    image_z, text_z = standardize_rows(megaloc_sim), standardize_rows(text_sim)
    print(f"\n===== MegaLoc + text (fixed weight {args.weight})")

    evaluate_in("megaloc_plus_text", image_z + args.weight * text_z,
                dict(method="megaloc+siglip2-text", weight=args.weight, image_weights=weights))
    print_weight_sweep(image_z, text_z, data)


def megaloc_similarity(args, data, device):
    """MegaLoc similarity of the photographs to the scan, reusing the database descriptors of --megaloc-run.

    Returns (similarity, fine-tuned weights file or None).
    """
    from viper.megaloc.model import load_model
    from viper.utils.embedder import Embedder

    with open(os.path.join(args.megaloc_run, "db_names.json")) as fh:
        names = json.load(fh)

    if names != [os.path.basename(p) for p in data.db_images]:
        raise SystemExit("the MegaLoc run was computed on a different list of database images; cannot fuse")
    
    with open(os.path.join(args.megaloc_run, "results.json")) as fh:
        previous = json.load(fh)

    size, weights = previous.get("size", DEFAULT_MEGALOC_SIZE), previous.get("weights")
    if weights:
        print(f"[fusion] the image model is fine-tuned: {weights}")

    embed = Embedder(load_model(weights), size, device)
    photo_desc = encode_in_batches(embed, data.query_images, args.batch_size).astype(np.float32)
    db_desc = np.load(os.path.join(args.megaloc_run, "db_desc.npy"), mmap_mode="r")
    
    return similarity(photo_desc, db_desc), weights
