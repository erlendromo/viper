"""The retrieval flow shared by the global-descriptor models (MegaLoc, SALAD).

    1. database   one descriptor per scan image      (cached: db_desc.npy, resumes after an interruption)
    2. queries    one descriptor per IHM photograph
    3. evaluation cosine similarity, metrics and result files (see viper.utils.evaluation)
"""
import numpy as np

from .cli import dataset_from_args, pick_device
from .descriptors import cached_descriptors, encode_in_batches, similarity
from .embedder import Embedder
from .evaluation import evaluate

PATCH_SIZE = 14


def add_model_arguments(parser, default_size):
    group = parser.add_argument_group("model")
    group.add_argument("--size", type=int, nargs=2, default=list(default_size), metavar=("H", "W"),
                       help=f"input size, multiples of {PATCH_SIZE} (default {default_size[0]} {default_size[1]})")
    group.add_argument("--batch-size", type=int, default=32, help="images per batch (default 32)")
    group.add_argument("--fp16", action="store_true", help="half precision on the GPU (faster, not used for the reported results)")
    group.add_argument("--weights", help="fine-tuned weights from training (last.pth); use a new --out folder")


def run_retrieval(args, method, load_model):
    """Score a global-descriptor model on the IHM photographs.

    method      name written to results.json ("+finetuned" is added when --weights is given)
    load_model  function weights_path_or_None -> torch module, called after the data is loaded
    """
    if args.size[0] % PATCH_SIZE or args.size[1] % PATCH_SIZE:
        raise SystemExit(f"--size values must be multiples of {PATCH_SIZE}")
    
    data = dataset_from_args(args)
    device = pick_device()
    embed = Embedder(load_model(args.weights), args.size, device, args.fp16)

    db_desc = cached_descriptors(embed, data.db_images, args.out, tag="db", batch_size=args.batch_size)
    query_desc = encode_in_batches(embed, data.query_images, args.batch_size).astype(np.float32)

    info = {"method": method + ("+finetuned" if args.weights else ""), "weights": args.weights,
            "size": args.size, "descriptor_dim": int(db_desc.shape[1])}
    
    evaluate(similarity(query_desc, db_desc), data, args.out, info, top_k=args.topk, montages=not args.no_montage)
