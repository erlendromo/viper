"""AnyLoc on the scan (database) with the IHM photographs (queries).

    python main.py anyloc --db-imgs data/latest/imgs --db-poses data/latest/poses \\
        --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json \\
        --out runs/anyloc

Steps
    1. vocabulary   k-means on patch features of --vocab-images random scan images  (cached: centers.npy)
    2. database     one VLAD descriptor per scan image                               (cached: db_desc.npy)
    3. queries      one VLAD descriptor per IHM photograph
    4. evaluation   cosine similarity, metrics and result files (see viper.utils.evaluation)

The cached files are reused when the command is run again with the same --out, so an interrupted run
resumes where it stopped.
"""
import os

import numpy as np

from viper.utils.cli import add_data_arguments, add_output_arguments, dataset_from_args, pick_device
from viper.utils.descriptors import cached_descriptors, similarity
from viper.utils.evaluation import evaluate

HELP = "AnyLoc: DINOv2 value features + VLAD, no training (the core paper)"

BATCH_SIZE = 8
SEED = 0
MAX_VOCAB_FEATURES = 100_000      # patch features sampled for k-means


def add_arguments(parser):
    add_data_arguments(parser)
    add_output_arguments(parser)
    group = parser.add_argument_group("AnyLoc")
    group.add_argument("--model", default="dinov2_vitg14", help="DINOv2 model on torch.hub (default dinov2_vitg14)")
    group.add_argument("--layer", type=int, default=31, help="transformer block for the value features (default 31)")
    group.add_argument("--clusters", type=int, default=32, help="VLAD vocabulary size (default 32)")
    group.add_argument("--size", type=int, nargs=2, default=[322, 560], metavar=("H", "W"), help="input size (default 322 560)")
    group.add_argument("--vocab-images", type=int, default=300, help="scan images sampled for the vocabulary (default 300)")


def run(args):
    from .features import DinoValueFeatures
    from .vlad import vlad

    data = dataset_from_args(args)
    os.makedirs(args.out, exist_ok=True)
    extractor = DinoValueFeatures(args.model, args.layer, tuple(args.size), pick_device())
    centres = vocabulary(extractor, data.db_images, args.vocab_images, args.clusters, args.out)

    def describe(paths):
        return np.stack([vlad(features, centres) for features in extractor(paths)])

    db_desc = cached_descriptors(describe, data.db_images, args.out, tag="db", batch_size=BATCH_SIZE)
    print("[queries] computing descriptors", flush=True)
    query_desc = np.concatenate([describe(data.query_images[i:i + BATCH_SIZE])
                                 for i in range(0, len(data.query_images), BATCH_SIZE)]).astype(np.float32)

    info = {"method": "AnyLoc", "model": args.model, "layer": args.layer, "clusters": args.clusters}
    evaluate(similarity(query_desc, db_desc), data, args.out, info, top_k=args.topk, montages=not args.no_montage)


def vocabulary(extractor, db_images, n_images, n_clusters, out_dir):
    """VLAD cluster centres from random scan images, loaded from <out>/centers.npy if it exists."""
    from .vlad import fit_vocabulary

    path = os.path.join(out_dir, "centers.npy")
    if os.path.exists(path):
        return np.load(path)
    
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(db_images), min(n_images, len(db_images)), replace=False)
    features = np.concatenate(list(extractor.per_image([db_images[i] for i in chosen], BATCH_SIZE)))
    sample = features[rng.choice(len(features), min(len(features), MAX_VOCAB_FEATURES), replace=False)]
    centres = fit_vocabulary(sample, n_clusters, seed=SEED)
    np.save(path, centres)
    
    return centres
