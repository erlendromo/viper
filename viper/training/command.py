"""Fine-tune MegaLoc on the scan images, with training pairs taken from the camera poses.

    python main.py train --db-imgs data/latest/imgs --db-poses data/latest/poses \\
        --baseline-run runs/megaloc --out runs/train_head_s0 --epochs 1 --seed 0

    add --dry-run first for a two-minute check that everything works

Steps
    1. pairs          positives within --pos-radius and --pos-angle, negatives beyond --neg-radius
    2. split          blocks of held-out validation frames; frames near them are not used for training
    3. hard negatives for every frame, the negatives the baseline finds most similar, mined from the cached
                      descriptors of --baseline-run (cached: hard_negatives_*.npy)
    4. training       backbone frozen, aggregation head trained with a multi-similarity loss; each anchor
                      is an augmented close-up, its positives and hard negatives are full frames
    5. validation     close-ups of the held-out frames against the scan, before training and after every epoch

Weights: last.pth (most recent epoch) and best.pth (best validation score). Score them on the IHM items with
    python main.py megaloc --weights runs/train_head_s0/last.pth ...usual arguments... --out runs/megaloc_ft_s0
"""
import json
import os

import numpy as np

from .model import MODELS

HELP = "fine-tune MegaLoc on the scan with pose-derived pairs (the IHM photographs are not used)"

DRY_RUN = dict(epochs=1, iters=3, val_queries=20, val_db_stride=8, anchors=4)
DRY_RUN_ANCHORS = 64           # anchors mined for hard negatives in a dry run


def add_arguments(parser):
    # the order of the arguments is the order of the "args" record in train_log.json
    data = parser.add_argument_group("data")
    data.add_argument("--db-imgs", required=True, help="folder with the scan images")
    data.add_argument("--db-poses", required=True, help="folder with the pose files")
    data.add_argument("--baseline-run", required=True, help="finished run of the same model (its cached descriptors give the hard negatives)")
    data.add_argument("--out", required=True, help="output folder")
    data.add_argument("--model", default="megaloc", choices=sorted(MODELS))
    data.add_argument("--size", type=int, nargs=2, default=[322, 322], metavar=("H", "W"), help="input size (default 322 322)")

    pairs = parser.add_argument_group("pairs from poses")
    pairs.add_argument("--pos-radius", type=float, default=1.0, help="positive: cameras within this many meters ... (default 1.0)")
    pairs.add_argument("--pos-angle", type=float, default=45.0, help="... and viewing directions within this many degrees (default 45)")
    pairs.add_argument("--neg-radius", type=float, default=4.0, help="negative: cameras further apart than this (default 4.0)")
    pairs.add_argument("--pos-min-dist", type=float, default=0.0, help="informative positives: camera moved at least this far ...")
    pairs.add_argument("--pos-min-angle", type=float, default=0.0, help="... or view turned at least this many degrees ...")
    pairs.add_argument("--pos-min-gap", type=int, default=0, help="... or at least this many frames apart (another pass). 0 = off")
    pairs.add_argument("--keyframe-dist", type=float, default=0.0, help="thin training frames: keep one when the camera moved this far ...")
    pairs.add_argument("--keyframe-angle", type=float, default=10.0, help="... or turned this many degrees (only with --keyframe-dist > 0)")

    batches = parser.add_argument_group("batches")
    batches.add_argument("--anchors", type=int, default=16, help="anchors per batch (default 16)")
    batches.add_argument("--n-pos", type=int, default=3, help="positives per anchor (default 3)")
    batches.add_argument("--n-hard", type=int, default=2, help="hard negatives per anchor (default 2)")
    batches.add_argument("--crop-min", type=float, default=0.15, help="smallest close-up crop, as a share of the image area (default 0.15)")

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--lr", type=float, default=1e-5, help="learning rate after warm-up (default 1e-5)")
    optim.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay (default 1e-3)")
    optim.add_argument("--epochs", type=int, default=10, help="maximum number of epochs (default 10)")
    optim.add_argument("--iters", type=int, default=300, help="batches per epoch (default 300)")
    optim.add_argument("--patience", type=int, default=3, help="stop after this many epochs without improvement (default 3)")
    optim.add_argument("--workers", type=int, default=8, help="data loader processes (default 8)")
    optim.add_argument("--unfreeze-blocks", type=int, default=0, help="also train the last N backbone blocks (0 = head only)")
    optim.add_argument("--head-prefixes", nargs="*", default=None, help="override which modules count as the head")

    val = parser.add_argument_group("validation")
    val.add_argument("--val-blocks", type=int, default=20, help="blocks of held-out frames (default 20)")
    val.add_argument("--val-block-len", type=int, default=40, help="consecutive frames per block (default 40)")
    val.add_argument("--val-queries", type=int, default=300, help="close-up queries drawn from the held-out frames (default 300)")
    val.add_argument("--val-db-stride", type=int, default=4, help="the validation database is every Nth scan frame (default 4)")
    val.add_argument("--val-area", type=float, default=0.10, help="size of the validation close-up, as a share of the image area "
                     "(smaller = harder; aim for an epoch-0 validation R@1 of roughly 0.4-0.8)")
    val.add_argument("--eval-batch", type=int, default=64, help="batch size for computing validation descriptors (default 64)")

    run_group = parser.add_argument_group("run")
    run_group.add_argument("--seed", type=int, default=0, help="random seed (anchors, augmentation, validation queries)")
    run_group.add_argument("--select", default="mean", choices=["mean", "R@1"],
                           help="validation score that picks best.pth: mean of R@1/R@5/R@10 (default) or R@1 alone")
    run_group.add_argument("--dry-run", action="store_true", help="tiny run to check that everything works")


def run(args):
    from .pairs import keyframes, load_posed_frames, make_split, positives_of
    from .validation import build_validation_set

    os.makedirs(args.out, exist_ok=True)
    if args.dry_run:
        for name, value in DRY_RUN.items():
            setattr(args, name, value)
    
    run_args = {k: v for k, v in vars(args).items() if k not in ("command", "run")}

    # ---- pairs and split (numpy only)
    images, pos, fwd = load_posed_frames(args.db_imgs, args.db_poses)
    print(f"[data] {len(images)} frames with poses")
    validation, training = make_split(pos, args.val_blocks, args.val_block_len, args.pos_radius)

    if args.keyframe_dist > 0:
        before = int(training.sum())
        training &= keyframes(pos, fwd, args.keyframe_dist, args.keyframe_angle)
        print(f"[data] keyframe thinning: {before} -> {int(training.sum())} training frames")
    
    positives = positives_of(pos, fwd, args.pos_radius, args.pos_angle, training,
                             args.pos_min_dist, args.pos_min_angle, args.pos_min_gap)
    anchors = np.array([i for i in np.where(training)[0] if len(positives[i])])
    print(f"[data] train frames {int(training.sum())} ({len(anchors)} usable as anchors), validation frames {int(validation.sum())}, "
          f"median positives per anchor {int(np.median([len(positives[i]) for i in anchors]))}")

    hard, anchors = hard_negatives(args, images, pos, training, anchors)
    print(f"[mining] anchors with at least one hard negative: {int((hard[anchors] >= 0).any(1).sum())}/{len(anchors)}")

    val_set = build_validation_set(pos, fwd, validation, args.val_queries, args.val_db_stride,
                                   args.pos_radius, args.pos_angle, args.seed)
    print(f"[val] {len(val_set.queries)} close-up queries against {len(val_set.database)} scan frames")

    if not len(val_set.queries):
        raise SystemExit("no validation query has a positive in the validation database: lower --val-db-stride")

    train_model(args, run_args, images, pos, fwd, anchors, positives, hard, val_set)


def hard_negatives(args, images, pos, training, anchors):
    """Hard negatives per frame, cached in the output folder. A dry run mines only the first anchors.

    Returns (hard negatives, anchors); the anchors are cut to the mined ones in a dry run.
    """
    from .pairs import mine_hard_negatives

    with open(os.path.join(args.baseline_run, "db_names.json")) as fh:
        row_of = {name: row for row, name in enumerate(json.load(fh))}
    
    if any(os.path.basename(p) not in row_of for p in images):
        raise SystemExit("--baseline-run does not cover all frames")

    # the file name records the settings the negatives depend on, so run_all.sh can share it between runs
    path = os.path.join(args.out, f"hard_negatives_kf{args.keyframe_dist}_{args.keyframe_angle}_neg{args.neg_radius}.npy")
    if os.path.exists(path) and len(np.load(path)) == len(images):
        print("[mining] using cached hard negatives")
        return np.load(path), anchors

    desc = np.load(os.path.join(args.baseline_run, "db_desc.npy"), mmap_mode="r")
    desc = np.asarray(desc, np.float32)[[row_of[os.path.basename(p)] for p in images]]

    if args.dry_run:
        anchors = anchors[:DRY_RUN_ANCHORS]
        return mine_hard_negatives(desc, pos, args.neg_radius, training, rows=anchors), anchors
    
    hard = mine_hard_negatives(desc, pos, args.neg_radius, training)
    np.save(path, hard)
    
    return hard, anchors


def train_model(args, run_args, images, pos, fwd, anchors, positives, hard, val_set):
    import torch

    from viper.utils.cli import pick_device

    from .batches import AnchorGroups, collate_groups
    from .model import select_trainable
    from .trainer import train
    from .validation import validate

    # the order of everything that draws random numbers is fixed: seed, model, then the data loader
    torch.manual_seed(args.seed)
    device = pick_device()
    model = MODELS[args.model](None).to(device).eval()     # eval mode throughout: dropout off
    params = select_trainable(model, args.unfreeze_blocks, args.head_prefixes)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    centres, forwards = torch.tensor(pos, device=device), torch.tensor(fwd, device=device)

    groups = AnchorGroups(images, anchors, positives, hard, tuple(args.size), args.n_pos, args.n_hard,
                          args.crop_min, args.iters * args.anchors)
    loader = torch.utils.data.DataLoader(groups, batch_size=args.anchors, num_workers=args.workers, collate_fn=collate_groups,
                                         drop_last=True, persistent_workers=args.workers > 0)

    def run_validation():
        return validate(model, images, val_set, args.size, device, args.eval_batch, args.seed, args.val_area)

    train(model, params, optimizer, loader, run_validation, centres, forwards, args, run_args, device)
