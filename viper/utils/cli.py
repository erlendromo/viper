"""Command-line arguments shared by every retrieval command, and the device choice."""
from .dataset import load_dataset


def add_data_arguments(parser):
    group = parser.add_argument_group("data")
    group.add_argument("--db-imgs", required=True, help="folder with the scan images (database)")
    group.add_argument("--db-poses", help="folder with the pose files; enables the relaxed ground truth and the localization error")
    group.add_argument("--queries", required=True, help="folder with the IHM photographs")
    group.add_argument("--json", required=True, help="IHM json (use the _v2 file)")
    group.add_argument("--pos-radius", type=float, default=1.0, help="relaxed ground truth: max camera distance to a GT image in meters (default 1.0)")
    group.add_argument("--pos-angle", type=float, default=45.0, help="relaxed ground truth: max difference in viewing direction in degrees (default 45)")


def add_output_arguments(parser):
    group = parser.add_argument_group("output")
    group.add_argument("--out", required=True, help="output folder; use a different one per model")
    group.add_argument("--topk", type=int, default=5, help="scan images kept per photograph in topk.json and the montages (default 5)")
    group.add_argument("--no-montage", action="store_true", help="do not write montage images")


def dataset_from_args(args):
    return load_dataset(args.db_imgs, args.queries, args.json, args.db_poses, args.pos_radius, args.pos_angle)


def pick_device():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (torch {torch.__version__}, cuda {torch.version.cuda})", flush=True)
    
    return device
