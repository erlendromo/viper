"""Statistics of the dataset: the scan, the IHM items and photographs, and the ground truth.

    python main.py dataset-stats --db-imgs data/latest/imgs --db-poses data/latest/poses \\
        --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json \\
        --out runs/dataset_stats

Evaluated items are those with at least one photograph and at least one annotated scan frame, as in
the evaluation. Printed and written to --out
    counts          scan frames, items, photographs, annotated frame ids
    photographs     photographs per item
    ground truth    correct scan frames per item, strict and relaxed at each radius in --radii, and the
                    Recall@1 a random ranking would reach (the share of the scan that counts as correct)
    materials       items per hazardous material
    decks           items per deck named in the IHM, and their annotated frames per deck level of the poses
    image sizes     resolutions of the photographs and the scan frames

Files: dataset_stats.json and dataset_stats.png. Needs no GPU and no models.
"""
import json
import os
from collections import Counter

import numpy as np

from .cli import add_data_arguments, dataset_from_args
from .dataset import relaxed_ground_truth
from .trajectory import add_deck_arguments, decks_from_args

HELP = "statistics of the scan, the IHM items and the ground truth (dataset_stats.json)"

RADII = [0.5, 1.0, 2.0]
COLOURS = ["#014693", "#3CBFBE", "#ED8013", "#B01B81"]


def add_arguments(parser):
    add_data_arguments(parser)
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--radii", type=float, nargs="+", default=RADII, help="radii of the relaxed ground truth (default 0.5 1.0 2.0)")
    add_deck_arguments(parser)


def run(args):
    if not args.db_poses:
        raise SystemExit("--db-poses is needed for the relaxed ground truth and the deck levels")
    
    data = dataset_from_args(args)
    os.makedirs(args.out, exist_ok=True)
    evaluated = [k for k in range(len(data.items)) if len(data.gt_strict[k]) and (data.query_item == k).any()]

    stats = dict(
        counts=counts(data, evaluated),
        photographs_per_item=photographs_per_item(data, evaluated),
        ground_truth=ground_truth_sizes(data, evaluated, args.radii, args.pos_angle),
        materials=items_by(data, evaluated, material),
        decks=items_by(data, evaluated, named_deck),
        frames_per_level=annotated_frames_per_level(data, evaluated, args),
        image_sizes=image_sizes(data),
    )

    with open(os.path.join(args.out, "dataset_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    print_summary(stats)
    plot(data, evaluated, stats, args.out)
    print(f"saved {os.path.join(args.out, 'dataset_stats.png')} and dataset_stats.json")


# ------------------------------------------------------------------------------------------------ statistics

def material(item):
    return str(item.get("Hazardous materials") or "unknown").strip()


def named_deck(item):
    """The deck in the IHM location, e.g. "Upper Deck, Galley" -> "Upper Deck"."""
    return str(item.get("Location") or "unknown").split(",")[0].strip()


def photo_counts(data):
    """Number of photographs found for every item (0 if none was found)."""
    return np.bincount(data.query_item, minlength=len(data.items)) if len(data.query_item) else np.zeros(len(data.items), int)


def distribution(values):
    values = np.asarray(values)

    return dict(n=int(len(values)), min=int(values.min()), median=float(np.median(values)),
                mean=round(float(values.mean()), 2), max=int(values.max()))


def counts(data, evaluated):
    photos = photo_counts(data)
    has_pose = int((~np.isnan(data.db_centres).any(1)).sum())

    return dict(scan_frames=len(data.db_images), scan_frames_with_pose=has_pose,
                items=len(data.items), items_with_photographs=int((photos > 0).sum()), photographs=int(photos.sum()),
                evaluated_items=len(evaluated), photographs_of_evaluated_items=int(photos[evaluated].sum()),
                annotated_frame_ids=data.gt_ids_total, annotated_frame_ids_in_scan=data.gt_ids_total - len(data.gt_ids_missing))


def photographs_per_item(data, evaluated):
    photos = photo_counts(data)
    with_photos = photos[photos > 0]

    return dict(all_items=dict(distribution(with_photos), histogram={int(k): int(v) for k, v in sorted(Counter(with_photos).items())}),
                evaluated_items=dict(distribution(photos[evaluated]),
                                     histogram={int(k): int(v) for k, v in sorted(Counter(photos[evaluated]).items())}))


def ground_truth_sizes(data, evaluated, radii, max_angle):
    """Correct scan frames per evaluated item, and the Recall@1 of a random ranking (share of the scan that is correct)."""
    n_frames = len(data.db_images)

    def summary(truth):
        sizes = [len(truth[k]) for k in evaluated]

        return dict(distribution(sizes), chance_recall_at_1=round(float(np.mean(sizes) / n_frames), 5))

    result = {"strict": summary(data.gt_strict)}
    for radius in radii:
        relaxed = relaxed_ground_truth(data.gt_strict, data.db_centres, data.db_forwards, radius, max_angle)
        result[f"relaxed_r{radius}"] = summary(relaxed)

    return result


def items_by(data, evaluated, key):
    """Items per category, for all items and for the evaluated items, largest first."""
    everything = Counter(key(item) for item in data.items)
    scored = Counter(key(data.items[k]) for k in evaluated)

    return [dict(name=name, all_items=n, evaluated_items=scored.get(name, 0)) for name, n in everything.most_common()]


def annotated_frames_per_level(data, evaluated, args):
    """Annotated frames of the evaluated items per deck level of the poses, split by the deck named in the IHM."""
    posed = np.where(~np.isnan(data.db_centres).any(1))[0]
    decks = decks_from_args(data.db_centres[posed], args)
    level = np.full(len(data.db_images), -1)
    level[posed] = decks["deck"]
    levels = [round(float(x), 2) for x in decks["levels"]]

    table = {}
    for k in evaluated:
        row = table.setdefault(named_deck(data.items[k]), [0] * len(levels))
        for frame in data.gt_strict[k]:
            if level[frame] >= 0:
                row[level[frame]] += 1

    order = sorted(table, key=lambda name: -np.dot(table[name], np.arange(len(levels))) / max(1, sum(table[name])))

    return dict(levels=levels, by_named_deck={name: table[name] for name in order})


def image_sizes(data):
    """The most common resolutions (width x height) of the photographs and the scan frames."""
    from PIL import Image

    def sizes(paths):
        found = Counter()
        for path in paths:
            with Image.open(path) as image:
                found[f"{image.size[0]}x{image.size[1]}"] += 1

        return dict(found.most_common(5))

    return dict(photographs=sizes(data.query_images), scan_frames=sizes(data.db_images))


# ------------------------------------------------------------------------------------------------ output

def print_summary(s):
    c = s["counts"]
    print(f"\n[counts] scan frames {c['scan_frames']} ({c['scan_frames_with_pose']} with pose), items {c['items']} "
          f"({c['items_with_photographs']} with photographs), photographs {c['photographs']}")
    print(f"[counts] evaluated items {c['evaluated_items']} with {c['photographs_of_evaluated_items']} photographs, "
          f"annotated frame ids {c['annotated_frame_ids_in_scan']}/{c['annotated_frame_ids']} in the scan")
    
    for group, p in s["photographs_per_item"].items():
        print(f"[photographs per item] {group}: median {p['median']}, mean {p['mean']}, max {p['max']}, histogram {p['histogram']}")

    for name, g in s["ground_truth"].items():
        print(f"[ground truth] {name:<12} correct frames per item: min {g['min']}, median {g['median']}, mean {g['mean']}, "
              f"max {g['max']}  random R@1 {100 * g['chance_recall_at_1']:.2f}%")
        
    for title, rows in (("materials", s["materials"]), ("decks named in the IHM", s["decks"])):
        print(f"[{title}] (all items / evaluated)")
        for r in rows:
            print(f"    {r['name']:<32}{r['all_items']:>5}{r['evaluated_items']:>6}")

    f = s["frames_per_level"]
    print(f"[annotated frames per pose level] levels {f['levels']}")

    for name, row in f["by_named_deck"].items():
        print(f"    {name:<32}{''.join(f'{n:>6}' for n in row)}")

    print(f"[image sizes] photographs {s['image_sizes']['photographs']}, scan frames {s['image_sizes']['scan_frames']}")


def plot(data, evaluated, stats, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    fig, (photos_ax, gt_ax, mat_ax) = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw=dict(wspace=0.35))

    # photographs per item
    hist = stats["photographs_per_item"]["evaluated_items"]["histogram"]
    photos_ax.bar(list(hist), list(hist.values()), color=COLOURS[0])
    photos_ax.set_xlabel("photographs per item")
    photos_ax.set_ylabel("evaluated items")
    photos_ax.set_xticks(list(hist))

    # correct frames per item, log scale
    radii = [name for name in stats["ground_truth"] if name.startswith("relaxed")]
    sizes = {"strict": [len(data.gt_strict[k]) for k in evaluated]}

    for name in radii:
        radius = float(name.split("_r")[1])
        relaxed = relaxed_ground_truth(data.gt_strict, data.db_centres, data.db_forwards, radius, data.pos_angle)
        sizes[f"relaxed, r = {radius:g} m"] = [len(relaxed[k]) for k in evaluated]

    top = max(max(v) for v in sizes.values())
    bins = np.logspace(0, np.log10(top + 1) + 0.05, 25)

    for colour, (label, values) in zip(COLOURS, sizes.items()):
        gt_ax.hist(values, bins=bins, histtype="step", linewidth=1.5, color=colour, label=label)

    gt_ax.set_xscale("log")
    gt_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    gt_ax.set_xlabel("correct scan frames per item")
    gt_ax.set_ylabel("evaluated items")
    gt_ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    # items per material
    rows = stats["materials"][::-1]
    y = np.arange(len(rows))
    mat_ax.barh(y + 0.2, [r["all_items"] for r in rows], height=0.4, color="#B0B0B0", label="all items")
    mat_ax.barh(y - 0.2, [r["evaluated_items"] for r in rows], height=0.4, color=COLOURS[0], label="evaluated")
    mat_ax.set_yticks(y, [r["name"] for r in rows], fontsize=8)
    mat_ax.set_xlabel("items")
    mat_ax.legend(fontsize=7, frameon=False)

    for ax in (photos_ax, gt_ax):
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    mat_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    for ax in (photos_ax, gt_ax, mat_ax):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
            
    fig.savefig(os.path.join(out_dir, "dataset_stats.png"), dpi=200, bbox_inches="tight")
