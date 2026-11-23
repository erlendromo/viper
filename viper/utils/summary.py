"""One table with the main numbers of every run in a run_all.sh output folder.

    python main.py summary runs_final --radii-extra 0.5 2.0

Reads results.json of every run kept for each radius (<out>/radius_<r>/<run>/) and of the text runs, writes
<out>/summary.csv and prints the table. Recall and mAP are per item with the relaxed ground truth. The
strict columns and the localization error use the annotated frames only.
"""
import csv
import json
import os

HELP = "table of the main numbers of all runs in a run_all.sh folder (summary.csv)"

MAIN_RADIUS = "1.0"
TEXT_RUNS = ("run_text_only", "run_siglip_image", "run_megaloc_plus_text", "run_ft_plus_text")   # only at the main radius


def add_arguments(parser):
    parser.add_argument("out", help="output folder of run_all.sh")
    parser.add_argument("--radii-extra", nargs="*", default=[], help="sensitivity radii, as used in run_all.sh")


def run(args):
    rows = []

    for radius in [MAIN_RADIUS] + args.radii_extra:
        for name, results in runs_at(args.out, radius):
            rows.append(summary_row(radius, name, results))

    path = os.path.join(args.out, "summary.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n================ summary (per item, relaxed ground truth)")
    print(f"{'radius':<7}{'run':<26}{'R@1':>7}{'  95% CI':>14}{'R@5':>7}{'R@10':>7}{'mAP@10':>8}{'loc':>7}")

    for x in rows:
        print(f"{x['radius']:<7}{x['run']:<26}{x['R1']:>7.3f}   [{x['R1_lo']:.2f}-{x['R1_hi']:.2f}]"
              f"{x['R5']:>7.3f}{x['R10']:>7.3f}{x['mAP10']:>8.3f}{str(x['loc_median']):>7}")
        
    seeds = [x["R1"] for x in rows if x["radius"] == MAIN_RADIUS and x["run"].startswith("run_ft_head_s") and "best" not in x["run"]]
    if seeds:
        print(f"\nfine-tuned (head only), {len(seeds)} seeds at radius {MAIN_RADIUS}: "
              f"R@1 mean {sum(seeds) / len(seeds):.3f}, min {min(seeds):.3f}, max {max(seeds):.3f}")
        
    print(f"saved {path}")


def runs_at(out, radius):
    """(run name, results.json contents) of every run kept for this radius, sorted by name; text runs last."""
    base = os.path.join(out, f"radius_{radius}")
    names = sorted(os.listdir(base)) if os.path.isdir(base) else []

    if radius == MAIN_RADIUS:
        names += [n for n in TEXT_RUNS if os.path.isdir(os.path.join(out, n))]

    for name in names:
        path = os.path.join(base, name, "results.json")

        if not os.path.exists(path):
            path = os.path.join(out, name, "results.json")

        with open(path) as fh:
            yield name, json.load(fh)


def summary_row(radius, name, results):
    relaxed = results.get("relaxed", {}).get("per_item") or results["per_item"]
    strict = results["per_item"]
    
    return dict(radius=radius, run=name, items=relaxed["n"],
                R1=relaxed["R@1"], R1_lo=relaxed["R@1_ci95"][0], R1_hi=relaxed["R@1_ci95"][1],
                R5=relaxed["R@5"], R10=relaxed["R@10"], mAP10=relaxed["mAP@10"],
                strict_R1=strict["R@1"], strict_R10=strict["R@10"],
                loc_median=strict.get("loc_err_median", ""), within_1=strict.get("within_1", ""))
