"""Compare several finished runs on the same items (paired).

    python main.py compare runs/anyloc runs/salad runs/megaloc --out comparisons/baselines.txt
    python main.py compare runs/anyloc runs/salad runs/megaloc --metric mAP@10 --out comparisons/baselines_map.txt

Every run folder needs the per_item.json written by the retrieval commands. The report is printed and
saved to --out. It contains
    a summary table             Recall@1/5/10, mAP@10 and localization error per run
    pairwise tests              --metric for every pair of runs: an exact McNemar test for Recall@K, a paired
                                sign-flip permutation test for mAP@10, and a paired bootstrap interval
    combining models            the best single run against "any run right" (a perfect selector per item)
    missed items                the items no run finds in the top 10, and the items only one run gets right
"""
import json
import os
import sys

import numpy as np

from .stats import bootstrap_difference, mcnemar_exact, permutation_test

HELP = "paired comparison of finished runs (summary, tests, missed items)"

METRICS = ("R@1", "R@5", "R@10", "mAP@10")


def add_arguments(parser):
    parser.add_argument("runs", nargs="+", help="run folders, each with a per_item.json")
    parser.add_argument("--truth", default="relaxed", choices=["relaxed", "strict"], help="ground truth (default relaxed)")
    parser.add_argument("--out", default="comparison.txt", help="text file for the report (default comparison.txt)")
    parser.add_argument("--metric", default="R@1", choices=list(METRICS),
                        help="metric for the pairwise tests. Recall metrics use an exact McNemar test; mAP@10 uses a paired sign-flip permutation test")


def load_run(folder):
    """The per-item records of a run, keyed by item index."""
    path = os.path.join(folder, "per_item.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: re-run that model with the current scripts (cached, so quick)")

    with open(path) as fh:
        return {record["index"]: record for record in json.load(fh)}


def run(args):
    names = [os.path.basename(os.path.normpath(r)) for r in args.runs]
    runs = [load_run(r) for r in args.runs]

    for name, records in zip(names, runs):
        if args.truth not in next(iter(records.values())):
            sys.exit(f"{name} has no '{args.truth}' results (was it run without poses?)")

    common = sorted(set.intersection(*[set(records) for records in runs]))
    if any(len(records) != len(common) for records in runs):
        print(f"note: runs cover different items; comparing the {len(common)} they share")

    scores = {name: {m: np.array([records[k][args.truth][m] for k in common]) for m in METRICS}
              for name, records in zip(names, runs)}
    loc_errors = {name: np.array([records[k].get("loc_err", np.nan) for k in common]) for name, records in zip(names, runs)}

    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    summary_table(out, names, scores, loc_errors, len(common), args.truth)
    pairwise_tests(out, names, scores, args.metric)
    combining_models(out, names, scores)
    missed_items(out, names, scores, runs[0], common)

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\nsaved to {args.out}")


def summary_table(out, names, scores, loc_errors, n_items, truth):
    out(f"Items compared: {n_items}   ground truth: {truth}")
    out()
    out(f"{'model':<16}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'mAP@10':>8}{'loc median':>12}{'<=1':>6}")

    for name in names:
        s, errors = scores[name], loc_errors[name][~np.isnan(loc_errors[name])]
        out(f"{name:<16}{s['R@1'].mean():>7.3f}{s['R@5'].mean():>7.3f}{s['R@10'].mean():>7.3f}{s['mAP@10'].mean():>8.3f}"
            + (f"{np.median(errors):>12.2f}{(errors <= 1).mean():>6.2f}" if len(errors) else ""))


def pairwise_tests(out, names, scores, metric):
    out()
    out(f"Pairwise, {metric} (same items)")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            x, y = scores[a][metric], scores[b][metric]
            diff, lower, upper = bootstrap_difference(x, y)
            interval = f"difference ({b} - {a}) = {diff:+.3f}, 95% CI [{lower:+.3f}, {upper:+.3f}]"

            if metric != "mAP@10":                  # 0/1 outcomes
                both = int(((x == 1) & (y == 1)).sum())
                only_a = int(((x == 1) & (y == 0)).sum())
                only_b = int(((x == 0) & (y == 1)).sum())
                neither = int(((x == 0) & (y == 0)).sum())
                out(f"  {a} vs {b}: both right {both}, only {a} {only_a}, only {b} {only_b}, both wrong {neither}")
                out(f"      {interval}, McNemar exact p = {mcnemar_exact(only_a, only_b):.3f}")
            else:
                out(f"  {a} vs {b}: higher for {b} on {int((y > x).sum())} items, higher for {a} on {int((x > y).sum())}, equal on {int((x == y).sum())}")
                out(f"      {interval}, permutation p = {permutation_test(x, y):.3f}")


def combining_models(out, names, scores):
    out()
    out("Combining models (upper bound if a perfect selector picked the right model per item)")

    for m in ("R@1", "R@10"):
        best_single = max(scores[name][m].mean() for name in names)
        any_right = np.max([scores[name][m] for name in names], 0).mean()
        out(f"  {m}: best single {best_single:.3f}, any model right {any_right:.3f}")


def missed_items(out, names, scores, records, common):
    """Items no run finds in the top 10, and items only one run gets right at top 1. records: any run's records."""
    missed = [k for r, k in enumerate(common) if all(scores[name]["R@10"][r] == 0 for name in names)]
    out()
    out(f"Items missed by every model in the top 10: {len(missed)} of {len(common)}")

    for k in missed:
        r = records[k]
        out(f"  [{r.get('Item')}] {r.get('Hazardous materials')} | {r.get('Location')} | {r.get('Name of equipment')} / "
            f"{r.get('Component')} | query: {r.get('search_query')}")
        
    for name in names:
        only = [records[k].get("Item") for r, k in enumerate(common)
                if scores[name]["R@1"][r] == 1 and all(scores[other]["R@1"][r] == 0 for other in names if other != name)]
        out(f"Items only {name} gets right at top 1: {len(only)} {only}")
