"""Compare several runs on the same items (paired).

  python compare_runs.py run_anyloc run_megaloc run_salad [--truth relaxed] [--out comparison.txt]

Each run folder must contain per_item.json, which anyloc_full.py / baselines_full.py write (re-run them once with the
current scripts if the file is missing; descriptors are cached, so that is quick).
Prints: a summary table, pairwise paired tests on Recall@1, which items every model misses, and the best possible
result if the models were combined.
"""
import argparse, json, os, sys
from math import comb
import numpy as np

def load(run):
    f = os.path.join(run, "per_item.json")
    if not os.path.exists(f): sys.exit(f"{f} not found: re-run that model with the current scripts (cached, so quick)")
    return {r["index"]: r for r in json.load(open(f))}

def mcnemar_exact(b, c):
    """two-sided exact test on the discordant pairs (b = only A right, c = only B right)"""
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)

def perm_test(x, y, n_perm=20000, seed=0):
    """two-sided paired sign-flip permutation test on the mean difference (for metrics that are not 0/1)"""
    d = y - x; obs = abs(d.mean())
    if obs == 0: return 1.0
    rng = np.random.default_rng(seed); signs = rng.choice([-1.0, 1.0], (n_perm, len(d)))
    return float(((np.abs((signs * d).mean(1)) >= obs - 1e-12).sum() + 1) / (n_perm + 1))

def boot_diff(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (n_boot, len(x)))
    d = (y[idx] - x[idx]).mean(1)
    return float((y - x).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("runs", nargs="+")
    ap.add_argument("--truth", default="relaxed", choices=["relaxed", "strict"]); ap.add_argument("--out", default="comparison.txt")
    ap.add_argument("--metric", default="R@1", choices=["R@1", "R@5", "R@10", "mAP@10"],
                    help="metric for the pairwise tests. Recall metrics use an exact McNemar test; mAP@10 uses a paired sign-flip permutation test")
    a = ap.parse_args()
    names = [os.path.basename(os.path.normpath(r)) for r in a.runs]
    data = [load(r) for r in a.runs]
    for n, d in zip(names, data):
        if a.truth not in next(iter(d.values())): sys.exit(f"{n} has no '{a.truth}' results (was it run without poses?)")
    common = sorted(set.intersection(*[set(d) for d in data]))
    if any(len(d) != len(common) for d in data):
        print(f"note: runs cover different items; comparing the {len(common)} they share")
    lines = []; P = lambda t="": (print(t), lines.append(t))
    M = {n: {m: np.array([d[k][a.truth][m] for k in common]) for m in ("R@1", "R@5", "R@10", "mAP@10")} for n, d in zip(names, data)}
    loc = {n: np.array([d[k].get("loc_err", np.nan) for k in common]) for n, d in zip(names, data)}

    P(f"Items compared: {len(common)}   ground truth: {a.truth}")
    P(); P(f"{'model':<16}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'mAP@10':>8}{'loc median':>12}{'<=1':>6}")
    for n in names:
        l = loc[n][~np.isnan(loc[n])]
        P(f"{n:<16}{M[n]['R@1'].mean():>7.3f}{M[n]['R@5'].mean():>7.3f}{M[n]['R@10'].mean():>7.3f}{M[n]['mAP@10'].mean():>8.3f}"
          + (f"{np.median(l):>12.2f}{(l <= 1).mean():>6.2f}" if len(l) else ""))

    met = a.metric; binary = met != "mAP@10"
    P(); P(f"Pairwise, {met} (same items)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = M[names[i]][met], M[names[j]][met]
            d, lo, hi = boot_diff(x, y)
            if binary:
                both = int(((x == 1) & (y == 1)).sum()); onlyx = int(((x == 1) & (y == 0)).sum())
                onlyy = int(((x == 0) & (y == 1)).sum()); none = int(((x == 0) & (y == 0)).sum())
                P(f"  {names[i]} vs {names[j]}: both right {both}, only {names[i]} {onlyx}, only {names[j]} {onlyy}, both wrong {none}")
                P(f"      difference ({names[j]} - {names[i]}) = {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], McNemar exact p = {mcnemar_exact(onlyx, onlyy):.3f}")
            else:
                P(f"  {names[i]} vs {names[j]}: higher for {names[j]} on {int((y > x).sum())} items, higher for {names[i]} on {int((x > y).sum())}, equal on {int((x == y).sum())}")
                P(f"      difference ({names[j]} - {names[i]}) = {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], permutation p = {perm_test(x, y):.3f}")

    P(); P("Combining models (upper bound if a perfect selector picked the right model per item)")
    for m in ("R@1", "R@10"):
        P(f"  {m}: best single {max(M[n][m].mean() for n in names):.3f}, any model right {np.max([M[n][m] for n in names], 0).mean():.3f}")

    miss = [k for r, k in enumerate(common) if all(M[n]["R@10"][r] == 0 for n in names)]
    P(); P(f"Items missed by every model in the top 10: {len(miss)} of {len(common)}")
    for k in miss:
        r = data[0][k]
        P(f"  [{r.get('Item')}] {r.get('Hazardous materials')} | {r.get('Location')} | {r.get('Name of equipment')} / {r.get('Component')} | query: {r.get('search_query')}")
    for n in names:
        only = [data[0][k].get("Item") for r, k in enumerate(common) if M[n]["R@1"][r] == 1 and all(M[o]["R@1"][r] == 0 for o in names if o != n)]
        P(f"Items only {n} gets right at top 1: {len(only)} {only}")
    open(a.out, "w").write("\n".join(lines) + "\n"); print(f"\nsaved to {a.out}")

if __name__ == "__main__":
    main()
