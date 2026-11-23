"""Retrieval metrics, bootstrap intervals and localization error."""
import numpy as np

RECALL_KS = (1, 5, 10)
DEPTH = 10                     # mAP and MRR are computed over the top 10


def rank(similarity, depth=DEPTH):
    """Database indices sorted by similarity, best first, for every row. Shape (rows, depth)."""
    return np.argsort(-similarity, axis=1)[:, :depth]


def retrieval_metrics(ranking, ground_truth):
    """Per-row scores for a ranking against ground truth (one array of database indices per row).

    Returns a dict of arrays with one value per row: R@1, R@5, R@10, mAP@10 and MRR@10.
    """
    scores = {f"R@{k}": [] for k in RECALL_KS}
    scores["mAP@10"] = []
    scores["MRR@10"] = []

    for row, correct in zip(ranking, ground_truth):
        hit = np.isin(row, correct)
        for k in RECALL_KS:
            scores[f"R@{k}"].append(float(hit[:k].any()))

        precision = np.cumsum(hit) / (np.arange(len(hit)) + 1)
        scores["mAP@10"].append(float((precision * hit)[:DEPTH].sum() / min(DEPTH, len(correct))))
        scores["MRR@10"].append(float(1 / (np.argmax(hit[:DEPTH]) + 1)) if hit[:DEPTH].any() else 0.0)

    return {name: np.array(values) for name, values in scores.items()}


def bootstrap_summary(scores, n_boot=2000, seed=0):
    """Mean of every metric with a 95% bootstrap interval over rows. All metrics share the same resamples."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(scores.values())))
    resamples = rng.integers(0, n, (n_boot, n))
    summary = {}

    for name, values in scores.items():
        means = values[resamples].mean(1)
        summary[name] = round(float(values.mean()), 4)
        summary[name + "_ci95"] = [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

    summary["n"] = n

    return summary


def localization_errors(top1, ground_truth, centres):
    """Distance from the camera of each top-1 image to the nearest ground-truth camera.

    Rows without a pose on either side are left out.
    """
    errors = []

    for retrieved, correct in zip(top1, ground_truth):
        error = distance_to_nearest(centres[retrieved], centres[correct])
        if error is not None:
            errors.append(error)

    return np.array(errors)


def distance_to_nearest(point, others):
    """Smallest distance from point to any row of others, ignoring NaN rows. None if nothing is left."""
    others = others[~np.isnan(others).any(1)]
    if not len(others) or np.isnan(point).any():
        return None
    
    return np.linalg.norm(others - point, axis=1).min()


def localization_summary(errors):
    """Median error and the share of rows within 1, 2 and 5 meters."""
    if not len(errors):
        return {}
    
    return {
        "loc_err_median": round(float(np.median(errors)), 3),
        "within_1": round(float((errors <= 1).mean()), 4),
        "within_2": round(float((errors <= 2).mean()), 4),
        "within_5": round(float((errors <= 5).mean()), 4),
        "n_with_pose": len(errors),
    }
