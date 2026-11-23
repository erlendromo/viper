"""Item prompts and the fusion of image and text scores.

The image score (MegaLoc) and the text score (SigLIP2) have different scales, so both are standardized per
photograph (zero mean and unit variance over all scan images) before they are added:

    fused = z(image) + weight * z(text)
"""
import numpy as np

from viper.utils.metrics import rank, retrieval_metrics

TEMPLATE = "This is a photo of {}."
SWEEP_WEIGHTS = (0, 0.25, 0.5, 1, 2, 4)


def item_prompts(items, template):
    """One prompt per item from its search_query (or the equipment name if there is none)."""
    return [template.format(str(item.get("search_query") or item.get("Name of equipment") or "").strip()) for item in items]


def standardize_rows(sim):
    """Zero mean and unit variance per row (one row per photograph)."""
    return (sim - sim.mean(1, keepdims=True)) / (sim.std(1, keepdims=True) + 1e-8)


def item_recall(sim, data):
    """Recall@1 and @10 per item (relaxed ground truth if available). Used for the printed sweep only."""
    truth = data.gt_relaxed or data.gt_strict
    items = [k for k in range(len(data.items)) if len(data.gt_strict[k]) and (data.query_item == k).any()]
    item_sim = np.stack([sim[data.query_item == k].max(0) for k in items])
    scores = retrieval_metrics(rank(item_sim), [truth[k] for k in items])

    return scores["R@1"].mean(), scores["R@10"].mean()


def print_weight_sweep(image_z, text_z, data):
    """Recall for other text weights. For analysis only: the reported weight is fixed and never tuned."""
    print("\nweight sweep, per item (analysis only, do not pick the weight from this):")
    print(f"  {'weight':>8}{'R@1':>8}{'R@10':>8}")

    for weight in SWEEP_WEIGHTS:
        r1, r10 = item_recall(image_z + weight * text_z, data)
        print(f"  {weight:>8}{r1:>8.3f}{r10:>8.3f}" + ("   <- MegaLoc alone" if weight == 0 else ""))
        
    r1, r10 = item_recall(text_z, data)
    print(f"  {'text':>8}{r1:>8.3f}{r10:>8.3f}")
