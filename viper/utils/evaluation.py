"""Score a similarity matrix against the ground truth and write the result files.

Scores are computed per photograph and per item. An item is scored with the best similarity over its
photographs, so it counts as found if any of its photographs finds a correct scan image.

Files written to the output folder
    results.json              all metrics with 95% bootstrap intervals, strict and relaxed ground truth
    per_item.json             one record per item, used for paired comparisons between models
    topk.json                 the top-k scan images of every photograph
    failures.json             the photographs whose top-1 image is wrong
    failures_per_item.json    the items whose top-1 image is wrong
    montage/*.jpg             query photograph and top-k scan images (unless montages=False)

Montages, topk.json and the failure lists use the relaxed ground truth when poses are available.
"""
import json
import os

import numpy as np

from .metrics import (DEPTH, bootstrap_summary, distance_to_nearest, localization_errors,
                      localization_summary, rank, retrieval_metrics)
from .montage import save_montage


def evaluate(sim, data, out_dir, info, top_k=5, montages=True):
    """sim: (photographs, database) similarity matrix. info: run details written at the top of results.json."""
    os.makedirs(out_dir, exist_ok=True)
    ids, centres = data.db_ids, data.db_centres

    # photographs and items that have ground truth
    photos = [i for i in range(len(data.query_images)) if len(data.gt_strict[data.query_item[i]])]
    items = [k for k in range(len(data.items)) if len(data.gt_strict[k]) and (data.query_item == k).any()]
    photo_ranking = rank(sim[photos], max(DEPTH, top_k))
    item_sim = np.stack([sim[data.query_item == k].max(0) for k in items])
    item_ranking = rank(item_sim, max(DEPTH, top_k))

    strict_photo = [data.gt_strict[data.query_item[i]] for i in photos]
    strict_item = [data.gt_strict[k] for k in items]
    scores = {"per_image": retrieval_metrics(photo_ranking, strict_photo),
              "per_item": retrieval_metrics(item_ranking, strict_item)}

    results = dict(info, n_database=len(data.db_images),
                   gt_ids_found=data.gt_ids_total - len(data.gt_ids_missing), gt_ids_total=data.gt_ids_total)
    for level, ranking, truth in (("per_image", photo_ranking, strict_photo), ("per_item", item_ranking, strict_item)):
        results[level] = bootstrap_summary(scores[level])
        if centres is not None:
            results[level].update(localization_summary(localization_errors(ranking[:, 0], truth, centres)))

    relaxed_item_scores = None
    if data.gt_relaxed is not None:
        relaxed_photo = [data.gt_relaxed[data.query_item[i]] for i in photos]
        relaxed_item = [data.gt_relaxed[k] for k in items]
        relaxed_item_scores = retrieval_metrics(item_ranking, relaxed_item)
        results["relaxed"] = {
            "pos_radius": data.pos_radius,
            "pos_angle": data.pos_angle,
            "per_image": bootstrap_summary(retrieval_metrics(photo_ranking, relaxed_photo)),
            "per_item": bootstrap_summary(relaxed_item_scores),
        }

    _write_json(results, out_dir, "results.json", indent=1)

    _write_per_item(data, items, item_ranking, scores["per_item"], relaxed_item_scores, out_dir)

    # from here on: relaxed ground truth when available
    truth = data.gt_relaxed if data.gt_relaxed is not None else data.gt_strict
    item_scores = relaxed_item_scores if relaxed_item_scores is not None else scores["per_item"]
    _write_topk_and_failures(data, photos, photo_ranking, sim, truth, top_k, montages, out_dir)
    failed_items = [
        dict(data.item_info(k), gt_ids=[ids[j] for j in truth[k]][:30], top_ids=[ids[j] for j in item_ranking[r, :top_k]])
        for r, k in enumerate(items) if item_scores["R@1"][r] == 0
    ]

    _write_json(failed_items, out_dir, "failures_per_item.json", indent=1)

    print_results(results)
    return results


def print_results(results):
    def line(summary):
        return "  ".join(f"{k} {summary[k]:.3f} [{summary[k + '_ci95'][0]:.2f}-{summary[k + '_ci95'][1]:.2f}]"
                         for k in ("R@1", "R@5", "R@10", "mAP@10"))

    print(f"strict  per image (n={results['per_image']['n']}): {line(results['per_image'])}")
    print(f"strict  per item  (n={results['per_item']['n']}): {line(results['per_item'])}")
    if "relaxed" in results:
        print(f"relaxed per image: {line(results['relaxed']['per_image'])}")
        print(f"relaxed per item : {line(results['relaxed']['per_item'])}")

    item = results["per_item"]
    if "loc_err_median" in item:
        print(f"localization error per item: median {item['loc_err_median']} m, within 1/2/5 m: "
              f"{item['within_1']:.2f}/{item['within_2']:.2f}/{item['within_5']:.2f}")


# ------------------------------------------------------------------------------------------------ helpers

def _write_per_item(data, items, item_ranking, strict_scores, relaxed_scores, out_dir):
    """One record per item with its scores under both ground truths, for compare_runs."""
    records = []
    for r, k in enumerate(items):
        top1 = item_ranking[r, 0]
        record = dict(index=k, n_images=int((data.query_item == k).sum()), top1_id=data.db_ids[top1], **data.item_info(k))
        record["strict"] = {m: float(strict_scores[m][r]) for m in strict_scores}

        if relaxed_scores is not None:
            record["relaxed"] = {m: float(relaxed_scores[m][r]) for m in relaxed_scores}

        if data.db_centres is not None:
            error = distance_to_nearest(data.db_centres[top1], data.db_centres[data.gt_strict[k]])
            if error is not None:
                record["loc_err"] = round(float(error), 3)

        records.append(record)

    _write_json(records, out_dir, "per_item.json", indent=1)


def _write_topk_and_failures(data, photos, photo_ranking, sim, truth, top_k, montages, out_dir):
    ids, images = data.db_ids, data.db_images
    if montages:
        os.makedirs(os.path.join(out_dir, "montage"), exist_ok=True)

    topk, failures = [], []
    for r, i in enumerate(photos):
        correct = truth[data.query_item[i]]
        top = photo_ranking[r, :top_k]
        hit = np.isin(top, correct)
        record = dict(query=data.query_images[i], item=data.items[data.query_item[i]].get("Item"),
                      gt_ids=[ids[j] for j in correct][:30],
                      top=[dict(id=ids[j], img=images[j], score=round(float(sim[i, j]), 4), correct=bool(ok))
                           for j, ok in zip(top, hit)])
        topk.append(record)

        if not hit[0]:
            failures.append(record)

        if montages:
            name = f"{'ok' if hit[0] else 'FAIL'}_{r:04d}_{os.path.basename(data.query_images[i])}.jpg"
            save_montage(data.query_images[i], [images[j] for j in top], hit, os.path.join(out_dir, "montage", name))
            
    _write_json(topk, out_dir, "topk.json", indent=1)
    _write_json(failures, out_dir, "failures.json", indent=1)


def _write_json(obj, out_dir, name, indent=None):
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(obj, fh, indent=indent)
