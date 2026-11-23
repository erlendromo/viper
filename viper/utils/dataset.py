"""The evaluation data: scan images (database), IHM photographs (queries) and their ground truth.

Database   every scan image below --db-imgs, identified by the number in its file name
Queries    the photographs listed under "image_paths" for each item in the IHM json
Ground     strict:  the scan images listed under "gt_image_ids" for the item
truth      relaxed: also every scan image whose camera is within --pos-radius meters and --pos-angle
                    degrees of a strict one (needs poses)
"""
import json
import os
from dataclasses import dataclass

import numpy as np

from .files import file_id, list_images
from .poses import load_poses, viewing_angle

ITEM_FIELDS = ("Item", "Hazardous materials", "Location", "Name of equipment", "Component", "Sample No.", "search_query")


@dataclass
class Dataset:
    db_images: list            # paths of the scan images
    db_ids: list               # frame id of each scan image
    db_centres: object         # (N, 3) camera centres, NaN without pose, or None if no poses were given
    db_forwards: object        # (N, 3) viewing directions, same layout as db_centres
    items: list                # the IHM items (dicts from the json)
    query_images: list         # paths of the IHM photographs
    query_item: np.ndarray     # index into items for every photograph
    gt_strict: dict            # item index -> array of database indices
    gt_relaxed: object         # item index -> array of database indices, or None without poses
    pos_radius: float
    pos_angle: float
    gt_ids_total: int          # number of gt_image_ids in the json
    gt_ids_missing: list       # gt_image_ids that are not in the database

    def item_info(self, item):
        """The descriptive fields of an item, copied into the output files."""
        return {field: self.items[item].get(field) for field in ITEM_FIELDS}


def load_dataset(db_imgs, queries, json_path, db_poses=None, pos_radius=1.0, pos_angle=45.0):
    db_images, db_ids = _load_database(db_imgs)
    db_centres = db_forwards = None

    if db_poses:
        db_centres, db_forwards = load_poses(db_poses, db_ids)
        n_posed = int((~np.isnan(db_centres).any(1)).sum())
        print(f"[db] poses found for {n_posed}/{len(db_images)} images")

    items = load_items(json_path)
    query_images, query_item = _find_query_images(items, queries, json_path)
    gt_strict, gt_ids_total, gt_ids_missing = _strict_ground_truth(items, db_ids)
    print(f"[queries] {len(query_images)} images from {len(items)} items")
    print(f"[gt] {gt_ids_total - len(gt_ids_missing)}/{gt_ids_total} ground-truth ids found in the database"
          + (f"; not found e.g. {sorted(set(gt_ids_missing))[:8]}" if gt_ids_missing else ""))
    
    no_gt = [items[k].get("Item", k) for k in range(len(items)) if not len(gt_strict[k])]
    if no_gt:
        print(f"[gt] {len(no_gt)} items have no usable ground truth and are skipped: {no_gt[:10]}")

    if not query_images:
        raise SystemExit("no query images found")

    gt_relaxed = None
    if db_centres is not None:
        gt_relaxed = relaxed_ground_truth(gt_strict, db_centres, db_forwards, pos_radius, pos_angle)
        sizes = [len(gt_relaxed[k]) for k in gt_strict if len(gt_strict[k])]
        strict_sizes = [len(v) for v in gt_strict.values() if len(v)]
        print(f"[gt] relaxed ground truth (within {pos_radius} m and {pos_angle} deg of a GT image): "
              f"median {int(np.median(sizes))} correct images per item (strict: {int(np.median(strict_sizes))})")

    return Dataset(db_images, db_ids, db_centres, db_forwards, items, query_images, query_item,
                   gt_strict, gt_relaxed, pos_radius, pos_angle, gt_ids_total, gt_ids_missing)


def relaxed_ground_truth(gt_strict, centres, forwards, radius, max_angle):
    """Add every scan image that looks at the same spot as one of the strict ground-truth images."""
    from scipy.spatial import cKDTree

    has_pose = ~np.isnan(centres).any(1)
    posed = np.where(has_pose)[0]
    tree = cKDTree(centres[has_pose])
    relaxed = {}

    for item, strict in gt_strict.items():
        correct = set(strict.tolist())
        for anchor in strict:
            if not has_pose[anchor]:
                continue

            for candidate in posed[tree.query_ball_point(centres[anchor], radius)]:
                if viewing_angle(forwards[candidate], forwards[anchor]) <= max_angle:
                    correct.add(int(candidate))

        relaxed[item] = np.array(sorted(correct), int)

    return relaxed


# ------------------------------------------------------------------------------------------------ helpers

def _load_database(db_imgs):
    images = [path for path in list_images(db_imgs) if file_id(path) is not None]
    ids = [file_id(path) for path in images]

    if len(set(ids)) != len(ids):
        raise SystemExit(f"{len(ids) - len(set(ids))} database images share a file-name number; ids would be ambiguous")
    
    print(f"[db] {len(images)} images, ids {min(ids)}..{max(ids)}")

    return images, ids


def load_items(json_path):
    """The list of IHM items from the json (a list, or a dict with an "items" list)."""
    with open(json_path) as fh:
        items = json.load(fh)

    if isinstance(items, dict):
        items = items.get("items", list(items.values()))

    return items


def _find_query_images(items, queries, json_path):
    """Resolve every path in "image_paths" to a file, relative to the query folder or the json.

    If a path does not exist, the file is looked up by name anywhere below those two folders,
    first with the same extension and then with any extension.
    """
    json_dir = os.path.dirname(os.path.abspath(json_path))
    pool = list_images(queries) + list_images(json_dir)
    by_name = {os.path.basename(p).lower(): p for p in reversed(pool)}                         # first match wins
    by_stem = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in reversed(pool)}

    images, owner, n_missing, n_searched = [], [], 0, 0
    for item, entry in enumerate(items):
        for listed in entry.get("image_paths", []):
            rel = listed.replace("\\", "/")
            candidates = (os.path.join(queries, rel), os.path.join(os.path.dirname(json_path), rel), rel)
            path = next((c for c in candidates if os.path.isfile(c)), None)

            if path is None:
                name = os.path.basename(rel).lower()
                path = by_name.get(name) or by_stem.get(os.path.splitext(name)[0])
                n_searched += path is not None

            if path is None:
                n_missing += 1
                if n_missing <= 5:
                    print("  missing query image:", listed)
                continue

            images.append(path)
            owner.append(item)

    if n_searched:
        print(f"[queries] {n_searched} images were located by file name in a subfolder or with another extension")

    if n_missing:
        in_folder = list_images(queries)
        print(f"[queries] {n_missing} listed images not found. The query folder holds {len(in_folder)} images, e.g. "
              f"{[os.path.relpath(p, queries) for p in in_folder[:3]]}")
        
    return images, np.array(owner)


def _strict_ground_truth(items, db_ids):
    """Map the gt_image_ids of every item to database indices."""
    index_of = {frame: row for row, frame in enumerate(db_ids)}
    gt, total, missing = {}, 0, []

    for item, entry in enumerate(items):
        rows = []
        for frame in entry.get("gt_image_ids", []):
            total += 1
            row = index_of.get(int(frame))

            if row is None:
                missing.append(int(frame))
            else:
                rows.append(row)

        gt[item] = np.unique(np.array(rows, int))
        
    return gt, total, missing
