"""Validation: close-up crops of held-out scan frames, retrieved against every Nth scan frame.

The IHM photographs are not used, so the validation score can be used during training. It only
approximates the real task: in the experiments it did not predict the results on the IHM items.
"""
from dataclasses import dataclass

import numpy as np

from viper.utils.metrics import rank, retrieval_metrics

from .augment import to_tensor, validation_closeup
from .pairs import angles_deg


@dataclass
class ValidationSet:
    queries: np.ndarray        # frame indices of the held-out frames used as close-up queries
    database: np.ndarray       # frame indices of the database (every db_stride-th frame of the scan)
    ground_truth: list         # per query: positions in `database` that show the same place


def build_validation_set(pos, fwd, validation, n_queries, db_stride, radius, max_angle, seed):
    rng = np.random.default_rng(seed)
    queries = np.sort(rng.choice(np.where(validation)[0], min(n_queries, int(validation.sum())), replace=False))
    database = np.arange(0, len(pos), db_stride)

    ground_truth = []
    for q in queries:
        same_place = ((np.linalg.norm(pos[database] - pos[q], axis=1) <= radius)
                      & (angles_deg(fwd[q:q + 1], fwd[database])[0] <= max_angle) & (database != q))
        ground_truth.append(np.where(same_place)[0])
    
    usable = [i for i, g in enumerate(ground_truth) if len(g)]

    return ValidationSet(queries[usable], database, [ground_truth[i] for i in usable])


def embed_paths(model, paths, size, device, batch_size, transform=None):
    """L2-normalized descriptors for image files. transform(i, PIL image) -> PIL image is optional.

    Images are resized with PIL bilinear interpolation (the evaluation commands use OpenCV area
    interpolation). This only affects the validation scores.
    """
    import torch
    from PIL import Image

    descriptors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = []
            for i, path in enumerate(paths[start:start + batch_size], start=start):
                image = Image.open(path).convert("RGB")
                if transform is not None:
                    image = transform(i, image)
                
                batch.append(to_tensor(image.resize((size[1], size[0]), Image.BILINEAR)))

            out = model(torch.stack(batch).to(device))
            descriptors.append(torch.nn.functional.normalize(out.float(), dim=1).cpu().numpy())
        
    return np.concatenate(descriptors)


def validate(model, images, val_set, size, device, batch_size, seed, area):
    """Recall@1, 5 and 10 of the close-up queries against the validation database."""
    db_desc = embed_paths(model, [images[i] for i in val_set.database], size, device, batch_size)
    query_desc = embed_paths(model, [images[i] for i in val_set.queries], size, device, batch_size,
                             validation_closeup(seed, area))
    scores = retrieval_metrics(rank(query_desc @ db_desc.T), val_set.ground_truth)
    
    return {k: float(scores[k].mean()) for k in ("R@1", "R@5", "R@10")}
