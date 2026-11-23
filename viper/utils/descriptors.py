"""Descriptor cache on disk and the similarity matrix.

Database descriptors take hours for the full scan, so they are written to <out>/<tag>_desc.npy as they
are computed (float16), together with <tag>_names.json (which images, in which order) and <tag>_done.txt
(how many are finished). A run that is interrupted resumes where it stopped. A cache whose image list
differs from the current one is recomputed from the start.
"""
import json
import os
import time

import numpy as np


def cached_descriptors(encode, paths, out_dir, tag="db", batch_size=8, save_every=500):
    """Descriptors for all paths, computed with encode(list of paths) -> (B, D) array and cached on disk."""
    os.makedirs(out_dir, exist_ok=True)
    desc_file, done_file, names_file = (os.path.join(out_dir, f"{tag}_{x}") for x in ("desc.npy", "done.txt", "names.json"))
    names = [os.path.basename(p) for p in paths]

    done, cache = 0, None
    if all(os.path.exists(f) for f in (desc_file, done_file, names_file)) and _read_json(names_file) == names:
        with open(done_file) as fh:
            done = int(fh.read().strip() or 0)

        cache = np.lib.format.open_memmap(desc_file, mode="r+")

    if done >= len(paths):
        print(f"[{tag}] using cached descriptors", flush=True)
        return cache

    print(f"[{tag}] computing descriptors {done}..{len(paths)}", flush=True)
    start = time.time()

    for first in range(done, len(paths), batch_size):
        batch = encode(paths[first:first + batch_size])
        if cache is None:
            cache = np.lib.format.open_memmap(desc_file, mode="w+", dtype=np.float16, shape=(len(paths), batch.shape[1]))
            with open(names_file, "w") as fh:
                json.dump(names, fh)

        cache[first:first + len(batch)] = batch
        finished = first + len(batch)

        if finished // save_every > first // save_every or finished == len(paths):
            cache.flush()
            with open(done_file, "w") as fh:
                fh.write(str(finished))

            print(f"  {finished}/{len(paths)} images, {time.time() - start:.0f} s", flush=True)

    return cache


def encode_in_batches(encode, paths, batch_size):
    """Descriptors for all paths without caching (used for the query photographs)."""
    return np.concatenate([encode(paths[start:start + batch_size]) for start in range(0, len(paths), batch_size)])


def similarity(query_desc, db_desc, chunk=2000):
    """Cosine similarity (descriptors are L2-normalized) as a (queries, database) float32 matrix.

    The database is read in chunks, so a float16 memmap never has to be loaded into memory at once.
    """
    sim = np.empty((len(query_desc), len(db_desc)), np.float32)
    for start in range(0, len(db_desc), chunk):
        sim[:, start:start + chunk] = query_desc @ np.asarray(db_desc[start:start + chunk], np.float32).T
        
    return sim


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)
