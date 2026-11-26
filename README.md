# ViPeR

**Visual place recognition for safe ship recycling.** ViPeR locates the inspection photographs of an Inventory of Hazardous Materials (IHM) in a camera scan of a ship. Each IHM item has one or more close-up photographs and a short text, and the task is to find the scan frames that show the same place. The test vessel is the museum ship Hestmanden.

Course project in IMT4392 Deep Learning at NTNU, with AnyLoc as the core paper. The repository contains everything needed to reproduce the results in the report from the raw data.

## Results

Retrieval per item on the 64 annotated IHM items. A scan frame counts as correct if its camera is within 1.0 m and 45° of an annotated frame (relaxed ground truth). Loc. is the median distance in meters from the top-ranked frame to an annotated frame.

| Model | R@1 | R@5 | R@10 | mAP@10 | Loc. (m) |
|---|---:|---:|---:|---:|---:|
| AnyLoc (core paper) | 35.9 | 45.3 | 48.4 | 34.3 | 5.78 |
| SALAD | 51.6 | 53.1 | 60.9 | 41.1 | 0.79 |
| MegaLoc | 53.1 | 62.5 | 64.1 | 43.1 | 0.58 |
| MegaLoc, fine-tuned (mean of 3 seeds) | 53.1 | 64.6 | 66.7 | 46.3 | 0.38 |
| Text only (SigLIP2) | 9.4 | 15.6 | 20.3 | 6.6 | 7.41 |
| MegaLoc + text | 59.4 | 64.1 | 67.2 | 45.0 | 0.29 |
| Fine-tuned (seed 0) + text | 51.6 | 65.6 | 73.4 | 41.4 | 0.55 |

The trained models clearly beat AnyLoc. Fine-tuning on the scan gives no reliable gain (Recall@1 from 48.4 to 60.9 depending on the seed). The text gives a gain that is not significant, but it finds items that the image models miss. The paired tests behind these statements are in `runs_final/comparisons/`.

## Repository layout

```
main.py                 single entry point: python main.py <command> ...
run_all.sh              the full experiment, from the raw data to the comparison tables
viper/
  anyloc/               AnyLoc: DINOv2 value features and VLAD
  salad/                SALAD (torch.hub)
  megaloc/              MegaLoc (torch.hub)
  siglip/               SigLIP2 text and image embeddings, fusion with MegaLoc
  training/             fine-tuning with pairs derived from the camera poses
  utils/                data, ground truth, metrics, evaluation, statistics, and the
                        compare, summary, pose-map, pose-check,
                        dataset-stats and run-info commands
```

All code lives inside the `viper` package. The torch.hub repositories of the models put their own top-level modules (for example `utils` and `models` in SALAD) on the import path while loading, so generic top-level package names would clash with them.

## Setup

Tested with Python 3.14, PyTorch 2.14 and CUDA 13.0 on one GPU.

```bash
python -m venv .venv
source .venv/bin/activate
# install PyTorch and torchvision for your CUDA version (https://pytorch.org), then:
pip install --upgrade pip
pip install -r requirements.txt
```

The model weights are downloaded on first use: AnyLoc (DINOv2), SALAD and MegaLoc through torch.hub, SigLIP2 from Hugging Face. SALAD loads the DINOv2 ViT-B/14 weights from `dl.fbaipublicfiles.com`. If that download fails, place `dinov2_vitb14_pretrain.pth` in `~/.cache/torch/hub/checkpoints/` (can be downloaded from [www.kaggle.com](https://www.kaggle.com/datasets/devchopin/dinov2-vit-pretrain)).

## Data

The data is not part of the repository. `run_all.sh` expects this layout (other paths can be set with environment variables, see below):

```
data/
  latest/
    imgs/       scan frames, for example 00012.png (19,378 frames)
    poses/      one pose file per frame, with the same number in the file name
  IHM-dataset/
    mapped/     the IHM inspection photographs
    IHM_full_with_queries-fixed_v2.json
```

- **Poses:** each file holds a 3×4 matrix [R | t] that maps world coordinates to camera coordinates. The camera position is −Rᵀt and the viewing direction is the third row of R. The unit is meters (checked against the hull dimensions and the deck spacing, see `python main.py pose-check` and `pose-map`).
- **IHM json:** a list of items. The fields used are `image_paths` (the photographs), `gt_image_ids` (the annotated scan frames, by the number in their file name), `search_query` (the text) and the descriptive fields `Item`, `Location`, `Hazardous materials`, `Name of equipment` and `Component`.
- **Test only:** the IHM photographs are only used for testing. Training uses scan frames only.

## Reproducing the results

```bash
source .venv/bin/activate
tmux new -s final                        # the full run takes many hours
bash run_all.sh 2>&1 | tee run_all.log
```

`run_all.sh` runs the protocol that was fixed before any result was seen:

1. **Baselines:** AnyLoc, SALAD and MegaLoc without training, at radius 1.0 m and at 0.5 and 2.0 m for sensitivity.
2. **Fine-tuning:** the MegaLoc head, one epoch of 300 batches, weights at the end of training (no selection on validation), seeds 0, 1 and 2.
3. **Ablations (seed 0):** two backbone blocks unfrozen, informative pairs with keyframes, and the conventional protocol (best epoch on validation, early stopping).
4. **Pose map:** the trajectory figure and the scale check.
5. **Text:** SigLIP2 text only, and fused with MegaLoc and with the fine-tuned model (fixed weight 1.0).
6. **Comparisons and summary:** paired tests between the runs, and one table with all runs.

Everything is written to `runs_final/`. If the run stops, the same command continues where it stopped: finished steps leave a marker in `runs_final/_done/` and are skipped. Running it again on a finished folder only rewrites the comparisons and the summary. The script refuses to write into a folder that was produced with other protocol settings.

| Variable | Default | Meaning |
|---|---|---|
| `DB_IMGS` | `data/latest/imgs` | scan frames |
| `DB_POSES` | `data/latest/poses` | pose files |
| `QUERIES` | `data/IHM-dataset/mapped` | IHM photographs |
| `JSON` | `data/IHM-dataset/IHM_full_with_queries-fixed_v2.json` | IHM items |
| `OUT` | `runs_final` | output folder |
| `SEEDS` | `0 1 2` | fine-tuning seeds |
| `RADII_EXTRA` | `0.5 2.0` | sensitivity radii |
| `TRAIN_EPOCHS` | `1` | epochs of the main fine-tuning |
| `BEST_EPOCHS` | `10` | maximum epochs of the best-epoch variant |
| `TEXT_WEIGHT` | `1.0` | weight of the text score in the fusion |
| `PY` | `python` | Python interpreter |

Main results in `runs_final/`:

| Path | Content |
|---|---|
| `summary.csv` | one row per run and radius: Recall@1/5/10 with interval, mAP@10, localization error |
| `comparisons/*.txt` | paired tests: McNemar for Recall@K, sign-flip permutation test for mAP@10 |
| `run_*/` | one folder per scored model (see "Output of a run") |
| `train_*/` | fine-tuned weights (`last.pth`, `best.pth`) and `train_log.json` |
| `siglip_megaloc/`, `siglip_ft/` | text only, photograph only, and fused runs |
| `pose_map/` | `pose_map.png` and `pose_stats.json` |
| `environment.txt` | Python, GPU, package versions, data counts and a hash of every source file |
| `logs/` | the console output of every step |

## Commands

Every command prints its options with `python main.py <command> --help`.

| Command | What it does |
|---|---|
| `anyloc` | AnyLoc: DINOv2 ViT-G/14 value features (layer 31) aggregated with VLAD (32 clusters), no training |
| `salad` | SALAD, pretrained, optionally with fine-tuned weights (`--weights`) |
| `megaloc` | MegaLoc, pretrained, optionally with fine-tuned weights (`--weights`) |
| `train` | fine-tunes MegaLoc (or SALAD) on the scan with pairs from the camera poses |
| `siglip` | SigLIP2: item text only, photograph only, and MegaLoc + text |
| `compare` | paired comparison of finished runs |
| `summary` | the summary table of a `run_all.sh` folder |
| `pose-map` | the camera trajectory by deck, and the scale check of the poses |
| `pose-check` | sanity check of the pose files: rotation, world-to-camera convention, scale and deck levels |
| `dataset-stats` | statistics of the scan, the IHM items and photographs, and the ground truth |
| `run-info` | environment, run time, disk space and training statistics of a finished `run_all.sh` folder |

Examples:

```bash
DATA="--db-imgs data/latest/imgs --db-poses data/latest/poses \
      --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json"

python main.py megaloc $DATA --out runs/megaloc
python main.py train --db-imgs data/latest/imgs --db-poses data/latest/poses \
    --baseline-run runs/megaloc --out runs/train_s0 --epochs 1 --seed 0
python main.py megaloc $DATA --weights runs/train_s0/last.pth --out runs/megaloc_ft_s0
python main.py siglip $DATA --megaloc-run runs/megaloc --out runs/siglip
python main.py compare runs/megaloc runs/megaloc_ft_s0 runs/siglip/megaloc_plus_text --out runs/comparison.txt
```

Add `--dry-run` to `train` for a short check that the setup works before a full training run.

## Output of a run

Every scoring command (`anyloc`, `salad`, `megaloc`, `siglip`) writes the same files:

| File | Content |
|---|---|
| `results.json` | all metrics with 95% bootstrap intervals, per photograph and per item, strict and relaxed ground truth |
| `per_item.json` | one record per item, used by `compare` for paired tests |
| `topk.json` | the top-ranked scan frames of every photograph |
| `failures.json`, `failures_per_item.json` | the photographs and items whose top frame is wrong |
| `montage/*.jpg` | photograph (blue) and top frames, green if correct and red if wrong |
| `db_desc.npy`, `db_names.json`, `db_done.txt` | cached database descriptors |

The database descriptors are cached, so a second run with the same `--out` (for example at another radius) only computes the photographs. An interrupted run continues where it stopped.

## Evaluation

- **Strict ground truth:** the scan frames annotated for an item (median 2 per item).
- **Relaxed ground truth:** also every frame whose camera is within `--pos-radius` meters (default 1.0) and `--pos-angle` degrees (default 45) of an annotated frame (median 116 per item). The scan is dense, so many neighbouring frames show the same place.
- **Per item:** an item is scored with the best similarity over its photographs. It counts as found if any of its photographs finds a correct frame.
- **Metrics:** Recall@1/5/10, mAP@10 and MRR@10, with 95% bootstrap intervals over items. The localization error is the distance from the top-ranked frame to the nearest annotated frame.
- **Comparisons:** models are compared on the same items, with an exact McNemar test for Recall@K and a paired sign-flip permutation test for mAP@10.

## Reproducibility

- **Random seeds:** every random choice uses a fixed seed (vocabulary sampling, bootstrap, training, validation queries).
- **Fixed protocol:** the protocol in `run_all.sh` was fixed before the results were seen. Fine-tuned models are scored with the weights of the last epoch, and the text weight is never tuned on the IHM items.
- **Traceability:** `environment.txt` records the package versions and a hash of every source file. `comparisons/versions.txt` does the same for the comparison tables.
- **GPU:** computations are not bit-exact across GPUs and drivers, so a rerun can differ in the last digits of a similarity score. Training is also affected by this, since small differences can grow over the batches.

## References

- AnyLoc: Keetha et al., *AnyLoc: Towards Universal Visual Place Recognition*, IEEE RA-L 2023. https://github.com/AnyLoc/AnyLoc
- SALAD: Izquierdo and Civera, *Optimal Transport Aggregation for Visual Place Recognition*, CVPR 2024. https://github.com/serizba/salad
- MegaLoc: Berton and Masone, *MegaLoc: One Retrieval to Place Them All*, CVPR Workshops 2025. https://github.com/gmberton/MegaLoc
- SigLIP 2: Tschannen et al., *SigLIP 2: Multilingual Vision-Language Encoders*, 2025. https://huggingface.co/google/siglip2-so400m-patch14-384
- DINOv2: Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, 2023. https://github.com/facebookresearch/dinov2
- Multi-similarity loss: Wang et al., *Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning*, CVPR 2019.
