#!/usr/bin/env bash
# Full ViPeR experiment from a clean slate: baselines, text fusion, fine-tuning, scoring, comparisons.
#
#   cd ~/viper && source .venv/bin/activate
#   tmux new -s final
#   bash src/run_all.sh 2>&1 | tee run_all.log
#
# Re-running it on a finished $OUT skips every model step and only rewrites the comparisons and the summary.
# Everything is written under $OUT (default: runs_final). The script refuses to start if $OUT already holds
# results from another protocol. If it stops part-way, run the same command again: finished steps are skipped
# (each step leaves a .done marker in $OUT/_done). To redo one step, delete its marker and its output folder.
#
# Protocol (fixed here, before any result is seen):
#   - Ground truth: relaxed, radius 1.0, angle 45 deg (main); radius 0.5 and 2.0 as sensitivity.
#   - Fine-tuning: MegaLoc, head only, exactly TRAIN_EPOCHS epoch(s), weights of the LAST epoch (no selection
#     on validation), repeated for every seed in SEEDS. Seed 0 of the main configuration is the model used
#     for the text fusion.
#   - Ablations (seed 0, same budget): last two backbone blocks unfrozen; informative pairs + keyframe thinning.
#   - Comparison variant (seed 0): head only, up to BEST_EPOCHS epochs with early stopping, keeping the epoch with
#     the best validation score (mean of R@1/R@5/R@10). This is the conventional "best weights" protocol.
#   - Text fusion: fixed weight 1.0, never tuned.
set -euo pipefail

# ----------------------------------------------------------------------------- configuration
SRC=${SRC:-src}
PY=${PY:-python}
DB_IMGS=${DB_IMGS:-data/latest/imgs}
DB_POSES=${DB_POSES:-data/latest/poses}
QUERIES=${QUERIES:-data/IHM-dataset/mapped}
JSON=${JSON:-data/IHM-dataset/IHM_full_with_queries-fixed_v2.json}
OUT=${OUT:-runs_final}
RADII_EXTRA=${RADII_EXTRA:-"0.5 2.0"}      # sensitivity radii; the main radius 1.0 always runs last
SEEDS=${SEEDS:-"0 1 2"}                   # fine-tuning seeds for the main configuration
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
TEXT_WEIGHT=${TEXT_WEIGHT:-1.0}
BEST_EPOCHS=${BEST_EPOCHS:-10}             # max epochs of the extra "best epoch on validation" variant

DB="--db-imgs $DB_IMGS --db-poses $DB_POSES"
IHM="--queries $QUERIES --json $JSON"
PROTOCOL="epochs=$TRAIN_EPOCHS seeds=$SEEDS radii=$RADII_EXTRA 1.0 text_weight=$TEXT_WEIGHT weights=last.pth best_variant_epochs=$BEST_EPOCHS"

# ----------------------------------------------------------------------------- checks
for f in anyloc_eval.py anyloc_full.py baselines_full.py siglip_text.py train_pose.py compare_runs.py; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f"; exit 1; }
done
grep -q "last.pth" "$SRC/train_pose.py" || { echo "$SRC/train_pose.py is an old version (it must save last.pth)"; exit 1; }
grep -q -- "--weights" "$SRC/baselines_full.py" || { echo "$SRC/baselines_full.py is an old version (no --weights)"; exit 1; }
grep -q "image_weights" "$SRC/siglip_text.py" || { echo "$SRC/siglip_text.py is an old version (no fine-tuned fusion)"; exit 1; }
grep -q -- "--metric" "$SRC/compare_runs.py" || { echo "$SRC/compare_runs.py is an old version (no --metric)"; exit 1; }
grep -q "per_item.json" "$SRC/anyloc_full.py" || { echo "$SRC/anyloc_full.py is an old version (no per_item.json)"; exit 1; }
for d in "$DB_IMGS" "$DB_POSES" "$QUERIES"; do [ -d "$d" ] || { echo "missing folder $d"; exit 1; }; done
[ -f "$JSON" ] || { echo "missing $JSON"; exit 1; }

mkdir -p "$OUT/_done" "$OUT/logs" "$OUT/comparisons"
if [ -f "$OUT/protocol.txt" ] && [ "$(cat "$OUT/protocol.txt")" != "$PROTOCOL" ]; then
  echo "$OUT was produced with a different protocol:"; echo "  there: $(cat "$OUT/protocol.txt")"; echo "  now:   $PROTOCOL"
  echo "Use an empty OUT folder (e.g. OUT=runs_final2 bash $0)."; exit 1
fi
echo "$PROTOCOL" > "$OUT/protocol.txt"

# ----------------------------------------------------------------------------- helpers
step() {          # step <name> <command...> : run once, log to $OUT/logs/<name>.log, skip if already done
  local name=$1; shift
  if [ -f "$OUT/_done/$name" ]; then echo "[skip] $name"; return 0; fi
  echo; echo "================ $name  ($(date '+%F %T'))"
  "$@" 2>&1 | tee "$OUT/logs/$name.log"
  touch "$OUT/_done/$name"
}

keep_results() {  # keep_results <run folder> <radius> : copy the two result files of a run for that radius
  local run=$1 r=$2 dst="$OUT/radius_$2/$(basename "$1")"
  mkdir -p "$dst"; cp "$run/results.json" "$run/per_item.json" "$dst/"
}

score_one() {     # score_one <run name> <radius> <script> <args...> : score, then keep that radius's result files
  local name=$1 r=$2 script=$3; shift 3
  $PY "$SRC/$script" "$@" $DB $IHM --out "$OUT/$name" --pos-radius "$r"
  keep_results "$OUT/$name" "$r"
}

score() {         # score <run name> <script> <extra args...> : all radii, main radius (1.0, with montages) last
  local name=$1 script=$2; shift 2
  for r in $RADII_EXTRA; do step "score_${name}_r$r" score_one "$name" "$r" "$script" "$@" --no-montage; done
  step "score_${name}_r1.0" score_one "$name" 1.0 "$script" "$@"
}

train() {         # train <train folder> <seed> <extra args...>
  local name=$1 seed=$2; shift 2
  mkdir -p "$OUT/$name"
  # hard negatives depend only on the baseline descriptors and the pair settings (encoded in the file name)
  for f in "$OUT"/train_*/hard_negatives_*.npy; do [ -e "$f" ] && cp -n "$f" "$OUT/$name/" || true; done
  step "$name" $PY "$SRC/train_pose.py" $DB --baseline-run "$OUT/run_megaloc" --out "$OUT/$name" \
       --epochs "$TRAIN_EPOCHS" --seed "$seed" "$@"
}

alias_run() {     # alias_run <source folder> <new name> : result files under a unique name, for compare_runs.py
  mkdir -p "$OUT/$2"; cp "$1/results.json" "$1/per_item.json" "$OUT/$2/"
}

# ----------------------------------------------------------------------------- 0. environment record
if [ ! -f "$OUT/environment.txt" ]; then
  {
    echo "date: $(date '+%F %T %Z')"; echo "host: $(hostname)"; echo "protocol: $PROTOCOL"
    echo; echo "--- python"; $PY --version 2>&1
    echo; echo "--- gpu"; (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available")
    echo; echo "--- data"; echo "database images: $(find "$DB_IMGS" -type f | wc -l)"; echo "pose files: $(find "$DB_POSES" -type f -name '*.txt' | wc -l)"
    echo "query images: $(find "$QUERIES" -type f | wc -l)"; sha256sum "$JSON"
    echo; echo "--- scripts (sha256)"; sha256sum "$SRC"/*.py "$0"
    echo; echo "--- packages"; ($PY -m pip freeze 2>/dev/null || echo "pip freeze not available")
  } > "$OUT/environment.txt"
fi

# ----------------------------------------------------------------------------- 1. zero-shot baselines
score run_anyloc  anyloc_full.py
score run_salad   baselines_full.py --model salad
score run_megaloc baselines_full.py --model megaloc

# ----------------------------------------------------------------------------- 2. fine-tuning (main configuration, all seeds)
FT_RUNS=""
for s in $SEEDS; do
  train "train_head_s$s" "$s"
  score "run_ft_head_s$s" baselines_full.py --model megaloc --weights "$OUT/train_head_s$s/last.pth"
  FT_RUNS="$FT_RUNS $OUT/run_ft_head_s$s"
done
MAIN_FT="run_ft_head_s$(echo $SEEDS | cut -d' ' -f1)"

# ----------------------------------------------------------------------------- 3. fine-tuning ablations (seed 0)
train train_blocks2_s0 0 --unfreeze-blocks 2
score run_ft_blocks2_s0 baselines_full.py --model megaloc --weights "$OUT/train_blocks2_s0/last.pth"
train train_infopairs_s0 0 --pos-min-dist 0.3 --pos-min-angle 15 --pos-min-gap 200 --keyframe-dist 0.15 --keyframe-angle 10
score run_ft_infopairs_s0 baselines_full.py --model megaloc --weights "$OUT/train_infopairs_s0/last.pth"

# conventional protocol for comparison: several epochs, early stopping, best epoch on the validation set
train train_head_best_s0 0 --epochs "$BEST_EPOCHS" --select mean
score run_ft_head_best_s0 baselines_full.py --model megaloc --weights "$OUT/train_head_best_s0/best.pth"

# ----------------------------------------------------------------------------- 3b. map of the scan trajectory (documents the pose data and its scale)
[ -f "$SRC/pose_map.py" ] && step pose_map $PY "$SRC/pose_map.py" --db-poses "$DB_POSES" --json "$JSON" --out "$OUT/pose_map"

# ----------------------------------------------------------------------------- 4. text (SigLIP2), with MegaLoc and with the fine-tuned model
step siglip_megaloc $PY "$SRC/siglip_text.py" $DB $IHM --megaloc-run "$OUT/run_megaloc" --out "$OUT/siglip_megaloc" --weight "$TEXT_WEIGHT"
mkdir -p "$OUT/siglip_ft"                                   # reuse the SigLIP2 database embeddings
for f in db_desc.npy db_done.txt db_names.json; do [ -e "$OUT/siglip_ft/$f" ] || cp "$OUT/siglip_megaloc/$f" "$OUT/siglip_ft/"; done
step siglip_ft $PY "$SRC/siglip_text.py" $DB $IHM --megaloc-run "$OUT/$MAIN_FT" --out "$OUT/siglip_ft" --weight "$TEXT_WEIGHT"
alias_run "$OUT/siglip_megaloc/text_only"         run_text_only
alias_run "$OUT/siglip_megaloc/siglip_image"      run_siglip_image
alias_run "$OUT/siglip_megaloc/megaloc_plus_text" run_megaloc_plus_text
alias_run "$OUT/siglip_ft/megaloc_plus_text"      run_ft_plus_text

# ----------------------------------------------------------------------------- 5. comparisons (always rewritten)
C="$OUT/comparisons"
cmp() { local out=$1; shift; echo; echo "================ comparison: $out"; $PY "$SRC/compare_runs.py" "$@" --out "$C/$out"; }
cmp baselines.txt     "$OUT/run_anyloc" "$OUT/run_salad" "$OUT/run_megaloc"
cmp finetuning.txt    "$OUT/run_megaloc" $FT_RUNS
cmp ablations.txt     "$OUT/run_megaloc" "$OUT/$MAIN_FT" "$OUT/run_ft_head_best_s0" "$OUT/run_ft_blocks2_s0" "$OUT/run_ft_infopairs_s0"
cmp text.txt          "$OUT/run_megaloc" "$OUT/run_text_only" "$OUT/run_megaloc_plus_text" "$OUT/$MAIN_FT" "$OUT/run_ft_plus_text"
# the same comparisons on mAP@10 (paired permutation test instead of McNemar)
MAP="--metric mAP@10"
cmp baselines_map.txt  "$OUT/run_anyloc" "$OUT/run_salad" "$OUT/run_megaloc" $MAP
cmp finetuning_map.txt "$OUT/run_megaloc" $FT_RUNS $MAP
cmp ablations_map.txt  "$OUT/run_megaloc" "$OUT/$MAIN_FT" "$OUT/run_ft_head_best_s0" "$OUT/run_ft_blocks2_s0" "$OUT/run_ft_infopairs_s0" $MAP
cmp text_map.txt       "$OUT/run_megaloc" "$OUT/run_text_only" "$OUT/run_megaloc_plus_text" "$OUT/$MAIN_FT" "$OUT/run_ft_plus_text" $MAP
for r in $RADII_EXTRA; do
  R="$OUT/radius_$r"
  cmp "baselines_r$r.txt"  "$R/run_anyloc" "$R/run_salad" "$R/run_megaloc"
  cmp "finetuning_r$r.txt" "$R/run_megaloc" $(for s in $SEEDS; do echo "$R/run_ft_head_s$s"; done)
done

# comparisons are rewritten on every run, so record which script versions produced them
{ echo "comparisons written: $(date '+%F %T %Z')"; sha256sum "$SRC/compare_runs.py" "$0"; } > "$C/versions.txt"

# ----------------------------------------------------------------------------- 6. one summary table
$PY - "$OUT" $RADII_EXTRA <<'PYEOF'
import json, os, sys, csv
out, radii = sys.argv[1], ["1.0"] + sys.argv[2:]
rows = []
for r in radii:
    base = os.path.join(out, f"radius_{r}")
    names = sorted(os.listdir(base)) if os.path.isdir(base) else []
    if r == "1.0": names += [n for n in ("run_text_only", "run_siglip_image", "run_megaloc_plus_text", "run_ft_plus_text") if os.path.isdir(os.path.join(out, n))]
    for n in names:
        f = os.path.join(base, n, "results.json")
        if not os.path.exists(f): f = os.path.join(out, n, "results.json")
        res = json.load(open(f)); p = res.get("relaxed", {}).get("per_item") or res["per_item"]; s = res["per_item"]
        rows.append(dict(radius=r, run=n, items=p["n"], R1=p["R@1"], R1_lo=p["R@1_ci95"][0], R1_hi=p["R@1_ci95"][1], R5=p["R@5"], R10=p["R@10"],
                         mAP10=p["mAP@10"], strict_R1=s["R@1"], strict_R10=s["R@10"], loc_median=s.get("loc_err_median", ""), within_1=s.get("within_1", "")))
with open(os.path.join(out, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("\n================ summary (per item, relaxed ground truth)")
print(f"{'radius':<7}{'run':<26}{'R@1':>7}{'  95% CI':>14}{'R@5':>7}{'R@10':>7}{'mAP@10':>8}{'loc':>7}")
for x in rows:
    print(f"{x['radius']:<7}{x['run']:<26}{x['R1']:>7.3f}   [{x['R1_lo']:.2f}-{x['R1_hi']:.2f}]{x['R5']:>7.3f}{x['R10']:>7.3f}{x['mAP10']:>8.3f}{str(x['loc_median']):>7}")
ft = [x["R1"] for x in rows if x["radius"] == "1.0" and x["run"].startswith("run_ft_head_s") and "best" not in x["run"]]
if ft: print(f"\nfine-tuned (head only), {len(ft)} seeds at radius 1.0: R@1 mean {sum(ft)/len(ft):.3f}, min {min(ft):.3f}, max {max(ft):.3f}")
print(f"saved {os.path.join(out, 'summary.csv')}")
PYEOF

echo; echo "All done ($(date '+%F %T')). Results: $OUT/summary.csv, $OUT/comparisons/, logs in $OUT/logs/, environment in $OUT/environment.txt"
