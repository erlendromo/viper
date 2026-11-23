#!/usr/bin/env bash
# Full ViPeR experiment from a clean slate: baselines, fine-tuning, text fusion, scoring, comparisons.
#
#   cd ~/viper && source .venv/bin/activate
#   tmux new -s final
#   bash run_all.sh 2>&1 | tee run_all.log
#
# Everything is written under $OUT (default: runs_final). Paths are relative to the repository root.
# If the script stops part-way, run the same command again: finished steps are skipped (each step leaves a
# marker in $OUT/_done). To redo one step, delete its marker and its output folder. Re-running it on a
# finished $OUT skips every model step and only rewrites the comparisons and the summary. The script refuses
# to start if $OUT holds results from another protocol.
#
# Protocol (fixed before any result was seen):
#   - Ground truth: relaxed, radius 1.0 m, angle 45 deg (main). Radius 0.5 and 2.0 m as sensitivity.
#   - Fine-tuning: MegaLoc, head only, exactly TRAIN_EPOCHS epoch(s), weights of the LAST epoch (no selection
#     on validation), repeated for every seed in SEEDS. The first seed is the model used for the text fusion.
#   - Ablations (seed 0, same budget): last two backbone blocks unfrozen, and informative pairs + keyframes.
#   - Comparison variant (seed 0): head only, up to BEST_EPOCHS epochs with early stopping, keeping the epoch
#     with the best validation score (mean of R@1/R@5/R@10). This is the conventional "best weights" protocol.
#   - Text fusion: fixed weight 1.0, never tuned.
set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------- configuration
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
BEST_EPOCHS=${BEST_EPOCHS:-10}             # max epochs of the "best epoch on validation" variant

VIPER="$PY main.py"
DB="--db-imgs $DB_IMGS --db-poses $DB_POSES"
IHM="--queries $QUERIES --json $JSON"
PROTOCOL="epochs=$TRAIN_EPOCHS seeds=$SEEDS radii=$RADII_EXTRA 1.0 text_weight=$TEXT_WEIGHT weights=last.pth best_variant_epochs=$BEST_EPOCHS"

# ----------------------------------------------------------------------------- checks
$VIPER --help > /dev/null || { echo "main.py does not start: check the environment (see README.md)"; exit 1; }
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

score_one() {     # score_one <run name> <radius> <command> <args...> : score, then keep that radius's result files
  local name=$1 r=$2 command=$3; shift 3
  $VIPER "$command" "$@" $DB $IHM --out "$OUT/$name" --pos-radius "$r"
  keep_results "$OUT/$name" "$r"
}

score() {         # score <run name> <command> <extra args...> : all radii, main radius (1.0, with montages) last
  local name=$1 command=$2; shift 2
  for r in $RADII_EXTRA; do step "score_${name}_r$r" score_one "$name" "$r" "$command" "$@" --no-montage; done
  step "score_${name}_r1.0" score_one "$name" 1.0 "$command" "$@"
}

train() {         # train <train folder> <seed> <extra args...>
  local name=$1 seed=$2; shift 2
  mkdir -p "$OUT/$name"
  # hard negatives depend only on the baseline descriptors and the pair settings (encoded in the file name)
  for f in "$OUT"/train_*/hard_negatives_*.npy; do [ -e "$f" ] && cp -n "$f" "$OUT/$name/" || true; done
  step "$name" $VIPER train $DB --baseline-run "$OUT/run_megaloc" --out "$OUT/$name" \
       --epochs "$TRAIN_EPOCHS" --seed "$seed" "$@"
}

alias_run() {     # alias_run <source folder> <new name> : result files under a unique name, for compare
  mkdir -p "$OUT/$2"; cp "$1/results.json" "$1/per_item.json" "$OUT/$2/"
}

code_hashes() {   # sha256 of every source file, so a result can be traced to the code that made it
  sha256sum main.py "$(basename "$0")" $(find viper -name '*.py' | sort)
}

# ----------------------------------------------------------------------------- 0. environment record
if [ ! -f "$OUT/environment.txt" ]; then
  {
    echo "date: $(date '+%F %T %Z')"; echo "host: $(hostname)"; echo "protocol: $PROTOCOL"
    echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repository')"
    echo; echo "--- python"; $PY --version 2>&1
    echo; echo "--- gpu"; (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available")
    echo; echo "--- data"; echo "database images: $(find "$DB_IMGS" -type f | wc -l)"; echo "pose files: $(find "$DB_POSES" -type f -name '*.txt' | wc -l)"
    echo "query images: $(find "$QUERIES" -type f | wc -l)"; sha256sum "$JSON"
    echo; echo "--- code (sha256)"; code_hashes
    echo; echo "--- packages"; ($PY -m pip freeze 2>/dev/null || echo "pip freeze not available")
  } > "$OUT/environment.txt"
fi

# ----------------------------------------------------------------------------- 1. zero-shot baselines
score run_anyloc  anyloc
score run_salad   salad
score run_megaloc megaloc

# ----------------------------------------------------------------------------- 2. fine-tuning (main configuration, all seeds)
FT_RUNS=""
for s in $SEEDS; do
  train "train_head_s$s" "$s"
  score "run_ft_head_s$s" megaloc --weights "$OUT/train_head_s$s/last.pth"
  FT_RUNS="$FT_RUNS $OUT/run_ft_head_s$s"
done
MAIN_FT="run_ft_head_s$(echo $SEEDS | cut -d' ' -f1)"

# ----------------------------------------------------------------------------- 3. fine-tuning ablations (seed 0)
train train_blocks2_s0 0 --unfreeze-blocks 2
score run_ft_blocks2_s0 megaloc --weights "$OUT/train_blocks2_s0/last.pth"
train train_infopairs_s0 0 --pos-min-dist 0.3 --pos-min-angle 15 --pos-min-gap 200 --keyframe-dist 0.15 --keyframe-angle 10
score run_ft_infopairs_s0 megaloc --weights "$OUT/train_infopairs_s0/last.pth"

# conventional protocol for comparison: several epochs, early stopping, best epoch on the validation set
train train_head_best_s0 0 --epochs "$BEST_EPOCHS" --select mean
score run_ft_head_best_s0 megaloc --weights "$OUT/train_head_best_s0/best.pth"

# ----------------------------------------------------------------------------- 3b. map of the scan trajectory (pose data and scale)
step pose_map $VIPER pose-map --db-poses "$DB_POSES" --json "$JSON" --out "$OUT/pose_map"

# ----------------------------------------------------------------------------- 4. text (SigLIP2), with MegaLoc and with the fine-tuned model
step siglip_megaloc $VIPER siglip $DB $IHM --megaloc-run "$OUT/run_megaloc" --out "$OUT/siglip_megaloc" --weight "$TEXT_WEIGHT"
mkdir -p "$OUT/siglip_ft"                                   # reuse the SigLIP2 database embeddings
for f in db_desc.npy db_done.txt db_names.json; do [ -e "$OUT/siglip_ft/$f" ] || cp "$OUT/siglip_megaloc/$f" "$OUT/siglip_ft/"; done
step siglip_ft $VIPER siglip $DB $IHM --megaloc-run "$OUT/$MAIN_FT" --out "$OUT/siglip_ft" --weight "$TEXT_WEIGHT"
alias_run "$OUT/siglip_megaloc/text_only"         run_text_only
alias_run "$OUT/siglip_megaloc/siglip_image"      run_siglip_image
alias_run "$OUT/siglip_megaloc/megaloc_plus_text" run_megaloc_plus_text
alias_run "$OUT/siglip_ft/megaloc_plus_text"      run_ft_plus_text

# ----------------------------------------------------------------------------- 5. comparisons (always rewritten)
C="$OUT/comparisons"
cmp() { local out=$1; shift; echo; echo "================ comparison: $out"; $VIPER compare "$@" --out "$C/$out"; }
BASELINES="$OUT/run_anyloc $OUT/run_salad $OUT/run_megaloc"
ABLATIONS="$OUT/run_megaloc $OUT/$MAIN_FT $OUT/run_ft_head_best_s0 $OUT/run_ft_blocks2_s0 $OUT/run_ft_infopairs_s0"
TEXT="$OUT/run_megaloc $OUT/run_text_only $OUT/run_megaloc_plus_text $OUT/$MAIN_FT $OUT/run_ft_plus_text"
cmp baselines.txt  $BASELINES
cmp finetuning.txt "$OUT/run_megaloc" $FT_RUNS
cmp ablations.txt  $ABLATIONS
cmp text.txt       $TEXT
# the same comparisons on mAP@10 (paired permutation test instead of McNemar)
MAP="--metric mAP@10"
cmp baselines_map.txt  $BASELINES $MAP
cmp finetuning_map.txt "$OUT/run_megaloc" $FT_RUNS $MAP
cmp ablations_map.txt  $ABLATIONS $MAP
cmp text_map.txt       $TEXT $MAP
for r in $RADII_EXTRA; do
  R="$OUT/radius_$r"
  cmp "baselines_r$r.txt"  "$R/run_anyloc" "$R/run_salad" "$R/run_megaloc"
  cmp "finetuning_r$r.txt" "$R/run_megaloc" $(for s in $SEEDS; do echo "$R/run_ft_head_s$s"; done)
done

# comparisons are rewritten on every run, so record which code produced them
{ echo "comparisons written: $(date '+%F %T %Z')"; code_hashes; } > "$C/versions.txt"

# ----------------------------------------------------------------------------- 6. one summary table
$VIPER summary "$OUT" --radii-extra $RADII_EXTRA

echo; echo "All done ($(date '+%F %T')). Results: $OUT/summary.csv, $OUT/comparisons/, logs in $OUT/logs/, environment in $OUT/environment.txt"
