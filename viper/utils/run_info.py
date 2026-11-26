"""What a finished run_all.sh folder took to produce: environment, time, disk space and training statistics.

    python main.py run-info runs_final --log run_all.log

Read from the run folder (nothing is recomputed)
    environment     environment.txt: date, GPU, Python and the versions of the key packages
    timing          the timers printed by the commands: database descriptors per model, hard-negative
                    mining and training per run. With --log (the console output of run_all.sh), also the
                    wall time of every step, from its start line in the log to its marker in _done/
    disk space      by kind of file: descriptor caches, checkpoints, montages, other
    training        per training run: frames, anchors, positives per anchor, hard negatives, validation
                    queries, trainable parameters, and the loss and validation score per epoch

Files: run_info.json in the run folder.
"""
import datetime
import json
import os
import re

HELP = "environment, run time, disk space and training statistics of a run_all.sh folder (run_info.json)"

PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "opencv-python", "pillow", "matplotlib",
            "transformers", "huggingface-hub", "safetensors", "pytorch-lightning", "pytorch-metric-learning")

PROGRESS = re.compile(r"(\d+)/(\d+) images, (\d+) ?s")                     # descriptor cache progress
STEP = re.compile(r"step (\d+): loss ([\d.]+)\s+\((\d+)s\)")                 # training progress
MINING = re.compile(r"\[mining\] (\d+)/(\d+) frames, (\d+)s")
STEP_START = re.compile(r"^=+ (\S+)\s+\((\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\)")
TRAIN_LINES = {
    "frames_with_pose": re.compile(r"\[data\] (\d+) frames with poses"),
    "keyframes": re.compile(r"keyframe thinning: (\d+) -> (\d+) training frames"),
    "split": re.compile(r"train frames (\d+) \((\d+) usable as anchors\), validation frames (\d+), median positives per anchor (\d+)"),
    "hard_negatives": re.compile(r"anchors with at least one hard negative: (\d+)/(\d+)"),
    "validation": re.compile(r"\[val\] (\d+) close-up queries against (\d+) scan frames"),
    "module": re.compile(r"^\s{4}(\S+)\s+([\d.]+)M\s+([\d.]+)M\s*$"),
}


def add_arguments(parser):
    parser.add_argument("out", help="output folder of run_all.sh, e.g. runs_final")
    parser.add_argument("--log", help="console output of run_all.sh (e.g. run_all.log), for the wall time per step")


def run(args):
    info = dict(
        environment=environment(os.path.join(args.out, "environment.txt")),
        descriptor_passes=descriptor_passes(args.out),
        training=training_runs(args.out),
        wall_time_per_step=wall_times(args.out, args.log) if args.log else {},
        disk_space=disk_space(args.out),
    )

    with open(os.path.join(args.out, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    print_summary(info)
    print(f"\nsaved {os.path.join(args.out, 'run_info.json')}")


# ------------------------------------------------------------------------------------------------ readers

def read_lines(path):
    if not os.path.exists(path):
        return []
    
    with open(path, errors="replace") as fh:
        return fh.read().splitlines()


def step_logs(out):
    """Step name -> lines of its log, for every finished step."""
    done = os.path.join(out, "_done")
    names = sorted(os.listdir(done)) if os.path.isdir(done) else []

    return {name: read_lines(os.path.join(out, "logs", f"{name}.log")) for name in names}


def environment(path):
    """Sections of environment.txt, with the key packages picked out of the pip freeze list."""
    lines = read_lines(path)
    if not lines:
        return {}
    
    sections, current = {"header": []}, "header"
    for line in lines:
        header = re.match(r"^--- (.+)$", line)
        if header:
            current = header.group(1).strip()
            sections[current] = []
        elif line.strip():
            sections[current].append(line.strip())

    fields = dict(line.split(": ", 1) for line in sections["header"] if ": " in line)
    versions = {}

    for line in sections.get("packages", []):
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(?:==|@)\s*(.+)$", line)
        if match and match.group(1).lower().replace("_", "-") in PACKAGES:
            versions[match.group(1).lower().replace("_", "-")] = match.group(2).strip()

    gpu = [l for l in sections.get("gpu", []) if not l.startswith("name")]

    return dict(date=fields.get("date"), git_commit=fields.get("git commit"), python=" ".join(sections.get("python", [])),
                gpu=gpu, data=sections.get("data", []), packages=versions)


def descriptor_passes(out):
    """Images and seconds of every database descriptor pass that was computed (not read from the cache)."""
    passes = {}
    for name, lines in step_logs(out).items():
        if not any("computing descriptors" in l for l in lines):
            continue

        progress = [PROGRESS.search(l) for l in lines]
        progress = [m for m in progress if m]

        if progress:
            done, total, seconds = map(int, progress[-1].groups())
            passes[name] = dict(images=done, of=total, seconds=seconds, images_per_second=round(done / max(seconds, 1), 1))

    return passes


def training_runs(out):
    runs = {}
    for name, lines in step_logs(out).items():
        if not name.startswith("train_"):
            continue

        record = dict(modules={})
        for line in lines:
            for key, pattern in TRAIN_LINES.items():
                m = pattern.search(line)
                if not m:
                    continue

                g = m.groups()
                if key == "module":
                    record["modules"][g[0]] = dict(total_M=float(g[1]), trainable_M=float(g[2]))
                elif key == "split":
                    record.update(train_frames=int(g[0]), anchors=int(g[1]), validation_frames=int(g[2]), median_positives=int(g[3]))
                elif key == "keyframes":
                    record["keyframes"] = dict(before=int(g[0]), after=int(g[1]))
                elif key == "hard_negatives":
                    record["anchors_with_hard_negatives"] = dict(n=int(g[0]), of=int(g[1]))
                elif key == "validation":
                    record["validation"] = dict(queries=int(g[0]), database=int(g[1]))
                else:
                    record[key] = int(g[0])

        steps = [m for m in (STEP.search(l) for l in lines) if m]
        if steps:
            record["training"] = dict(batches_logged=int(steps[-1].group(1)), seconds=int(steps[-1].group(3)),
                                      loss_curve=[(int(m.group(1)), float(m.group(2))) for m in steps])
            
        mining = [m for m in (MINING.search(l) for l in lines) if m]
        if mining:
            record["mining_seconds"] = int(mining[-1].group(3))

        log_file = os.path.join(out, name, "train_log.json")
        if os.path.exists(log_file):
            with open(log_file) as fh:
                record["epochs"] = json.load(fh)["log"]

        runs[name] = record

    return runs


def wall_times(out, log_path):
    """Seconds from the last start line of each step in the run_all.sh log to its marker in _done/."""
    starts = {}
    for line in read_lines(log_path):
        m = STEP_START.match(line)
        if m:
            starts[m.group(1)] = datetime.datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S")

    times = {}
    for name in step_logs(out):
        marker = os.path.join(out, "_done", name)
        if name in starts:
            end = datetime.datetime.fromtimestamp(os.path.getmtime(marker))
            seconds = (end - starts[name]).total_seconds()

            if seconds >= 0:
                times[name] = int(seconds)

    return times


def disk_space(out):
    """Bytes by kind of file, and the largest top-level folders."""
    kinds, folders = {}, {}
    for root, _, files in os.walk(out):
        for f in files:
            size = os.path.getsize(os.path.join(root, f))
            if f.endswith("_desc.npy"):
                kind = "descriptor caches"
            elif f.endswith(".pth"):
                kind = "checkpoints"
            elif os.path.basename(root) == "montage":
                kind = "montages"
            elif f.startswith("hard_negatives"):
                kind = "hard negatives"
            else:
                kind = "other"

            kinds[kind] = kinds.get(kind, 0) + size
            top = os.path.relpath(root, out).split(os.sep)[0]
            folders[top] = folders.get(top, 0) + size

    largest = dict(sorted(folders.items(), key=lambda kv: -kv[1])[:8])

    return dict(total=sum(kinds.values()), by_kind=kinds, largest_folders=largest)


# ------------------------------------------------------------------------------------------------ output

def human_time(seconds):
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min" if seconds >= 60 else f"{seconds} s"


def human_size(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def print_summary(info):
    env = info["environment"]
    print(f"[environment] {env.get('date')} | python {env.get('python')} | gpu {env.get('gpu')} | commit {env.get('git_commit')}")
    print(f"[packages] {env.get('packages')}")
    print("[descriptor passes] (database descriptors computed, not read from the cache)")

    for name, p in info["descriptor_passes"].items():
        print(f"    {name:<34}{p['images']:>6} images  {human_time(p['seconds']):>8}  {p['images_per_second']:>6} images/s")

    print("[training]")
    for name, r in info["training"].items():
        t = r.get("training", {})
        params = {k: f"{v['trainable_M']}/{v['total_M']}M" for k, v in r.get("modules", {}).items()}
        hn = r.get("anchors_with_hard_negatives", {})

        print(f"    {name:<22} train frames {r.get('train_frames')}, anchors {r.get('anchors')}, median positives {r.get('median_positives')}, "
              f"hard negatives for {hn.get('n')}/{hn.get('of')} anchors, validation {r.get('validation')}"
              + (f", keyframes {r['keyframes']}" if 'keyframes' in r else ""))
        print(f"    {'':<22} trainable/total {params}, mining {human_time(r['mining_seconds']) if 'mining_seconds' in r else 'cached'}, "
              f"{t.get('batches_logged')} batches in {human_time(t['seconds']) if t else '-'}, "
              f"epochs {[(e['epoch'], e.get('loss'), e['val'].get('R@1')) for e in r.get('epochs', [])]}")
        
    if info["wall_time_per_step"]:
        total = sum(info["wall_time_per_step"].values())
        print(f"[wall time per step] total {human_time(total)}")

        for name, s in sorted(info["wall_time_per_step"].items(), key=lambda kv: -kv[1]):
            print(f"    {name:<34}{human_time(s):>8}")
            
    d = info["disk_space"]
    print(f"[disk space] total {human_size(d['total'])}: " + ", ".join(f"{k} {human_size(v)}" for k, v in sorted(d['by_kind'].items(), key=lambda kv: -kv[1])))
    print("    largest folders: " + ", ".join(f"{k} {human_size(v)}" for k, v in d["largest_folders"].items()))
