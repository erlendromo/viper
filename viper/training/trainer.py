"""The training loop.

Files written to the output folder
    best.pth          weights with the best validation score so far (epoch 0 = the pretrained model)
    last.pth          weights after the most recent epoch (the main protocol uses these: no selection)
    train_log.json    arguments, and loss and validation scores per epoch
"""
import json
import os
import time

import numpy as np

from .loss import multi_similarity_loss

LOG_EVERY = 50                 # batches between loss printouts
VALIDATION_SOLVED = 0.85       # validation Recall@1 above which the task is too easy to show a gain


def selection_score(name):
    """Validation score that picks the best epoch: mean of R@1, R@5 and R@10, or R@1 alone."""
    if name == "mean":
        return lambda v: float(np.mean([v["R@1"], v["R@5"], v["R@10"]]))
    
    return lambda v: v["R@1"]


def pair_masks(frames, centres, forwards, pos_radius, cos_min, neg_radius):
    """Positive and negative masks between all images of a batch, from their camera poses."""
    import torch

    distance = torch.cdist(centres[frames], centres[frames])
    cos_angle = forwards[frames] @ forwards[frames].T
    itself = torch.eye(len(frames), dtype=torch.bool, device=frames.device)
    positive = (distance <= pos_radius) & (cos_angle >= cos_min) & ~itself
    negative = distance > neg_radius

    return positive, negative


def train(model, params, optimizer, loader, validate, centres, forwards, args, run_args, device):
    """Train for up to args.epochs epochs with early stopping. validate() -> dict of validation scores."""
    import torch

    score = selection_score(args.select)
    cos_min = float(np.cos(np.radians(args.pos_angle)))
    warmup = max(1, args.iters // 2)                       # batches of linear learning-rate warm-up
    log, start_time = [], time.time()

    def save(name, epoch, val):
        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val": val}, os.path.join(args.out, name))

    def write_log():
        with open(os.path.join(args.out, "train_log.json"), "w") as fh:
            json.dump(dict(args=run_args, log=log), fh, indent=1)

    val = validate()
    best, epochs_without_gain = score(val), 0
    print(f"[epoch 0] before training: val {val}", flush=True)
    log.append(dict(epoch=0, val=val))

    if val["R@1"] > VALIDATION_SOLVED:
        print("[val] NOTE: the validation task is nearly solved before training, so it cannot show an improvement. "
              "Stop and rerun with a smaller --val-area (e.g. half the current value).", flush=True)
    
    save("best.pth", 0, val)

    step = 0
    for epoch in range(1, args.epochs + 1):
        losses, skipped = [], 0
        for images, frames in loader:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * min(1.0, (step + 1) / warmup)
            
            images, frames = images.to(device, non_blocking=True), frames.to(device)
            embeddings = torch.nn.functional.normalize(model(images).float(), dim=1)
            positive, negative = pair_masks(frames, centres, forwards, args.pos_radius, cos_min, args.neg_radius)
            loss = multi_similarity_loss(embeddings, positive, negative)

            if loss is None:
                skipped += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            losses.append(loss.item())

            step += 1
            if step % LOG_EVERY == 0:
                print(f"    step {step}: loss {np.mean(losses[-LOG_EVERY:]):.4f}  ({time.time() - start_time:.0f}s)", flush=True)

        val = validate()
        record = dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None, skipped_batches=skipped, val=val)
        log.append(record)
        improved = score(val) > best
        print(f"[epoch {epoch}] loss {record['loss']}  val {val}  score {score(val):.3f}" + ("  <- best" if improved else ""), flush=True)
        write_log()
        save("last.pth", epoch, val)
        
        if improved:
            best, epochs_without_gain = score(val), 0
            save("best.pth", epoch, val)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"no improvement for {args.patience} epochs: stopping")
                break

    write_log()
    print(f"done. best validation score ({args.select}) {best:.3f}; weights in {os.path.join(args.out, 'best.pth')}")
