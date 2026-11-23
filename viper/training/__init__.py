"""Fine-tuning a place recognition model on the scan, with training pairs taken from the camera poses.

The IHM photographs are never used here.

    pairs.py        positives, negatives, keyframes, train/validation split and hard negatives (numpy only)
    augment.py      image transforms: close-up augmentation for anchors, the fixed validation close-ups
    batches.py      the training dataset: one anchor with its positives and hard negatives per sample
    loss.py         multi-similarity loss
    model.py        the models that can be trained, and which parameters are trained
    validation.py   close-ups of held-out scan frames, retrieved against the scan
    trainer.py      the training loop, checkpoints and log
    command.py      the `train` command
"""
