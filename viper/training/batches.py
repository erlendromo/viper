"""The training dataset. One sample = one anchor (augmented close-up) + its positives + its hard negatives.

The class is defined at module level so it can be pickled: Python 3.14 starts DataLoader workers in fresh
processes. The transforms are built lazily inside each worker.
"""
import numpy as np

from .augment import build_transforms, to_tensor


class AnchorGroups:
    """Each item is a tensor (1 + n_pos + n_hard, 3, H, W) and the frame indices of its images.

    The anchor is drawn at random on every access, so `length` only sets the number of samples per epoch.
    """

    def __init__(self, images, anchors, positives, hard_negatives, size, n_pos, n_hard, crop_min, length):
        self.images = images
        self.anchors = anchors
        self.positives = positives
        self.hard_negatives = hard_negatives
        self.size = size
        self.n_pos = n_pos
        self.n_hard = n_hard
        self.crop_min = crop_min
        self.length = length
        self.transforms = None

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        import torch
        from PIL import Image

        if self.transforms is None:
            self.transforms = build_transforms(self.size, self.crop_min)
        anchor_tf, erase, full_frame_tf = self.transforms

        # the order of the random draws below fixes the training samples for a given seed: keep it
        anchor = int(self.anchors[torch.randint(len(self.anchors), (1,)).item()])
        positives = self.positives[anchor]
        positives = positives[torch.randint(len(positives), (self.n_pos,)).numpy()]          # with replacement
        hard = self.hard_negatives[anchor][self.hard_negatives[anchor] >= 0]
        hard = hard[torch.randperm(len(hard))[:self.n_hard].numpy()] if len(hard) else np.array([], int)
        frames = [anchor] + np.atleast_1d(positives).tolist() + np.atleast_1d(hard).tolist()

        def open_rgb(i):
            return Image.open(self.images[i]).convert("RGB")

        tensors = [erase(to_tensor(anchor_tf(open_rgb(anchor))))]
        tensors += [to_tensor(full_frame_tf(open_rgb(i))) for i in frames[1:]]
        
        return torch.stack(tensors), torch.tensor(frames)


def collate_groups(batch):
    """Concatenate the groups of a batch into one image tensor and one index tensor."""
    import torch

    return torch.cat([images for images, _ in batch]), torch.cat([frames for _, frames in batch])
