"""Image transforms for training and validation.

The anchor of every training sample imitates an inspector's close-up photograph. Each augmentation matches
a cause found in the failure analysis: random crop (close-up), colour jitter (lighting), perspective
(viewing angle) and random erasing (the inspector's reference marker). Positives and negatives stay full
frames, as the scan images are at test time.
"""
import numpy as np

from viper.utils.images import IMAGENET_MEAN, IMAGENET_STD


def to_tensor(image):
    """PIL image -> ImageNet-normalized (3, H, W) float tensor."""
    import torch

    x = torch.from_numpy(np.asarray(image, np.uint8).copy()).permute(2, 0, 1).float() / 255

    return (x - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)


def build_transforms(size, crop_min):
    """(anchor transform, random erasing for the anchor tensor, transform for positives and negatives)."""
    from torchvision import transforms as T

    anchor = T.Compose([
        T.RandomResizedCrop(size, scale=(crop_min, 1.0), ratio=(0.6, 1.8), antialias=True),   # close-up
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.05),                 # lighting
        T.RandomPerspective(distortion_scale=0.3, p=0.3),                                     # viewing angle
        T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.5))], p=0.2),
    ])

    erase = T.RandomErasing(p=0.3, scale=(0.02, 0.2))                                         # occlusion
    full_frame = T.Compose([T.Resize(size, antialias=True), T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)])

    return anchor, erase, full_frame


def validation_closeup(seed, area=0.10):
    """A fixed hard close-up per validation query: transform(i, image) -> image, the same for every run.

    A crop of `area` of the image, with a brightness and contrast change and some blur, all drawn from a
    random generator seeded with seed + i.
    """
    from PIL import ImageEnhance, ImageFilter

    def transform(i, image):
        rng = np.random.default_rng(seed + i)
        width, height = image.size
        scale = np.sqrt(area)
        crop_w, crop_h = max(8, int(width * scale)), max(8, int(height * scale))
        x0, y0 = int(rng.integers(0, width - crop_w + 1)), int(rng.integers(0, height - crop_h + 1))

        image = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.5, 1.5)))
        image = ImageEnhance.Contrast(image).enhance(float(rng.uniform(0.6, 1.4)))
        
        return image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.0, 1.5))))

    return transform
