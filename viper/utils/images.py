"""Reading image files into normalized batches for the DINOv2-based models (AnyLoc, MegaLoc, SALAD)."""
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_rgb(path, size):
    """RGB uint8 array resized to size = (height, width) with area interpolation."""
    image = cv2.imread(path)
    if image is None:
        raise SystemExit(f"cannot read image {path}")
    
    height, width = size

    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (width, height), interpolation=cv2.INTER_AREA)


class ImageLoader:
    """loader(paths) -> (B, 3, H, W) float tensor on the device, scaled to [0, 1] and ImageNet-normalized."""

    def __init__(self, size, device):
        import torch

        self.size = tuple(size)
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, paths):
        import torch

        images = [torch.from_numpy(read_rgb(p, self.size)).permute(2, 0, 1).float() / 255 for p in paths]
        
        return (torch.stack(images).to(self.device) - self.mean) / self.std
