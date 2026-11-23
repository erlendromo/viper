"""Local features for AnyLoc: the value facet of one DINOv2 attention layer, one vector per image patch."""
import cv2
import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DinoValueFeatures:
    """Calling the extractor on a list of image paths returns an array (images, patches, channels).

    model_name  a DINOv2 model from torch.hub, e.g. dinov2_vitg14
    layer       the transformer block whose attention values are used (31 for ViT-G/14, as in AnyLoc)
    image_size  (height, width) the images are resized to; both should be multiples of the patch size 14
    """

    def __init__(self, model_name, layer, image_size, device):
        self.device = device
        self.height, self.width = image_size
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device).eval()
        self._qkv = None
        self.model.blocks[layer].attn.qkv.register_forward_hook(self._keep_qkv)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @property
    def dim(self):
        """Channels per patch feature."""
        return self.model.embed_dim

    def _keep_qkv(self, module, inputs, output):
        """Forward hook: store the query, key and value projections of the chosen layer."""
        self._qkv = output

    def load_image(self, path):
        """RGB image, resized, as a (3, H, W) tensor in [0, 1]."""
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return torch.from_numpy(image).permute(2, 0, 1).float() / 255

    def __call__(self, paths):
        batch = (torch.stack([self.load_image(p) for p in paths]).to(self.device) - self.mean) / self.std
        with torch.no_grad():
            self.model.forward_features(batch)              # the hook captures the layer output
        values = self._qkv[..., 2 * self.dim:]              # qkv = [queries | keys | values]
        
        return values[:, 1:].float().cpu().numpy()          # drop the class token, keep the patches

    def per_image(self, paths, batch_size=8):
        """Yield the (patches, channels) features of one image at a time, computed in batches."""
        for start in range(0, len(paths), batch_size):
            yield from self(paths[start:start + batch_size])
