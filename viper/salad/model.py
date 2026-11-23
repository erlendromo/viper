"""The SALAD model from torch.hub. It returns one 8448-d descriptor per image.

The hub repository has its own top-level `utils` and `models` folders, which torch.hub puts on the import
path while loading. This is why all ViPeR code lives inside the `viper` package.
"""
from viper.utils.embedder import load_finetuned_weights

HUB_REPO = "serizba/salad"
HUB_ENTRY = "dinov2_salad"
INPUT_SIZE = (322, 322)          # (height, width), as in the SALAD paper


def load_model(weights=None):
    """Pretrained SALAD on the CPU, with fine-tuned weights loaded on top if a path is given."""
    import torch

    model = torch.hub.load(HUB_REPO, HUB_ENTRY)
    if weights:
        load_finetuned_weights(model, weights)
    
    return model
