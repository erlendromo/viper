"""The MegaLoc model from torch.hub. It returns one 8448-d descriptor per image."""
from viper.utils.embedder import load_finetuned_weights

HUB_REPO = "gmberton/MegaLoc"
HUB_ENTRY = "get_trained_model"
INPUT_SIZE = (322, 322)          # (height, width), as in the MegaLoc paper


def load_model(weights=None):
    """Pretrained MegaLoc on the CPU, with fine-tuned weights loaded on top if a path is given."""
    import torch

    model = torch.hub.load(HUB_REPO, HUB_ENTRY)
    if weights:
        load_finetuned_weights(model, weights)
    
    return model
