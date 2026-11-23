"""Running a global-descriptor model (one vector per image, e.g. MegaLoc or SALAD) on image files."""
from .images import ImageLoader


class Embedder:
    """embedder(paths) -> (B, D) float32 array of L2-normalized descriptors.

    model   a torch module that maps an image batch (B, 3, H, W) to descriptors (B, D)
    size    (height, width) the images are resized to
    fp16    half precision on the GPU (faster; not used for the reported results)
    """

    def __init__(self, model, size, device, fp16=False):
        self.model = model.to(device).eval()
        self.load_images = ImageLoader(size, device)
        self.half_precision = fp16 and device == "cuda"

    def __call__(self, paths):
        import torch

        batch = self.load_images(paths)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.half_precision):
            descriptors = self.model(batch)

        return torch.nn.functional.normalize(descriptors.float(), dim=1).cpu().numpy()


def load_finetuned_weights(model, path):
    """Load weights saved by the training command ({"state_dict": ...}) or a plain state dict."""
    import torch

    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    print(f"[model] loaded fine-tuned weights from {path}", flush=True)
    
    return model
