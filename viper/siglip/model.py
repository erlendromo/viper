"""Image and text embeddings from SigLIP2. Needs: pip install transformers pillow"""

CHECKPOINT = "google/siglip2-so400m-patch14-384"     # smaller and faster: google/siglip2-base-patch16-224
TEXT_LENGTH = 64                                     # tokens; SigLIP2 is trained with padding to this length


class SigLIP:
    """embed_images(paths) and embed_texts(strings) -> (B, D) float32 arrays, L2-normalized.

    Images and texts are embedded into the same space, so a text can be compared with an image.
    """

    def __init__(self, checkpoint, device):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.model = AutoModel.from_pretrained(checkpoint).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(checkpoint)

    def embed_images(self, paths):
        import torch
        from PIL import Image

        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            return self._normalized(self.model.get_image_features(**inputs))

    def embed_texts(self, texts):
        import torch

        inputs = self.processor(text=[t.lower() for t in texts], padding="max_length", max_length=TEXT_LENGTH,
                                truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            return self._normalized(self.model.get_text_features(**inputs))

    @staticmethod
    def _normalized(output):
        """Older transformers versions return a tensor, newer ones an output object with pooler_output."""
        import torch

        features = output if torch.is_tensor(output) else output.pooler_output

        return torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
