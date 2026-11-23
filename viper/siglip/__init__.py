"""SigLIP2 (Tschannen et al., 2025): a vision-language model, used for the IHM text of each item. No training.

    model.py      image and text embeddings from a SigLIP2 checkpoint (Hugging Face transformers)
    fusion.py     item prompts, per-photo standardization and the fused score
    command.py    the `siglip` command: text only, image only, and MegaLoc + text
"""
