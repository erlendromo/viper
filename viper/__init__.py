"""ViPeR: visual place recognition of IHM inspection photographs in a camera scan of a ship.

Subpackages
    viper.utils    data loading, ground truth, metrics, evaluation and other shared code
    viper.anyloc   AnyLoc (DINOv2 value features + VLAD), the core paper
    viper.salad    SALAD, pretrained place recognition model, optionally with fine-tuned weights
    viper.megaloc  MegaLoc, pretrained place recognition model, optionally with fine-tuned weights
    viper.training fine-tuning on the scan with pairs from the camera poses
    viper.siglip   SigLIP2 text and image embeddings, and the fusion with MegaLoc

Every command is run through main.py in the repository root.
"""
