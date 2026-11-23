"""Multi-similarity loss (Wang et al., 2019) with explicit positive and negative masks."""


def multi_similarity_loss(embeddings, pos_mask, neg_mask, alpha=1.0, beta=50.0, base=0.0, margin=0.1):
    """Mean loss over the anchors that have an informative positive and an informative negative.

    embeddings  (N, D) L2-normalized descriptors of all images in the batch
    pos_mask    (N, N) bool, True where the pair shows the same place
    neg_mask    (N, N) bool, True where the pair shows different places; pairs in neither mask are ignored

    Mining: a positive is informative if it is less similar than the hardest negative plus the margin, and a
    negative if it is more similar than the hardest positive minus the margin. Returns None if no anchor in
    the batch has both, so the batch can be skipped.
    """
    import torch

    sim = embeddings @ embeddings.T
    hardest_pos = sim.masked_fill(~pos_mask, 1e4).min(1, keepdim=True).values
    hardest_neg = sim.masked_fill(~neg_mask, -1e4).max(1, keepdim=True).values
    informative_pos = pos_mask & (sim - margin < hardest_neg)
    informative_neg = neg_mask & (sim + margin > hardest_pos)

    valid = informative_pos.any(1) & informative_neg.any(1)
    if not valid.any():
        return None

    zero = torch.zeros(len(sim), 1, device=sim.device, dtype=sim.dtype)        # the "1 +" inside the log
    pos_term = torch.logsumexp(torch.cat([zero, (-alpha * (sim - base)).masked_fill(~informative_pos, float("-inf"))], 1), 1) / alpha
    neg_term = torch.logsumexp(torch.cat([zero, (beta * (sim - base)).masked_fill(~informative_neg, float("-inf"))], 1), 1) / beta
    
    return (pos_term + neg_term)[valid].mean()
