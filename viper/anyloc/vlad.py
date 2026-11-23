"""VLAD: a vocabulary of cluster centres, and one global descriptor per image built from its patch features."""
import numpy as np


def fit_vocabulary(features, n_clusters, seed=0):
    """Cluster centres (n_clusters, channels) from a sample of patch features, with mini-batch k-means."""
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters, random_state=seed, n_init=3, batch_size=8192).fit(features)

    return kmeans.cluster_centers_.astype(np.float32)


def vlad(features, centres):
    """VLAD descriptor of one image: (patches, channels) -> (clusters * channels,).

    Every patch is assigned to its nearest centre, the residuals (patch - centre) are summed per centre,
    each cluster's sum is L2-normalized (intra-normalization), and the whole vector is L2-normalized.
    """
    f = features.astype(np.float32)
    squared_distance = (f ** 2).sum(1)[:, None] - 2 * f @ centres.T + (centres ** 2).sum(1)[None]
    nearest = squared_distance.argmin(1)

    residuals = np.zeros_like(centres)
    for k in range(len(centres)):
        members = nearest == k
        if members.any():
            residuals[k] = (f[members] - centres[k]).sum(0)

    residuals /= np.linalg.norm(residuals, axis=1, keepdims=True) + 1e-12
    descriptor = residuals.ravel()
    
    return descriptor / (np.linalg.norm(descriptor) + 1e-12)
