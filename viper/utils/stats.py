"""Paired statistical tests for comparing two models on the same items."""
from math import comb

import numpy as np


def mcnemar_exact(only_a, only_b):
    """Two-sided exact McNemar test for 0/1 outcomes.

    only_a: items only model A gets right; only_b: items only model B gets right. Items both get right or
    both get wrong carry no information about the difference and are not used.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n)


def permutation_test(x, y, n_perm=20000, seed=0):
    """Two-sided paired sign-flip permutation test on the mean difference y - x (for scores that are not 0/1)."""
    diff = y - x
    observed = abs(diff.mean())

    if observed == 0:
        return 1.0
    
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], (n_perm, len(diff)))

    return float(((np.abs((signs * diff).mean(1)) >= observed - 1e-12).sum() + 1) / (n_perm + 1))


def bootstrap_difference(x, y, n_boot=10000, seed=0):
    """Mean of y - x with a paired 95% bootstrap interval: (difference, lower, upper)."""
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(x), (n_boot, len(x)))
    means = (y[resamples] - x[resamples]).mean(1)
    
    return float((y - x).mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
