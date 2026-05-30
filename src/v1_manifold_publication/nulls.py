from __future__ import annotations
import numpy as np

def shuffled_labels(y, rng=None):
    rng = np.random.default_rng(rng)
    y = np.asarray(y)
    return rng.permutation(y)

def circular_shift(y, min_shift: int = 10, rng=None):
    rng = np.random.default_rng(rng)
    y = np.asarray(y)
    n = len(y)
    if n <= 2 * min_shift:
        shift = rng.integers(1, n)
    else:
        shift = rng.integers(min_shift, n - min_shift)
    return np.roll(y, int(shift)), int(shift)

def block_permutation(y, block_size: int = 30, rng=None):
    rng = np.random.default_rng(rng)
    y = np.asarray(y)
    blocks = [np.arange(i, min(i + block_size, len(y))) for i in range(0, len(y), block_size)]
    order = rng.permutation(len(blocks))
    return y[np.concatenate([blocks[i] for i in order])]

def empirical_pvalue(observed: float, null_values, alternative: str = "greater") -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if len(null_values) == 0:
        return np.nan
    if alternative == "greater":
        count = np.sum(null_values >= observed)
    elif alternative == "less":
        count = np.sum(null_values <= observed)
    elif alternative == "two-sided":
        count = np.sum(np.abs(null_values) >= abs(observed))
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    return float((count + 1) / (len(null_values) + 1))
