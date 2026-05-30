from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupKFold, KFold

def contiguous_blocks(n_samples: int, n_blocks: int = 5) -> np.ndarray:
    n_blocks = max(2, min(int(n_blocks), int(n_samples)))
    groups = np.zeros(n_samples, dtype=int)
    for block_id, idx in enumerate(np.array_split(np.arange(n_samples), n_blocks)):
        groups[idx] = block_id
    return groups

def group_kfold_indices(groups, n_splits: int | None = None):
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if n_splits is None:
        n_splits = len(unique)
    n_splits = max(2, min(int(n_splits), len(unique)))
    cv = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    yield from cv.split(dummy, dummy, groups=groups)

def kfold_indices(n_samples: int, n_splits: int = 5, random_state: int = 42, shuffle: bool = True):
    cv = KFold(n_splits=max(2, min(n_splits, n_samples)), shuffle=shuffle, random_state=random_state if shuffle else None)
    dummy = np.zeros(n_samples)
    yield from cv.split(dummy)

def leave_one_session_groups(session_ids):
    session_ids = np.asarray(session_ids)
    for sess in np.unique(session_ids):
        test = np.where(session_ids == sess)[0]
        train = np.where(session_ids != sess)[0]
        if len(train) and len(test):
            yield train, test

def leave_one_mouse_groups(mouse_ids):
    mouse_ids = np.asarray(mouse_ids)
    for mouse in np.unique(mouse_ids):
        test = np.where(mouse_ids == mouse)[0]
        train = np.where(mouse_ids != mouse)[0]
        if len(train) and len(test):
            yield train, test
