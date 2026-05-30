from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path

def bar_with_points(df, x, y, title, ylabel=None, xlabel=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    order = df.groupby(x)[y].mean().sort_values(ascending=False).index.tolist()
    means = df.groupby(x)[y].mean().loc[order]
    sems = df.groupby(x)[y].sem().loc[order]
    ax.bar(range(len(order)), means.values, yerr=sems.values)
    for i, key in enumerate(order):
        vals = df.loc[df[x] == key, y].to_numpy(dtype=float)
        jitter = np.linspace(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=20, alpha=0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel(ylabel or y)
    ax.set_xlabel(xlabel or x)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def heatmap(df, index, columns, values, title, cmap=None, annotate=True):
    table = df.pivot(index=index, columns=columns, values=values)
    fig, ax = plt.subplots(figsize=(max(6, table.shape[1] * 1.1), max(3, table.shape[0] * 0.6)))
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index)
    if annotate:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                val = table.to_numpy()[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(values)
    fig.tight_layout()
    return fig
