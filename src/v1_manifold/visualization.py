from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
    })


def save_figure(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_population_heatmap(R: np.ndarray, title: str = "Population response matrix"):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    im = ax.imshow(R.T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Movie frame")
    ax.set_ylabel("Cell")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="normalized ΔF/F")
    return fig


def plot_embedding_2d(
    Z: np.ndarray,
    color=None,
    title: str = "Latent embedding",
    colorbar_label: str = "Movie frame",
    point_size: float = 12,
    alpha: float = 1.0,
):
    """Plot a 2D latent embedding with an explicit colorbar label."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if color is None:
        color = np.arange(Z.shape[0])
    sc = ax.scatter(
        Z[:, 0],
        Z[:, 1],
        c=color,
        s=point_size,
        alpha=alpha,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=colorbar_label)
    return fig


def plot_metric_bar(df: pd.DataFrame, x: str, y: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered = df.sort_values(y, ascending=False)
    ax.bar(ordered[x].astype(str), ordered[y])
    ax.set_ylabel(y.replace("_", " "))
    ax.set_xlabel(x.replace("_", " "))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    return fig


def plot_geometry_timeseries(values: np.ndarray, ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(np.arange(len(values)), values, linewidth=1.3)
    ax.set_xlabel("Movie frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels=None, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45)
        ax.set_yticks(np.arange(len(labels)), labels)
    fig.colorbar(im, ax=ax)
    return fig


def plot_cka_layer_curve(df: pd.DataFrame, title: str = "DNN-to-V1 alignment"):
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, g in df.groupby("model"):
        ax.plot(np.arange(len(g)), g["linear_cka"], marker="o", label=model)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Linear CKA")
    ax.set_title(title)
    ax.legend(frameon=False)
    return fig
