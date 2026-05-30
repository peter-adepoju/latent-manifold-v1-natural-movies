from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.preprocessing import StandardScaler


def _as_movie_array(movie_template: np.ndarray) -> np.ndarray:
    """Normalize Allen natural movie templates to [frames, height, width]."""
    frames = np.asarray(movie_template)
    if frames.ndim == 4:
        # Common possibilities: [frames, height, width, channels] or [frames, channels, height, width].
        if frames.shape[-1] in (1, 3, 4):
            frames = frames[..., :3].mean(axis=-1)
        elif frames.shape[1] in (1, 3, 4):
            frames = frames[:, :3].mean(axis=1)
        else:
            frames = frames.mean(axis=-1)
    if frames.ndim != 3:
        raise ValueError(f"Movie template must be [frames, height, width], got {frames.shape}")
    return frames


def _to_grayscale_frame(frame: np.ndarray) -> np.ndarray:
    """Convert one frame to finite float32 grayscale in approximately [0, 1]."""
    image = np.asarray(frame)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if finite.any() and np.nanmax(image[finite]) > 1.5:
        image = image / 255.0
    return np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)


def _downsample_frame(frame: np.ndarray, factor: int = 4) -> np.ndarray:
    """Downsample a frame by block averaging for efficient spectral features."""
    if factor <= 1:
        return frame
    h, w = frame.shape
    h2, w2 = h // factor, w // factor
    if h2 < 4 or w2 < 4:
        return frame
    cropped = frame[: h2 * factor, : w2 * factor]
    return cropped.reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def _spectral_features_for_frame(
    frame: np.ndarray,
    n_orientation_bins: int = 8,
    n_spatial_frequency_bins: int = 5,
    downsample_factor: int = 4,
) -> dict[str, float | int | bool]:
    """Compute deterministic visual features from one natural movie frame.

    I use Fourier power to estimate dominant orientation and spatial-frequency
    content, and simple image statistics for luminance and contrast. These
    quantities are real stimulus-derived features, not invented labels.
    """
    raw = _to_grayscale_frame(frame)
    image = _downsample_frame(raw, factor=downsample_factor)
    image = image - float(np.mean(image))

    h, w = image.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(image * window))
    power = np.abs(spectrum) ** 2

    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    FX, FY = np.meshgrid(fx, fy)
    radius = np.sqrt(FX**2 + FY**2)
    angle = np.mod(np.arctan2(FY, FX), np.pi)

    valid = radius > np.percentile(radius, 5)
    power_valid = power[valid]
    radius_valid = radius[valid]
    angle_valid = angle[valid]
    total_power = float(power_valid.sum() + 1e-12)

    orientation_edges = np.linspace(0.0, np.pi, int(n_orientation_bins) + 1)
    orientation_energy = np.zeros(int(n_orientation_bins), dtype=float)
    for k in range(int(n_orientation_bins)):
        mask = (angle_valid >= orientation_edges[k]) & (angle_valid < orientation_edges[k + 1])
        orientation_energy[k] = float(power_valid[mask].sum())

    dominant_orientation_bin = int(np.argmax(orientation_energy))
    dominant_orientation_angle = float(
        0.5 * (orientation_edges[dominant_orientation_bin] + orientation_edges[dominant_orientation_bin + 1])
    )
    orientation_selectivity = float(orientation_energy.max() / (orientation_energy.sum() + 1e-12))

    # Fixed radial bins make spatial-frequency categories comparable across frames.
    positive_radius = radius_valid[radius_valid > 0]
    max_radius = float(np.max(positive_radius)) if len(positive_radius) else 0.5
    sf_edges = np.linspace(0.0, max_radius + 1e-12, int(n_spatial_frequency_bins) + 1)
    sf_energy = np.zeros(int(n_spatial_frequency_bins), dtype=float)
    for k in range(int(n_spatial_frequency_bins)):
        mask = (radius_valid >= sf_edges[k]) & (radius_valid < sf_edges[k + 1])
        sf_energy[k] = float(power_valid[mask].sum())
    spatial_frequency_bin = int(np.argmax(sf_energy))
    spatial_frequency_centroid = float(np.sum(radius_valid * power_valid) / total_power)

    luminance_mean = float(np.mean(raw))
    luminance_std = float(np.std(raw))
    rms_contrast = float(luminance_std / (abs(luminance_mean) + 1e-12))

    result: dict[str, float | int | bool] = {
        "luminance_mean": luminance_mean,
        "luminance_std": luminance_std,
        "rms_contrast": rms_contrast,
        "dominant_orientation_bin": dominant_orientation_bin,
        "dominant_orientation_angle_rad": dominant_orientation_angle,
        "orientation_selectivity": orientation_selectivity,
        "spatial_frequency_centroid": spatial_frequency_centroid,
        "spatial_frequency_bin": spatial_frequency_bin,
        "total_spectral_power": total_power,
        "fallback_labels": False,
    }
    for k, value in enumerate(orientation_energy):
        result[f"orientation_energy_bin_{k}"] = float(value / total_power)
    for k, value in enumerate(sf_energy):
        result[f"spatial_frequency_energy_bin_{k}"] = float(value / total_power)
    return result


def build_real_movie_feature_table(
    movie_template: np.ndarray,
    n_orientation_bins: int = 8,
    n_spatial_frequency_bins: int = 5,
    downsample_factor: int = 4,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """Extract real per-frame visual features from an Allen natural movie template."""
    frames = _as_movie_array(movie_template)
    if max_frames is not None:
        frames = frames[: int(max_frames)]
    rows = []
    for frame_idx in range(frames.shape[0]):
        row = _spectral_features_for_frame(
            frames[frame_idx],
            n_orientation_bins=n_orientation_bins,
            n_spatial_frequency_bins=n_spatial_frequency_bins,
            downsample_factor=downsample_factor,
        )
        row["movie_frame"] = int(frame_idx)
        rows.append(row)
    return pd.DataFrame(rows)


def assert_real_stimulus_features(features: pd.DataFrame) -> None:
    """Fail loudly if decoding would use fallback/non-visual labels."""
    required = {
        "dominant_orientation_bin",
        "spatial_frequency_bin",
        "spatial_frequency_centroid",
        "rms_contrast",
        "fallback_labels",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Frame feature table is missing real stimulus-derived columns: {sorted(missing)}")
    if features["fallback_labels"].astype(bool).any():
        raise ValueError(
            "I am refusing to run stimulus-feature decoding with fallback labels. "
            "Run notebook 04 to extract real visual features from the Allen natural movie template."
        )


def summarize_class_balance(y, target_name: str = "target") -> pd.DataFrame:
    """Summarize class counts and fractions for an imbalanced decoding target."""
    values = pd.Series(y, name=target_name)
    counts = values.value_counts(dropna=False).sort_index()
    table = counts.rename_axis(target_name).reset_index(name="n_samples")
    table["fraction"] = table["n_samples"] / max(int(table["n_samples"].sum()), 1)
    table["majority_class"] = table["n_samples"] == table["n_samples"].max()
    return table


def add_coarse_orientation_target(
    features: pd.DataFrame,
    source_col: str = "dominant_orientation_bin",
    target_col: str = "dominant_orientation_coarse",
    top_k: int = 2,
    other_label: str = "other",
) -> pd.DataFrame:
    """Add a coarse orientation target by keeping frequent orientation bins and grouping rare bins.

    Natural movie spectral-orientation labels are often highly imbalanced. I keep
    the most frequent bins as explicit classes and merge rare bins into an
    ``other`` class so that classification metrics are interpretable.
    """
    if source_col not in features.columns:
        raise KeyError(f"{source_col!r} is missing from the feature table.")

    out = features.copy()
    counts = out[source_col].value_counts().sort_values(ascending=False)
    top_bins = set(counts.head(int(top_k)).index.tolist())

    out[target_col] = out[source_col].map(
        lambda value: f"bin_{int(value)}" if value in top_bins and pd.notna(value) else other_label
    )
    out[f"{target_col}_is_other"] = out[target_col].eq(other_label)
    return out


def add_confident_orientation_target(
    features: pd.DataFrame,
    coarse_col: str = "dominant_orientation_coarse",
    confidence_col: str = "orientation_selectivity",
    target_col: str = "dominant_orientation_confident",
    confidence_quantile: float = 0.50,
    confidence_threshold: float | None = None,
    ambiguous_label: str = "ambiguous",
) -> pd.DataFrame:
    """Add an ambiguity-aware orientation target for natural movie frames.

    I use orientation selectivity as a confidence score. Frames below the
    configured threshold are labeled ``ambiguous`` rather than forcing a hard
    orientation class. This prevents noisy natural-movie frames from becoming
    overinterpreted classification labels.
    """
    if coarse_col not in features.columns:
        raise KeyError(f"{coarse_col!r} is missing. Run add_coarse_orientation_target first.")
    if confidence_col not in features.columns:
        raise KeyError(f"{confidence_col!r} is missing from the feature table.")

    out = features.copy()
    confidence = pd.to_numeric(out[confidence_col], errors="coerce")
    if confidence_threshold is None:
        confidence_threshold = float(confidence.quantile(float(confidence_quantile)))

    out["orientation_confidence"] = confidence
    out["orientation_confidence_threshold"] = float(confidence_threshold)
    out[target_col] = np.where(
        confidence >= float(confidence_threshold),
        out[coarse_col].astype(str),
        ambiguous_label,
    )
    out[f"{target_col}_is_ambiguous"] = out[target_col].eq(ambiguous_label)
    return out


def orientation_label_interpretation_table(
    features: pd.DataFrame,
    confident_col: str = "dominant_orientation_confident",
    fine_col: str = "dominant_orientation_bin",
    coarse_col: str = "dominant_orientation_coarse",
    ambiguous_label: str = "ambiguous",
) -> pd.DataFrame:
    """Summarize how I should interpret orientation labels in natural movies."""
    rows: list[dict[str, object]] = []
    if fine_col in features.columns:
        fine_counts = features[fine_col].value_counts(dropna=False)
        rows.append({
            "finding": "n_fine_orientation_classes",
            "value": int(fine_counts.shape[0]),
        })
        rows.append({
            "finding": "fine_orientation_majority_fraction",
            "value": float(fine_counts.max() / max(fine_counts.sum(), 1)),
        })
    if coarse_col in features.columns:
        coarse_counts = features[coarse_col].value_counts(dropna=False)
        rows.append({
            "finding": "n_coarse_orientation_classes",
            "value": int(coarse_counts.shape[0]),
        })
        rows.append({
            "finding": "coarse_orientation_majority_fraction",
            "value": float(coarse_counts.max() / max(coarse_counts.sum(), 1)),
        })
    if confident_col in features.columns:
        conf_counts = features[confident_col].value_counts(dropna=False)
        ambiguous_fraction = float(conf_counts.get(ambiguous_label, 0) / max(conf_counts.sum(), 1))
        non_ambiguous = conf_counts.drop(labels=[ambiguous_label], errors="ignore")
        rows.append({
            "finding": "fraction_ambiguous_orientation_frames",
            "value": ambiguous_fraction,
        })
        if not non_ambiguous.empty:
            rows.append({
                "finding": "dominant_confident_orientation_class",
                "value": str(non_ambiguous.idxmax()),
            })
            rows.append({
                "finding": "minority_confident_orientation_class",
                "value": str(non_ambiguous.idxmin()),
            })

    rows.extend([
        {
            "finding": "recommended_primary_decoding_target",
            "value": "continuous visual features",
        },
        {
            "finding": "recommended_orientation_decoding_status",
            "value": "secondary/exploratory with class balance, ambiguity, and null baselines",
        },
    ])
    return pd.DataFrame(rows)


def balanced_confident_orientation_subset(
    features: pd.DataFrame,
    target_col: str = "dominant_orientation_confident",
    frame_col: str = "movie_frame",
    classes: tuple[str, ...] | None = None,
    ambiguous_label: str = "ambiguous",
    n_per_class: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a balanced subset of non-ambiguous confident orientation frames.

    I use this for a small, cautious binary/low-class orientation decoding
    analysis after excluding ambiguous frames and downsampling majority classes.
    """
    if target_col not in features.columns:
        raise KeyError(f"{target_col!r} is missing from the feature table.")
    if frame_col not in features.columns:
        raise KeyError(f"{frame_col!r} is missing from the feature table.")

    subset = features.loc[features[target_col].astype(str) != ambiguous_label].copy()
    if classes is not None:
        classes = tuple(str(c) for c in classes)
        subset = subset.loc[subset[target_col].astype(str).isin(classes)].copy()
    counts = subset[target_col].value_counts()
    if len(counts) < 2:
        return subset.iloc[0:0].copy()
    if n_per_class is None:
        n_per_class = int(counts.min())
    n_per_class = int(max(1, min(n_per_class, counts.min())))
    balanced = (
        subset
        .groupby(target_col, group_keys=False)
        .sample(n=n_per_class, random_state=random_state)
        .sort_values(frame_col)
        .reset_index(drop=True)
    )
    return balanced


def safe_classification_target(y, min_classes: int = 2, min_count_per_class: int = 2) -> bool:
    """Return True when a class target has enough support for cross-validation."""
    counts = pd.Series(y).value_counts(dropna=False)
    if len(counts) < int(min_classes):
        return False
    return bool((counts >= int(min_count_per_class)).all())


def gabor_frame_features(movie_template: np.ndarray, n_orientation_bins: int = 8) -> pd.DataFrame:
    """Backward-compatible edge-based frame features.

    This function now marks outputs as real stimulus features because they are
    computed from pixels. New analyses should prefer build_real_movie_feature_table.
    """
    frames = _as_movie_array(movie_template)
    rows = []
    orientations = np.linspace(0, np.pi, n_orientation_bins, endpoint=False)
    for i, frame in enumerate(frames):
        img = _to_grayscale_frame(frame)
        img = (img - img.mean()) / (img.std() + 1e-8)
        gx = ndimage.sobel(img, axis=1)
        gy = ndimage.sobel(img, axis=0)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = (np.arctan2(gy, gx) + np.pi) % np.pi
        hist, _ = np.histogram(angle.ravel(), bins=np.r_[orientations, np.pi], weights=magnitude.ravel())
        fft_power = np.abs(np.fft.rfft2(img)) ** 2
        yy, xx = np.indices(fft_power.shape)
        radius = np.sqrt((yy - yy.mean()) ** 2 + xx**2)
        spatial_frequency = float(np.sum(radius * fft_power) / (np.sum(fft_power) + 1e-8))
        rows.append({
            "movie_frame": i,
            "dominant_orientation_bin": int(np.argmax(hist)),
            "orientation_selectivity": float(hist.max() / (hist.sum() + 1e-8)),
            "spatial_frequency_centroid": spatial_frequency,
            "fallback_labels": False,
        })
    df = pd.DataFrame(rows)
    df["spatial_frequency_bin"] = pd.qcut(
        df["spatial_frequency_centroid"].rank(method="first"),
        q=min(5, len(df)),
        labels=False,
        duplicates="drop",
    ).astype(int)
    return df


def fallback_frame_labels(n_frames: int = 900, n_orientation_bins: int = 8, n_sf_bins: int = 5) -> pd.DataFrame:
    """Create non-visual frame labels only for explicit smoke tests.

    These labels are marked as fallback labels and must not be used for real
    decoding, figures, tables, or claims.
    """
    frame = np.arange(n_frames)
    return pd.DataFrame({
        "movie_frame": frame,
        "dominant_orientation_bin": frame % n_orientation_bins,
        "orientation_selectivity": np.nan,
        "rms_contrast": np.nan,
        "spatial_frequency_centroid": np.nan,
        "spatial_frequency_bin": frame % n_sf_bins,
        "fallback_labels": True,
    })


def build_frame_feature_table(population_matrix: np.ndarray, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a tidy frame-level table with neural population summary statistics."""
    X = np.asarray(population_matrix)
    if X.ndim != 2:
        raise ValueError("Population matrix must be [frames, cells].")
    df = pd.DataFrame({
        "movie_frame": np.arange(X.shape[0]),
        "population_mean": X.mean(axis=1),
        "population_std": X.std(axis=1),
        "population_l2_norm": np.linalg.norm(X, axis=1),
        "n_cells": X.shape[1],
    })
    if labels is not None:
        df = df.merge(labels, on="movie_frame", how="left")
    return df


def make_single_trial_design_matrix(trial_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, repeat labels, and movie-frame labels from [repeat, cell, frame] tensor."""
    tensor = np.asarray(trial_tensor)
    if tensor.ndim != 3:
        raise ValueError("trial_tensor must have shape [repeat, cell, frame].")
    n_repeats, n_cells, n_frames = tensor.shape
    X = tensor.transpose(0, 2, 1).reshape(n_repeats * n_frames, n_cells)
    repeat = np.repeat(np.arange(n_repeats), n_frames)
    frame = np.tile(np.arange(n_frames), n_repeats)
    return X.astype(np.float32), repeat, frame


def standardize_matrix(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xz = scaler.fit_transform(np.asarray(X))
    return Xz.astype(np.float32), scaler
