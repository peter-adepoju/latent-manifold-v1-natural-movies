from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.fft import fft2, fftshift

EPS = 1e-12

def to_grayscale(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=float)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    arr = arr - np.nanmin(arr)
    denom = np.nanmax(arr) + EPS
    return arr / denom

def resize_frame(frame: np.ndarray, shape=(96, 160)) -> np.ndarray:
    arr = to_grayscale(frame)
    zoom = (shape[0] / arr.shape[0], shape[1] / arr.shape[1])
    return ndimage.zoom(arr, zoom, order=1)

def rms_contrast(frame: np.ndarray) -> float:
    f = to_grayscale(frame)
    return float(np.nanstd(f) / (np.nanmean(f) + EPS))

def local_contrast_features(frame: np.ndarray, window_sigma: float = 3.0) -> dict:
    """Summarize local contrast, not just whole-frame luminance variation."""
    f = to_grayscale(frame)
    local_mean = ndimage.gaussian_filter(f, sigma=window_sigma)
    local_sq = ndimage.gaussian_filter(f * f, sigma=window_sigma)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    return {
        "local_contrast_mean": float(np.nanmean(local_std)),
        "local_contrast_std": float(np.nanstd(local_std)),
        "local_contrast_p95": float(np.nanpercentile(local_std, 95)),
    }

def edge_features(frame: np.ndarray) -> dict:
    """Compute simple edge-density features from image gradients."""
    f = to_grayscale(frame)
    gy = ndimage.sobel(f, axis=0)
    gx = ndimage.sobel(f, axis=1)
    mag = np.sqrt(gx**2 + gy**2)
    threshold = np.nanpercentile(mag, 90)
    return {
        "edge_density_p90": float(np.nanmean(mag >= threshold)),
        "edge_magnitude_mean": float(np.nanmean(mag)),
        "edge_magnitude_p95": float(np.nanpercentile(mag, 95)),
    }

def fourier_power_features(frame: np.ndarray) -> dict:
    f = to_grayscale(frame)
    F = np.abs(fftshift(fft2(f))) ** 2
    h, w = F.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr_norm = rr / (rr.max() + EPS)
    power = F / (F.sum() + EPS)
    centroid = float(np.sum(rr_norm * power))
    low = float(power[rr_norm < 0.15].sum())
    mid = float(power[(rr_norm >= 0.15) & (rr_norm < 0.35)].sum())
    high = float(power[rr_norm >= 0.35].sum())
    radial_bins = np.linspace(0.02, 1.0, 20)
    radial_power, centers = [], []
    for a, b in zip(radial_bins[:-1], radial_bins[1:]):
        mask = (rr_norm >= a) & (rr_norm < b)
        if mask.any():
            radial_power.append(power[mask].mean() + EPS)
            centers.append((a + b) / 2)
    if len(radial_power) >= 3:
        slope = np.polyfit(np.log(centers), np.log(radial_power), deg=1)[0]
    else:
        slope = np.nan
    return {
        "spatial_frequency_centroid": centroid,
        "low_frequency_power": low,
        "mid_frequency_power": mid,
        "high_frequency_power": high,
        "fourier_power_slope": float(slope),
        "total_spectral_power": float(F.sum()),
    }

def orientation_energy_features(frame: np.ndarray, orientations_deg=(0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5)) -> dict:
    f = to_grayscale(frame)
    gy, gx = np.gradient(f)
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    energies = []
    for ori in orientations_deg:
        dist = np.minimum(np.abs(ang - ori), 180 - np.abs(ang - ori))
        weight = np.exp(-(dist**2) / (2 * 15**2))
        energies.append(float(np.sum(mag * weight)))
    energies = np.asarray(energies, dtype=float)
    p = energies / (energies.sum() + EPS)
    dominant = int(np.argmax(p))
    selectivity = float((p.max() - np.median(p)) / (p.max() + EPS))
    out = {f"orientation_energy_{i}": float(v) for i, v in enumerate(p)}
    out.update({
        "dominant_orientation_bin": dominant,
        "dominant_orientation_angle_deg": float(orientations_deg[dominant]),
        "orientation_selectivity": selectivity,
        "orientation_entropy": float(-np.sum(p * np.log(p + EPS))),
    })
    return out

def gabor_energy_bank_features(
    frame: np.ndarray,
    frequencies=(0.04, 0.08, 0.16, 0.24),
    orientations_deg=(0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5),
    kernel_size: int = 31,
) -> dict:
    """Compute a compact V1-like Gabor energy bank from one frame.

    This is intentionally dependency-light. It is slower than a dedicated image
    library but good enough for publication feature extraction over short Allen
    natural movies.
    """
    f = to_grayscale(frame)
    f = (f - np.nanmean(f)) / (np.nanstd(f) + EPS)
    half = int(kernel_size) // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    rows = {}
    total = 0.0
    for freq in frequencies:
        sigma = max(2.0, 0.6 / (float(freq) + EPS))
        envelope = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        for ori in orientations_deg:
            theta = np.deg2rad(float(ori))
            x_theta = xx * np.cos(theta) + yy * np.sin(theta)
            carrier_cos = np.cos(2 * np.pi * float(freq) * x_theta)
            carrier_sin = np.sin(2 * np.pi * float(freq) * x_theta)
            even = envelope * carrier_cos
            odd = envelope * carrier_sin
            even = even - even.mean()
            odd = odd - odd.mean()
            resp_even = ndimage.convolve(f, even, mode="reflect")
            resp_odd = ndimage.convolve(f, odd, mode="reflect")
            energy = float(np.nanmean(resp_even**2 + resp_odd**2))
            key = f"gabor_energy_f{str(freq).replace('.', 'p')}_ori{str(ori).replace('.', 'p')}"
            rows[key] = energy
            total += energy
    for key in list(rows):
        rows[f"{key}_fraction"] = float(rows[key] / (total + EPS))
    rows["gabor_total_energy"] = float(total)
    return rows

def temporal_features(frames: np.ndarray) -> pd.DataFrame:
    frames = np.asarray([to_grayscale(f) for f in frames], dtype=float)
    rows = []
    prev = None
    for i, f in enumerate(frames):
        if prev is None:
            diff = np.zeros_like(f)
        else:
            diff = f - prev
        gy, gx = np.gradient(f)
        motion_x = float(np.nanmean(diff * gx))
        motion_y = float(np.nanmean(diff * gy))
        motion_norm = float(np.sqrt(motion_x**2 + motion_y**2))
        temporal_contrast = float(np.nanstd(diff) / (np.nanmean(np.abs(f)) + EPS))
        rows.append({
            "movie_frame": i,
            "temporal_absdiff_mean": float(np.mean(np.abs(diff))),
            "temporal_absdiff_rms": float(np.sqrt(np.mean(diff**2))),
            "temporal_absdiff_p95": float(np.percentile(np.abs(diff), 95)),
            "temporal_contrast": temporal_contrast,
            "motion_energy": float(np.nanmean(diff**2)),
            "motion_proxy_x": motion_x,
            "motion_proxy_y": motion_y,
            "motion_proxy_magnitude": motion_norm,
            "motion_proxy_direction_cos": float(motion_x / (motion_norm + EPS)),
            "motion_proxy_direction_sin": float(motion_y / (motion_norm + EPS)),
        })
        prev = f
    return pd.DataFrame(rows)

def extract_frame_feature_table(
    frames: np.ndarray,
    resize_shape=(96, 160),
    include_gabor: bool = False,
    gabor_frequencies=(0.04, 0.08, 0.16, 0.24),
    gabor_orientations_deg=(0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5),
) -> pd.DataFrame:
    rows = []
    resized = []
    for i, frame in enumerate(frames):
        f = resize_frame(frame, resize_shape)
        resized.append(f)
        row = {
            "movie_frame": i,
            "luminance_mean": float(np.mean(f)),
            "luminance_std": float(np.std(f)),
            "rms_contrast": rms_contrast(f),
        }
        row.update(fourier_power_features(f))
        row.update(orientation_energy_features(f))
        row.update(local_contrast_features(f))
        row.update(edge_features(f))
        if include_gabor:
            row.update(gabor_energy_bank_features(
                f,
                frequencies=gabor_frequencies,
                orientations_deg=gabor_orientations_deg,
            ))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.merge(temporal_features(np.asarray(resized)), on="movie_frame", how="left")

def add_scene_change_score(features: pd.DataFrame, diff_col="temporal_absdiff_rms") -> pd.DataFrame:
    out = features.copy()
    x = out[diff_col].to_numpy(dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + EPS
    out["scene_change_score"] = (x - med) / (1.4826 * mad)
    out["scene_change_candidate"] = out["scene_change_score"] > np.nanpercentile(out["scene_change_score"], 95)
    return out

def feature_hierarchy_columns(features: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Return publication feature groups from low-level to model-level features."""
    groups = {
        "level1_luminance_contrast": [
            "luminance_mean",
            "luminance_std",
            "rms_contrast",
            "local_contrast_mean",
            "local_contrast_std",
            "local_contrast_p95",
        ],
        "level2_spatial_orientation": [
            "spatial_frequency_centroid",
            "low_frequency_power",
            "mid_frequency_power",
            "high_frequency_power",
            "fourier_power_slope",
            "orientation_selectivity",
            "orientation_entropy",
            "edge_density_p90",
            "edge_magnitude_mean",
            "edge_magnitude_p95",
        ],
        "level3_temporal_motion": [
            "temporal_absdiff_mean",
            "temporal_absdiff_rms",
            "temporal_absdiff_p95",
            "temporal_contrast",
            "motion_energy",
            "motion_proxy_magnitude",
            "motion_proxy_direction_cos",
            "motion_proxy_direction_sin",
            "scene_change_score",
        ],
        "level4_v1_like_gabor": [],
        "level5_deep_model": [],
    }
    if features is not None:
        columns = list(features.columns)
        groups["level4_v1_like_gabor"] = [c for c in columns if c.startswith("gabor_energy_")]
        groups["level5_deep_model"] = [c for c in columns if c.startswith("dnn_") or c.startswith("vit_")]
        groups = {name: [c for c in cols if c in columns] for name, cols in groups.items()}
    return groups
