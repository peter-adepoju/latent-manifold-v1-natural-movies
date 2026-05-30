from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .geometry import rdm


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute unbiased-enough centered linear CKA for same-sample matrices."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / (norm_x * norm_y + 1e-12))


def rsa_from_features(X: np.ndarray, Y: np.ndarray, metric: str = "correlation") -> float:
    rx = rdm(X, metric=metric)
    ry = rdm(Y, metric=metric)
    iu = np.triu_indices_from(rx, k=1)
    return float(spearmanr(rx[iu], ry[iu]).correlation)


def preprocess_movie_frames_for_torch(frames: np.ndarray, image_size: int = 224):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required for DNN frame preprocessing.") from exc
    arr = np.asarray(frames)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    elif arr.ndim == 4 and arr.shape[-1] in [1, 3]:
        arr = np.moveaxis(arr, -1, 1)
    else:
        raise ValueError(f"Expected frames [N,H,W] or [N,H,W,C], got {arr.shape}")
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 3, axis=1)
    x = torch.tensor(arr)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def get_torchvision_model(name: str, pretrained: bool = True):
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError("torchvision is required for DNN alignment.") from exc
    name = name.lower()
    if name == "alexnet":
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        return models.alexnet(weights=weights), ["features.2", "features.5", "features.8", "features.10", "classifier.1", "classifier.4"]
    if name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        return models.vgg16(weights=weights), ["features.4", "features.9", "features.16", "features.23", "features.30", "classifier.0", "classifier.3"]
    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights), ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    raise ValueError(f"Unsupported torchvision model: {name}")


def extract_layer_activations(model_name: str, frames: np.ndarray, batch_size: int = 32, pretrained: bool = True, device: str = "cpu") -> dict[str, np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for DNN activation extraction.") from exc
    model, layer_names = get_torchvision_model(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tensors = preprocess_movie_frames_for_torch(frames).to(device)
    activations: OrderedDict[str, list[np.ndarray]] = OrderedDict((name, []) for name in layer_names)

    module_map = dict(model.named_modules())
    hooks = []
    for layer_name in layer_names:
        if layer_name not in module_map:
            continue
        def _hook(module, inp, out, key=layer_name):
            y = out.detach()
            if y.ndim > 2:
                y = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(y, 1), 1)
            activations[key].append(y.cpu().numpy())
        hooks.append(module_map[layer_name].register_forward_hook(_hook))

    with torch.no_grad():
        for i in range(0, tensors.shape[0], batch_size):
            _ = model(tensors[i:i + batch_size])
    for h in hooks:
        h.remove()
    return {k: np.vstack(v) for k, v in activations.items() if len(v) > 0}


def compare_neural_to_dnn(neural_X: np.ndarray, dnn_features: dict[str, np.ndarray], model_name: str) -> pd.DataFrame:
    rows = []
    for layer, feat in dnn_features.items():
        n = min(neural_X.shape[0], feat.shape[0])
        rows.append({
            "model": model_name,
            "layer": layer,
            "n_frames": n,
            "linear_cka": linear_cka(neural_X[:n], feat[:n]),
            "rsa_spearman": rsa_from_features(neural_X[:n], feat[:n]),
        })
    return pd.DataFrame(rows)
