from __future__ import annotations

from pathlib import Path


DEFAULT_MODEL_PLAN = [
    {"model_name": "alexnet", "layers": ["features.2", "features.5", "features.12"]},
    {"model_name": "vgg16", "layers": ["features.3", "features.8", "features.22"]},
    {"model_name": "resnet50", "layers": ["layer1", "layer2", "layer3", "layer4"]},
    {"model_name": "vit_b_16", "layers": ["encoder"]},
]


def dnn_feature_file_name(session_id: str, model_name: str, layer_name: str, trained: bool = True) -> str:
    state = "pretrained" if trained else "untrained"
    safe_layer = layer_name.replace(".", "_").replace("/", "_")
    return f"session_{session_id}_{model_name}_{state}_{safe_layer}_features.npy"


def preprocess_frames_for_torch(frames: np.ndarray, image_size: int = 224):
    """Return a torch tensor for torchvision models without importing torch globally."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("torch is required for DNN feature extraction.") from exc
    import numpy as np

    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
        arr = np.repeat(arr, 3, axis=1)
    elif arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        arr = arr[..., :3].transpose(0, 3, 1, 2)
    elif arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
        arr = arr[:, :3]
    else:
        raise ValueError(f"Unsupported frame array shape: {arr.shape}")
    if np.nanmax(arr) > 1.5:
        arr = arr / 255.0
    tensor = torch.tensor(np.nan_to_num(arr), dtype=torch.float32)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]
    return (tensor - mean) / std


def planned_dnn_outputs(session_id: str, output_dir: str | Path, model_plan=None) -> list[Path]:
    output_dir = Path(output_dir)
    rows = []
    for spec in (model_plan or DEFAULT_MODEL_PLAN):
        for layer in spec["layers"]:
            rows.append(output_dir / dnn_feature_file_name(session_id, spec["model_name"], layer, trained=True))
            rows.append(output_dir / dnn_feature_file_name(session_id, spec["model_name"], layer, trained=False))
    return rows


def get_torchvision_model(model_name: str, pretrained: bool = True):
    """Load a torchvision model with weights kept optional."""
    try:
        import torchvision.models as models
    except ImportError as exc:
        raise ImportError("torchvision is required for DNN feature extraction.") from exc

    name = model_name.lower()
    if name == "alexnet":
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        model = models.alexnet(weights=weights)
    elif name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
    elif name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
    else:
        raise ValueError(f"Unsupported torchvision model: {model_name}")
    model.eval()
    return model


def resolve_module(model, layer_name: str):
    module = model
    for part in layer_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def extract_layer_features(
    model,
    frames,
    layer_names: list[str],
    batch_size: int = 32,
    device: str = "cpu",
    image_size: int = 224,
) -> dict[str, np.ndarray]:
    """Extract flattened activations from named model layers."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for DNN feature extraction.") from exc
    import numpy as np

    model = model.to(device)
    activations: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}
    hooks = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            out = output.detach().float().cpu()
            out = out.reshape(out.shape[0], -1).numpy()
            activations[name].append(out)
        return hook

    for name in layer_names:
        hooks.append(resolve_module(model, name).register_forward_hook(make_hook(name)))

    tensor = preprocess_frames_for_torch(frames, image_size=image_size)
    with torch.no_grad():
        for start in range(0, tensor.shape[0], int(batch_size)):
            batch = tensor[start:start + int(batch_size)].to(device)
            _ = model(batch)

    for hook in hooks:
        hook.remove()

    return {
        name: np.concatenate(chunks, axis=0).astype(np.float32)
        for name, chunks in activations.items()
        if chunks
    }
