from __future__ import annotations

from pathlib import Path
import importlib
import importlib.metadata as importlib_metadata
import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap


def fit_pca(
    X: np.ndarray,
    n_components: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA, float]:
    X = np.asarray(X)
    n_components = int(min(n_components, X.shape[0], X.shape[1]))

    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(X)

    lambdas = pca.explained_variance_
    participation_ratio = float(
        (lambdas.sum() ** 2) / (np.sum(lambdas**2) + 1e-12)
    )

    return scores.astype(np.float32), pca, participation_ratio


def fit_umap(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP is not installed. Install it with: python -m pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )

    return reducer.fit_transform(X).astype(np.float32)


def fit_isomap(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 10,
) -> np.ndarray:
    reducer = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    return reducer.fit_transform(X).astype(np.float32)


def _package_version(package_name: str) -> str:
    """Return an installed package version without failing the analysis."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def _import_cebra_with_helpful_error():
    """Import CEBRA and convert common dependency failures into clear messages.

    CEBRA may fail during import if the installed NumPy version is too old for the
    installed CEBRA build. In that case, the traceback can look like:

        AttributeError: module 'numpy' has no attribute 'dtypes'

    I catch that here so notebook 07 explains the actual environment issue instead
    of crashing with a cryptic CEBRA-internal traceback.
    """
    numpy_version = np.__version__
    cebra_version = _package_version("cebra")

    if not hasattr(np, "dtypes"):
        raise ImportError(
            "CEBRA cannot be imported because the current NumPy installation does "
            "not expose np.dtypes, which the installed CEBRA package expects.\n\n"
            f"Detected versions: numpy={numpy_version}, cebra={cebra_version}.\n\n"
            "Fix this environment with:\n"
            '    python -m pip install --upgrade "numpy>=1.26,<2" cebra\n\n'
            "Then restart the VS Code/Jupyter kernel before rerunning notebook 07. "
            "If AllenSDK becomes unstable after upgrading NumPy, keep AllenSDK in "
            "the current environment and run notebook 07 in a separate CEBRA-focused "
            "environment using the already processed data in data/interim/ and "
            "data/processed/."
        )

    try:
        return importlib.import_module("cebra")

    except AttributeError as exc:
        message = str(exc)

        if "numpy" in message and "dtypes" in message:
            raise ImportError(
                "CEBRA failed to import because it expects NumPy to provide "
                "np.dtypes, but the active environment does not provide it.\n\n"
                f"Detected versions: numpy={numpy_version}, cebra={cebra_version}.\n\n"
                "Run:\n"
                '    python -m pip install --upgrade "numpy>=1.26,<2" cebra\n\n'
                "Then restart the notebook kernel and rerun notebook 07."
            ) from exc

        raise ImportError(
            "CEBRA failed during import because of an AttributeError inside one of "
            "its dependencies.\n\n"
            f"Detected versions: numpy={numpy_version}, cebra={cebra_version}.\n"
            f"Original error: {message}"
        ) from exc

    except ImportError as exc:
        raise ImportError(
            "CEBRA is not available in this environment.\n\n"
            f"Detected versions: numpy={numpy_version}, cebra={cebra_version}.\n\n"
            "Install or repair it with:\n"
            '    python -m pip install --upgrade "numpy>=1.26,<2" cebra'
        ) from exc


def fit_cebra_embedding(
    X: np.ndarray,
    labels: np.ndarray | None = None,
    output_dimension: int = 3,
    model_architecture: str = "offset10-model",
    conditional: str = "time_delta",
    max_iterations: int = 2000,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    distance: str = "cosine",
    device: str = "cpu",
    random_state: int = 42,
):
    """Fit a CEBRA embedding with a scikit-learn-style API.

    This function intentionally performs dependency validation before constructing
    the CEBRA model, because CEBRA/NumPy/PyTorch version mismatches can otherwise
    produce hard-to-interpret import tracebacks.
    """
    cebra = _import_cebra_with_helpful_error()

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(
            f"CEBRA expects a 2D matrix [samples, features], but X has shape {X.shape}."
        )

    if labels is None:
        labels = np.arange(X.shape[0])[:, None]
    else:
        labels = np.asarray(labels)

    if labels.shape[0] != X.shape[0]:
        raise ValueError(
            f"CEBRA label/sample mismatch: X has {X.shape[0]} samples, "
            f"but labels has {labels.shape[0]} rows."
        )

    np.random.seed(random_state)

    model = cebra.CEBRA(
        model_architecture=model_architecture,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dimension=output_dimension,
        max_iterations=max_iterations,
        conditional=conditional,
        distance=distance,
        device=device,
        verbose=True,
        num_hidden_units=64,
        time_offsets=10,
    )

    embedding = model.fit_transform(X, labels)
    return np.asarray(embedding, dtype=np.float32), model


def save_embedding_npz(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def save_model(model, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, path)

    except Exception as exc:
        warnings.warn(
            "The model could not be saved with joblib. This can happen for some "
            f"third-party model objects. Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        raise

    return path
