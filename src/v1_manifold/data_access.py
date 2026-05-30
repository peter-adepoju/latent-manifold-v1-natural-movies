from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


CRE_LINE_LAYER_MAP = {
    "Cux2-CreERT2": "L2/3",
    "Cux2-CreERT2;Camk2a-tTA;Ai93": "L2/3",
    "Rorb-IRES2-Cre": "L4",
    "Rbp4-Cre_KL100": "L5",
    "Ntsr1-Cre_GN220": "L6",
    "Emx1-IRES-Cre": "pan-excitatory",
}


def get_boc(manifest_file: str | Path):
    """Create the Allen BrainObservatoryCache object only when AllenSDK is installed."""
    try:
        from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    except ImportError as exc:
        raise ImportError(
            "AllenSDK is required for real data access. Install it with `pip install allensdk==2.16.2` "
            "or create the conda environment from environment.yml."
        ) from exc
    manifest_file = Path(manifest_file)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    return BrainObservatoryCache(manifest_file=str(manifest_file))


def query_v1_natural_movie_experiments(boc, cfg: dict[str, Any]) -> pd.DataFrame:
    """Query V1 experiments with natural movie stimuli and attach layer labels."""
    allen_cfg = cfg.get("allen", {})
    experiments = boc.get_ophys_experiments(
        targeted_structures=allen_cfg.get("targeted_structures", ["VISp"]),
        stimuli=allen_cfg.get("stimuli", ["natural_movie_one"]),
    )
    df = pd.DataFrame(experiments)
    if df.empty:
        return df

    cre_lines = allen_cfg.get("cre_lines") or []
    if cre_lines and "cre_line" in df.columns:
        df = df[df["cre_line"].isin(cre_lines)].copy()

    if "cre_line" in df.columns:
        df["putative_layer"] = df["cre_line"].map(CRE_LINE_LAYER_MAP).fillna("unknown")
    else:
        df["putative_layer"] = "unknown"

    max_experiments = int(allen_cfg.get("max_experiments", 0) or 0)
    if max_experiments > 0:
        sort_cols = [c for c in ["cre_line", "imaging_depth", "id"] if c in df.columns]
        df = df.sort_values(sort_cols).head(max_experiments).reset_index(drop=True)
    return df


def save_experiment_catalog(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def download_selected_nwb_files(boc, experiments: pd.DataFrame, download: bool = False) -> list[int]:
    """Download selected NWB files through AllenSDK when requested.

    I keep the default `download=False` because this repository should not force a large data download.
    Calling `get_ophys_experiment_data` downloads/caches the NWB through the AllenSDK manifest.
    """
    downloaded: list[int] = []
    if not download:
        return downloaded
    if "id" not in experiments.columns:
        raise ValueError("Experiment catalog must contain an `id` column.")
    for experiment_id in experiments["id"].astype(int).tolist():
        _ = boc.get_ophys_experiment_data(experiment_id)
        downloaded.append(experiment_id)
    return downloaded


def get_experiment_data(boc, experiment_id: int):
    """Load one Allen ophys experiment object."""
    return boc.get_ophys_experiment_data(int(experiment_id))


def extract_dff_and_metadata(data_set) -> tuple[pd.DataFrame, Any, Any]:
    """Extract cell IDs, ΔF/F traces, and fluorescence timestamps from an Allen experiment.

    AllenSDK's ``get_dff_traces`` returns ``(timestamps, dff_traces)`` for the
    Visual Coding ophys data object. I therefore obtain cell specimen IDs from
    ``get_cell_specimen_ids`` instead of accidentally treating timestamps as
    cell metadata.
    """
    timestamps, dff = data_set.get_dff_traces()

    try:
        cell_specimen_ids = data_set.get_cell_specimen_ids()
    except Exception as exc:
        raise RuntimeError(
            "I could not obtain cell specimen IDs from the AllenSDK experiment object."
        ) from exc

    if len(cell_specimen_ids) != dff.shape[0]:
        raise ValueError(
            "Cell metadata and dF/F traces disagree: "
            f"{len(cell_specimen_ids)} cell IDs for {dff.shape[0]} dF/F rows."
        )

    cell_df = pd.DataFrame({"cell_specimen_id": cell_specimen_ids})

    try:
        specimen_table = data_set.get_cell_specimen_table()
        if isinstance(specimen_table, pd.DataFrame):
            table = specimen_table.reset_index()
            if "cell_specimen_id" in table.columns:
                cell_df = cell_df.merge(table, how="left", on="cell_specimen_id")
    except Exception:
        pass

    return cell_df, dff, timestamps


def get_stimulus_table(data_set, stimulus_name: str) -> pd.DataFrame:
    stim = data_set.get_stimulus_table(stimulus_name).copy()
    required = {"start", "end"}
    missing = required.difference(stim.columns)
    if missing:
        raise ValueError(f"Stimulus table for {stimulus_name} is missing required columns: {missing}")
    return stim


def get_stimulus_template(data_set, stimulus_name: str):
    """Try to load the natural movie template from the NWB-backed Allen object."""
    for attr in ["get_stimulus_template", "get_stimulus_template_array"]:
        if hasattr(data_set, attr):
            try:
                return getattr(data_set, attr)(stimulus_name)
            except Exception:
                continue
    try:
        return data_set.get_stimulus_template(stimulus_name)
    except Exception:
        return None
