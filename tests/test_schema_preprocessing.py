import numpy as np
import pandas as pd

from v1_manifold.schema import validate_dff_traces, validate_trial_tensor
from v1_manifold.preprocessing import quality_filter_cells, build_trial_tensor, repeat_average_population_matrix, zscore_cells


def test_validate_dff_traces_accepts_cells_by_time():
    result = validate_dff_traces(np.ones((5, 100)))
    assert result["n_cells"] == 5
    assert result["n_timepoints"] == 100


def test_quality_filter_cells_returns_boolean_mask():
    rng = np.random.default_rng(0)
    dff = rng.normal(0.2, 0.1, size=(4, 200))
    keep, qc = quality_filter_cells(dff, max_nan_fraction=0.2, min_snr=0.1, min_mean_dff=-1.0)
    assert keep.dtype == bool
    assert len(qc) == 4
    assert "snr" in qc.columns


def test_build_trial_tensor_shape():
    rng = np.random.default_rng(0)
    dff = zscore_cells(rng.normal(0.2, 0.1, size=(3, 50)))
    stim = pd.DataFrame({
        "start": np.arange(0, 40, 2),
        "end": np.arange(1, 41, 2),
        "frame": np.tile(np.arange(10), 2),
    })
    tensor, stim_aug = build_trial_tensor(dff, stim, expected_movie_frames=10, max_repeats=2)
    summary = validate_trial_tensor(tensor)
    assert summary["n_repeats"] == 2
    assert summary["n_cells"] == 3
    assert summary["n_movie_frames"] == 10
    R = repeat_average_population_matrix(tensor)
    assert R.shape == (10, 3)



def test_extract_dff_and_metadata_uses_cell_ids_not_timestamps():
    from v1_manifold.data_access import extract_dff_and_metadata

    class MockAllenDataSet:
        def get_dff_traces(self):
            timestamps = np.arange(100) / 30.0
            dff = np.ones((4, 100))
            return timestamps, dff

        def get_cell_specimen_ids(self):
            return np.array([11, 22, 33, 44])

        def get_cell_specimen_table(self):
            return pd.DataFrame({
                "cell_specimen_id": [11, 22, 33, 44],
                "quality": ["ok", "ok", "ok", "ok"],
            })

    cell_meta, dff, timestamps = extract_dff_and_metadata(MockAllenDataSet())
    assert len(cell_meta) == dff.shape[0] == 4
    assert len(timestamps) == dff.shape[1] == 100
    assert set(cell_meta["cell_specimen_id"]) == {11, 22, 33, 44}
