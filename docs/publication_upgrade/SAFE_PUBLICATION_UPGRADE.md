# Safe publication upgrade policy

This project now treats long-running outputs as protected evidence. Publication-level
code and manuscript upgrades should not overwrite existing files under:

- `data/`
- `models/`
- `reports/`

Those directories can be read for audit, provenance, and manuscript planning, but
they should be regenerated only by an explicit analysis command chosen for that
purpose.

## Read-only audit

Run this command to inspect the current publication state without writing files:

```bash
python scripts/publication_audit_readiness.py
```

The same read-only behavior is now the default for:

```bash
python scripts/publication_run_all.py
```

To refresh lightweight derived publication tables from already saved embeddings,
use the explicit write flag:

```bash
python scripts/publication_run_all.py --write-derived
```

## What counts as safe project upgrading

Safe project upgrades include:

- editing source code under `src/`;
- adding tests under `tests/`;
- adding scripts under `scripts/`;
- editing planning documents under `docs/` and `manuscript/`;
- changing configuration files when they do not trigger automatic overwrites.

Potentially destructive or expensive actions include:

- deleting or replacing NWB, H5, NPZ, PT, or JOBLIB files;
- rerunning preprocessing over existing session files without an output plan;
- overwriting report tables that serve as the current manuscript source of truth;
- retraining CEBRA, dRNN, or decoder models without saving to a new versioned output path.
- extracting DNN activations or CEBRA variants into canonical `data/processed/`
  or `models/` paths.

## Recommended regeneration convention

When an analysis must be rerun, write into a versioned output namespace first:

```text
reports/tables/publication_upgrade_v2/
reports/figures/publication_upgrade_v2/
models/publication_upgrade_v2/
```

Only promote a new run to the canonical folders after checking the audit report,
session counts, random seeds, dependency versions, and figure/table integrity.

## Heavy optional commands

These commands are safe as dry runs by default:

```bash
python scripts/publication_extract_dnn_features.py --frames-npy path/to/frames.npy --session-id 500855614
python scripts/publication_run_cebra_variants.py --session-id 500855614 --variant all
python scripts/publication_run_latent_dynamics.py
```

They write only when given `--write-output`, and should then target a versioned
namespace such as `publication_upgrade_v2`.
