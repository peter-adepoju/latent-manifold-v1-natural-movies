# Upgrade workflow

1. Copy this upgrade pack into the current project root.
2. Run `python scripts/publication_validate_environment.py`.
3. Run notebooks `11`--`17`.
4. Add more Allen VISp sessions and rerun notebooks `11`--`17`.
5. Use `reports/tables/publication_upgrade` and `reports/figures/publication_upgrade` as the manuscript source of truth.
6. Keep the original single-session results as `v1.0` and present the multi-session analyses as `v2.0`.

## Recommended run order

```text
00--10 existing notebooks  -> already completed
11 publication cohort/QC
12 enhanced stimulus features
13 multisession encoding/decoding/nulls
14 layer geometry statistics
15 brain-model alignment
16 cross-session manifold alignment
17 publication figures/manuscript assets
```
