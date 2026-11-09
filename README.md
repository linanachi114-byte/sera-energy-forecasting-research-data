# SERA numerical research data

This deposit supports Energy forecasting with branch-specific reference forecasts
and residual experts. It covers the ten public series and 30 dataset–horizon tasks
reported in the manuscript.

## Contents

- primary_seed_mae.csv: 540 model–task–seed records for nine models, seeds 41/44.
- five_seed_mae.csv: 750 records for SERA, N, iTransformer, TimeMixer and TQNet,
  seeds 41–45.
- errors/: signed and absolute test errors with matched target indices for the
  990 distinct prediction archives represented in these comparisons. Values are
  retained in the corresponding target's original numerical scale.
- component_seed_mae.csv: all 360 matched decoder adaptation measurements.
- component_task_mae.csv: seed-averaged component results.
- semantic_development_sensitivity.csv: fixed-model input interventions on the
  development partition.
- dataset_registry.json: processing configuration, source identifiers, hashes,
  target definitions, covariates and chronological partition sizes.
- figure_data/forecast_error_examples.csv: the signed and absolute errors for
  the 168 origins shown for household power and German load.
- training_cost_cells.csv and parameter_counts.csv: recorded computational
  measurements.
- model_sources/: the selected implementation's original source snapshots.
- error_archive_provenance.json and SHA256SUMS.json: source-output and
  deposit-file digests.

Model labels follow the numerical records: S_current is SERA, N_KA is the
numerical gate, and S_KA is the direction-only semantic gate. Each horizon is
stored in native observation steps. ETTm1 (G) uses four observations per hour;
all other series are hourly.

## Reproduce the reported statistics

Install NumPy using requirements.txt, extract this deposit and run:

    python reproduce_results.py --output reproduced_results.json

The script verifies the deposit hashes, reconstructs each MAE from per-origin
errors, checks the published primary and five-seed summaries, and reconstructs
the component-effect means. It requires no original machine paths, checkpoints,
network access, model fitting or language-model calls.

Within each task, average the original-unit MAEs across seeds first. Compute
100 * (1 - SERA_MAE / comparator_MAE) and average the 30 task reductions.
Dataset-bootstrap descriptive intervals retain each dataset's three leads.
Component adaptation uses the opposite ratio, 100 * (control/reference - 1).
Repeated training seeds evaluate variability on the same benchmark periods.

## Sources and reuse

Third-party observation series are not included. Obtain them from the cited
providers and follow their respective licence terms. The forecast-error examples
use UCI Individual Household Electric Power Consumption (DOI 10.24432/C58K54)
and the OPSD time-series release dated 2020-10-06
(https://doi.org/10.25832/time_series/2020-10-06; primary source: ENTSO-E
Transparency Platform). The observed traces shown in the article can be
reconstructed from those providers' data using the manuscript's preprocessing
and the target indices recorded here. Source attributions and applicable
provider terms remain in force.

The deposit contains numerical outputs and implementation snapshots, not API
credentials or language-model response texts. The snapshots preserve their
original imports and are implementation evidence; reproduce_results.py is the
self-contained analysis entry point, not a replacement for the training system.
Public distribution and any reuse licence are determined by the copyright
holders and the repository record.
