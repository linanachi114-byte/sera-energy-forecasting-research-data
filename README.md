# Research data: Dual-reference residual aggregation for energy forecasting

Version 2.0.0. Authors: Jiaqi Li, Ruidong Bian and Yu Zhang.

Coverage: thirteen energy-related series, physical leads 1/12/24 hours,
seeds 41/44. The primary system is the two-output numerical decoder DRA.
Its retained corrected outputs are E (DLinear reference and covariates) and
T (PatchTST reference, covariates and projected temporal state).

`primary_seed_mae.csv`: 390 model/task/seed test records for DRA, N, PatchTST,
TimeXer and TQNet. `task_mae.csv` contains seed-mean task errors.
`expected_primary_summary.csv` contains mean task-relative reductions,
descriptive source-cluster bootstrap intervals and task/series wins.
`decoder_seed_mae.csv`: 624 development/test records for four matched output
configurations. `decoder_summary.csv` reports changes relative to H+E+T.
`dataset_registry.json` identifies sources, processed hashes and partitions.
`training_cost_cells.csv` records DRA's five encoder/decoder fitting stages;
its separately trained DLinear reference is frozen during these stages.

Run offline:

```text
python reproduce_results.py
```

Dependencies: Python, NumPy and pandas. Torch is only needed for `model.py`.
The script reconstructs all 1,014 recorded MAEs from per-origin errors and
reproduces all aggregate reductions and descriptive intervals. Error archives
contain only signed/absolute errors and target/origin indices. Raw observed
series, credentials, manuscript files and language-model outputs are excluded.
The figure-data file contains errors and indices, not observed traces.

Within a task, average the two seed MAEs before forming relative reductions.
Then average all 39 tasks equally. Do not pool raw errors across incompatible
target units. Intervals resample eleven source groups with all leads retained:
ETTm1/ETTm2 stay together, GEFCom load/price stay together, and the other nine
series form separate groups. NumPy RNG seed: 7112026; draws: 20,000.

The four decoder comparisons share a numerical pretrained encoder and a
reference-assigned decoder initialization, followed by one additional matched
fit per retained output set. The E+T choice was fixed using the ten-series
development design and retained for the three additional sequences. The
encoder's numerical initialization mixture has three outputs; the final
prediction gate has two. The tested historical periods were examined during
research exploration and do not constitute a new independent holdout.

`model.py` is a portable definition of the normalized two-expert decoder and
its numerical preprocessing/loss functions. Reference architectures follow
the cited official PatchTST and DLinear implementations. Executed local-run
source snapshots preserve the training and recovery recipe; their original
project imports and input paths need adapting to a separate training setup.
The supplied offline reproduction script requires no such local paths.

Source-data licenses and redistribution terms remain with their providers.
This package does not grant rights to those source datasets. BDG2 processing
uses the source-verified, fixed 49-meter Hog cohort specified in the registry.
