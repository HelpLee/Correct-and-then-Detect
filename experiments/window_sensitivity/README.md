# Fresh source-domain window sensitivity study

The main comparison uses 75 models trained from scratch: 25 history/horizon combinations × seeds 42, 43, 44. Completed runs were preserved and reused. No archived metrics, weights or scalers were reused.

- Model: 18-input, 3-layer LSTM with 64 hidden units, dropout 0.2 and a direct multi-step linear output head.
- Source healthy data: `legacy/codes/data/source_all_data_b1.csv`; 104,000 finite, nonmissing rows. Original row order retained.
- Training: rows [0, 72800); validation: [72800, 88400); final test: [88400, 104000), not evaluated.
- Both MinMax scalers are fitted on the training partition only and saved with the experiment.
- Windows cannot cross partition boundaries or timestamp discontinuities (negative jump or gap >1 second). This also isolates the single timestamp rollback row. Repeated integer-second timestamps are permitted for the approximately 10 Hz observations.
- All configurations are scored on the same 12,930 physical validation decision rows. Validation window lengths and forecast lead times differ; the evaluated target times do not.
- Adam, learning rate 0.005, batch size 64, maximum 100 epochs, early-stopping patience 10. Training minimizes all-output MSE; the best epoch minimizes last-output validation MSE, matching the reported metric.
- Each checkpoint, per-epoch training log and validation prediction vector is saved in its combination/seed folder. Standard deviations use ddof=1.
- Nominal forecasting sensitivity only. These results do not measure PROF fault-detection performance. Longer forecast horizons are harder prediction tasks; their lower scores do not by themselves establish the best operational forecast lead time.
- Validation is used for model/epoch selection; these are development scores, not an unbiased final test estimate. Three seeds quantify training variability, not confidence over new datasets.

Best by mean validation MSE: **60/10**, MSE 0.155125 ± 0.004043, R² 0.897011 ± 0.002684.

| History | Horizon | Validation R² mean ± SD | Validation MSE mean ± SD | MSE rank |
|---:|---:|---:|---:|---:|
| 20 | 10 | 0.889088 ± 0.003735 | 0.167059 ± 0.005626 | 5 |
| 20 | 15 | 0.843172 ± 0.002017 | 0.236220 ± 0.003038 | 6 |
| 20 | 20 | 0.779183 ± 0.002669 | 0.332602 ± 0.004020 | 11 |
| 20 | 25 | 0.701262 ± 0.004803 | 0.449969 ± 0.007234 | 17 |
| 20 | 30 | 0.641982 ± 0.003217 | 0.539259 ± 0.004845 | 21 |
| 30 | 10 | 0.894124 ± 0.002774 | 0.159474 ± 0.004179 | 3 |
| 30 | 15 | 0.842068 ± 0.001907 | 0.237883 ± 0.002872 | 7 |
| 30 | 20 | 0.771692 ± 0.004414 | 0.343885 ± 0.006649 | 13 |
| 30 | 25 | 0.703328 ± 0.004958 | 0.446857 ± 0.007468 | 16 |
| 30 | 30 | 0.630415 ± 0.004420 | 0.556681 ± 0.006657 | 24 |
| 40 | 10 | 0.895544 ± 0.001845 | 0.157336 ± 0.002780 | 2 |
| 40 | 15 | 0.837125 ± 0.002445 | 0.245328 ± 0.003683 | 10 |
| 40 | 20 | 0.770333 ± 0.001475 | 0.345932 ± 0.002221 | 14 |
| 40 | 25 | 0.701115 ± 0.004667 | 0.450191 ± 0.007030 | 18 |
| 40 | 30 | 0.628346 ± 0.005551 | 0.559797 ± 0.008361 | 25 |
| 50 | 10 | 0.893677 ± 0.001584 | 0.160147 ± 0.002386 | 4 |
| 50 | 15 | 0.838885 ± 0.005661 | 0.242676 ± 0.008527 | 9 |
| 50 | 20 | 0.767597 ± 0.000978 | 0.350054 ± 0.001474 | 15 |
| 50 | 25 | 0.696428 ± 0.005205 | 0.457250 ± 0.007840 | 20 |
| 50 | 30 | 0.633711 ± 0.006972 | 0.551717 ± 0.010501 | 23 |
| **60** | **10** | **0.897011 ± 0.002684** | **0.155125 ± 0.004043** | **1** |
| 60 | 15 | 0.839935 ± 0.003203 | 0.241095 ± 0.004824 | 8 |
| 60 | 20 | 0.775905 ± 0.004436 | 0.337539 ± 0.006682 | 12 |
| 60 | 25 | 0.700035 ± 0.002117 | 0.451817 ± 0.003188 | 19 |
| 60 | 30 | 0.637503 ± 0.007571 | 0.546006 ± 0.011404 | 22 |

Validation checks passed: 75 unique runs; 3 seeds per setting; identical target arrays across all runs; metrics recalculated from saved predictions agree with the recorded results.

Run with the existing `robot` Python environment: `run_experiment.py`, then `make_report.py`. `protocol.json` records the input hash, preprocessing, training configuration and split boundaries.

## Public commands

`python reproduce.py windows` verifies saved predictions, mean/sample-SD summaries and checkpoints, then renders Figure 10. Retrain with `python experiments/window_sensitivity/run_experiment.py --seed 42` (also 43 and 44); new runs are saved separately to `results/generated/window_sensitivity/`. Build their report with `python experiments/window_sensitivity/make_report.py --input-dir results/generated/window_sensitivity`. Training data come from the existing artifact release.
