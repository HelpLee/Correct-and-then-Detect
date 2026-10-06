# Correct-and-then-Detect

Data, models and reproducible experiments for digital-twin-based fault detection with transfer learning and prediction-based recursive observation feedback (PROF).

This repository contains experimental resources and reproduction instructions. Manuscripts, revision packages and response letters are maintained separately. A link to the finalized article repository will be added when it is ready.

[Data](docs/DATA.md) · [Experiments](docs/EXPERIMENTS.md) · [Training](docs/TRAINING.md) · [Validation](docs/VALIDATION.md) · [Provenance](docs/PROVENANCE.md)

## Results

### Transfer learning and PROF

![TL/PROF ablation](assets/ablation.png)

Five-seed factorial comparison of transfer learning (TL) and PROF on the target fault test set. The combined configuration achieves mean F1 0.9145 with sample SD 0.0045; the baseline without either component achieves 0.7493 with SD 0.0245. Numerical records are in `results/reference/ablation_by_seed_tau1p8.csv` and `ablation_summary_tau1p8.csv`.

### Target-data requirements

![Detection performance versus healthy target training data](assets/target_data_budget.png)

Detection performance at different healthy target training-data budgets, comparing transferred initialization with target-only learning. Curves and shaded bands show five-seed means and sample SDs. Numerical records are in `results/reference/target_detection_by_ratio_tau1p8.csv` and its summary.

The window-sensitivity study remains available as an auxiliary reproducible experiment; it is not a featured result preview.

## Quick start

Use Python 3.11. Install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Nominal modeling, transfer learning and fault detection

Download the [data and model artifact bundle](https://github.com/HelpLee/Correct-and-then-Detect/releases/download/v0.1.0/correct-and-then-detect-artifacts.zip) and extract it inside this repository, preserving the directory structure.

```bash
python reproduce.py verify
python reproduce.py smoke
python reproduce.py full
```

`full` evaluates released nominal/transfer checkpoints, source threshold calibration, target-data scaling and TL/PROF ablation, measures inference latency, and generates experimental charts. Outputs are written under ignored `results/generated/`. Individual stages: `nominal`, `detection`, `figures`, `compare`, `windows`.

The default detector uses a 30-step history, a direct 10-step output and the last output for each residual decision. The residual threshold is 1.8 °C, calibrated on source validation data. Raw observations remain immutable; PROF changes only the internal feedback buffer. Target experiments use seeds 42–46; the updated window study uses seeds 42–44.

### Updated window sensitivity

The complete 25-combination × three-seed study is included directly in [experiments/window_sensitivity](experiments/window_sensitivity): results, 75 checkpoints, prediction arrays, scalers and training histories.

```bash
python reproduce.py windows
```

This verifies saved predictions and summaries, then writes plots to `results/generated/figures/`. All configurations use the same 12,930 validation decision times. The lowest mean validation MSE is obtained with history 60 / horizon 10: MSE 0.155125, R² 0.897011. The final test partition is reserved and is not evaluated in this study.

## Fresh training and plotting

```bash
python experiments/window_sensitivity/run_experiment.py --seed 42
python experiments/window_sensitivity/run_experiment.py --seed 43
python experiments/window_sensitivity/run_experiment.py --seed 44
python experiments/window_sensitivity/make_report.py --input-dir results/generated/window_sensitivity
```

Fresh training writes separately from the published experiment. The source training CSV is supplied by the artifact bundle. See [training protocols](docs/TRAINING.md) for nominal and target-domain training.

Standalone experimental plotting scripts are in `scripts/plotting/`; their numerical inputs are in `results/reference/figure_data/`. Run a script directly to save its outputs under `results/generated/`. The full experiment renderer is `scripts/render_results.py`.

## Repository layout

```text
src/correct_detect/    causal windows, checkpoint models, freezing and PROF
experiments/          updated sensitivity results, weights and reproducible scripts
scripts/              evaluation, verification and plotting
configs/              explicit experiment settings
results/reference/    numerical references and plotting data
results/generated/    local evaluation outputs (ignored)
data/                 descriptive cleaned data and DoE points (artifact bundle)
artifacts/            checksum manifest for released data and weights
assets/               two current experimental result previews
legacy/               numerical records and original modules required by reproduction
tests/                protocol tests
docs/                 data, experiments, training and validation instructions
```

## Contributors

Haibo Li · Zhiguo Zeng · Hu Yang · Xu Li

GitHub: [HelpLee](https://github.com/HelpLee), [sonic160](https://github.com/sonic160).

[CITATION.cff](CITATION.cff) describes this software repository. Final article citation and links will be added after publication. See [distribution notes](docs/DISTRIBUTION.md) for resource packaging.
