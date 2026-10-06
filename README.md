<a id="readme-top"></a>

<div align="center">

<h1>Correct-and-then-Detect</h1>

Companion code, data and pretrained models for *Correct-and-then-Detect: Adaptive Fault Detection through Data-driven Digital Twins and Transfer Learning*.

<p>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.11"></a>
  <a href="requirements.txt"><img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?style=flat-square&amp;logo=pytorch&amp;logoColor=white" alt="PyTorch 2.5.1"></a>
  <a href="https://github.com/HelpLee/Correct-and-then-Detect/releases/tag/v0.1.0"><img src="https://img.shields.io/badge/Release-v0.1.0-0F766E?style=flat-square&amp;logo=github&amp;logoColor=white" alt="Release v0.1.0"></a>
</p>

<p>
  <a href="#getting-started"><strong>Getting started</strong></a> &nbsp;&middot;&nbsp;
  <a href="#reproducing-the-experiments"><strong>Reproduce</strong></a> &nbsp;&middot;&nbsp;
  <a href="#framework-overview"><strong>Framework</strong></a> &nbsp;&middot;&nbsp;
  <a href="#selected-results"><strong>Results</strong></a>
</p>

</div>

The framework learns nominal system behavior from robot observations and uses a digital twin to generate reference predictions for residual-based fault detection. Transfer learning adapts the twin using healthy target-domain data. Prediction-based recursive observation feedback (PROF) replaces detected abnormal observations in the internal feedback buffer with nominal predictions before constructing subsequent context windows.

This repository provides the resources to reproduce nominal modeling, target-domain adaptation, residual analysis and fault-detection experiments, together with the scripts used to generate their figures.

<p align="center">
  <a href="docs/DATA.md">Data</a> &nbsp;&middot;&nbsp;
  <a href="docs/EXPERIMENTS.md">Experiments</a> &nbsp;&middot;&nbsp;
  <a href="docs/TRAINING.md">Training</a> &nbsp;&middot;&nbsp;
  <a href="docs/VALIDATION.md">Validation</a>
</p>

<details>
<summary><strong>Contents</strong></summary>

- [Getting started](#getting-started)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Training and plotting](#training-and-plotting)
- [Framework overview](#framework-overview)
- [Selected results](#selected-results)
- [Repository structure](#repository-structure)
- [Contributors and citation](#contributors-and-citation)

</details>

## Getting started

Use Python 3.11 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download the [data and pretrained model bundle](https://github.com/HelpLee/Correct-and-then-Detect/releases/download/v0.1.0/correct-and-then-detect-artifacts.zip) and extract it into the repository root, preserving its directory structure. Then check the resources and run the reproduction pipeline:

```bash
python reproduce.py verify
python reproduce.py smoke
python reproduce.py full
```

The pipeline evaluates the supplied checkpoints, calculates prediction and detection metrics, generates figures and compares the results with the saved numerical references. Outputs are saved under `results/generated/`.

## Reproducing the experiments

Individual stages can also be run using the following commands:

| Experiment or stage | Command | Output |
|---|---|---|
| Nominal modeling and target-domain adaptation | `python reproduce.py nominal` | Prediction metrics, model comparisons, residual alignment and inference timing |
| Fault detection | `python reproduce.py detection` | Threshold calibration, target-data budgets and TL/PROF ablation |
| Window sensitivity | `python reproduce.py windows` | Verification of saved predictions and summary plots for 25 history/horizon combinations across three seeds |
| Figures | `python reproduce.py figures` | Experimental charts from the numerical results |
| Reference comparison | `python reproduce.py compare` | Comparison of generated metrics with saved reference values |

Run `nominal` and `detection` before generating figures or comparing results; `full` runs these stages in sequence. The window-sensitivity command can be run independently.

The detector uses a 30-step history and a direct 10-step prediction, with the last predicted output used for each residual decision. The threshold is 1.8 °C, calibrated on source validation data. PROF updates the internal feedback buffer while preserving raw observations for residual calculation. Target-domain experiments use seeds 42–46. The window-sensitivity experiment uses seeds 42–44 and evaluates every combination at the same 12,930 source validation decision times.

See [experiment details](docs/EXPERIMENTS.md) and [validation protocols](docs/VALIDATION.md) for dataset partitions, checkpoint pairing and evaluation settings. Protocol tests can be run with:

```bash
python -m unittest discover -s tests -v
```

## Training and plotting

Training entry points cover source nominal models, target-domain adaptation and window sensitivity:

```bash
python train_models.py --mode source --seeds 42
python train_models.py --mode target --seeds 42 43 44 45 46 --ratios 10 20 30 40 50 60 70 80 --freeze 0 01 012 no
python experiments/window_sensitivity/run_experiment.py --seed 42
```

For the window experiment, repeat the command with seeds 43 and 44, then generate its summaries:

```bash
python experiments/window_sensitivity/make_report.py --input-dir results/generated/window_sensitivity
```

Training saves model weights, scalers, histories and metrics under `results/generated/`. The [training guide](docs/TRAINING.md) describes the configurations and output directories.

Plotting scripts are available in `scripts/plotting/`, with numerical inputs in `results/reference/figure_data/`. Run a plotting script directly to generate its figure, or use `python reproduce.py figures` for the experiment renderer.

## Framework overview

![Data-driven digital twin framework](assets/fig01_framework_overview.png)

The framework combines representative task design and data acquisition, rolling-window nominal prediction, and transfer-learning adaptation using verified healthy data when the system's nominal behavior evolves.

## Selected results

### Transfer learning and PROF

![Transfer learning and PROF ablation](assets/ablation.png)

Across five seeds on the target fault test set, the combined TL/PROF configuration achieves mean F1 0.9145 with sample standard deviation 0.0045. The configuration without either component achieves mean F1 0.7493 with standard deviation 0.0245. Per-seed results and summaries are available in `results/reference/ablation_by_seed_tau1p8.csv` and `results/reference/ablation_summary_tau1p8.csv`.

### Healthy target-data budget

![Detection performance versus healthy target training data](assets/target_data_budget.png)

Detection performance at different healthy target training-data budgets, comparing transferred initialization with target-only learning. Curves and shaded bands show five-seed means and sample standard deviations. Numerical results are available in `results/reference/target_detection_by_ratio_tau1p8.csv` and its summary.

## Repository structure

```text
src/correct_detect/    window construction, models, layer freezing and PROF
configs/              experiment settings
experiments/          experiment scripts, saved results and checkpoints
scripts/              evaluation, verification and plotting
results/reference/    numerical references and figure data
results/generated/    locally generated metrics, models and figures
data/                 cleaned data and DoE points supplied in the bundle
artifacts/            checksum manifest for bundled data and models
assets/               framework overview and selected result figures
legacy/               original modules and resources used by evaluation
tests/                protocol tests
docs/                 data, experiment, training and validation guides
```

## Contributors and citation

Haibo Li · Zhiguo Zeng · Hu Yang · Xu Li

GitHub: [HelpLee](https://github.com/HelpLee), [sonic160](https://github.com/sonic160).

Citation metadata is provided in [CITATION.cff](CITATION.cff). The final article citation and link will be added after publication. See [distribution notes](docs/DISTRIBUTION.md) for the resource bundle contents.
