# Scientific provenance and known inconsistencies

Current manuscript: `2nd_rev/manuscript_r2/manuscript_r2.tex`, with its second-round response and figures. The earlier manuscript snapshot is retained under `paper/`. Canonical residual threshold: 1.8 °C.

## Two source-model tracks

The historical model is `legacy/codes/models_reg_v3/source_best_model_v3_b1.pth` with its b1 scalers; source test R² is approximately 0.8591. The controlled `head` benchmark in `legacy/codes/model_comparison_source_611/results/checkpoints/` has R² approximately 0.8651. Figure 13 actually uses the archived `fc` source model and `experiments_transfer_v3_b1_42/exp_0_80%/transfer_best_model.pth`. The historical controlled alignment runner is a separate experiment and must not overwrite these canonical records.

The source-comparison subdirectory's historical README described the 0.8651 rerun as canonical. The latest manuscript supersedes that description. This package documents the distinction rather than silently altering results.

## Historical training issue

The original target-training notebook and some source/target scripts assign `best_model_state = model.state_dict()` without copying tensors. Subsequent optimizer steps can mutate that snapshot. The new training CLI uses `deepcopy` to implement true validation-best model selection, saves to a separate output tree, and does not claim exact historical numerical equivalence. The original notebook is retained as evidence of the result-generating implementation.

## Manuscript parameter inconsistency

The current manuscript parameter table states hidden dimension 256, while the archived checkpoints and downstream evaluators use 64. The organization task does not change manuscript scientific content. This discrepancy should be corrected by the authors before final submission.

## Scope of reproduction

`reproduce.py full` evaluates released nominal/benchmark weights, source threshold calibration, all target ratios and ablations, then draws charts from generated outputs. Historical records remain unchanged. There is no public `benchmark` retraining alias. The window-sensitivity figure is rendered from the new three-seed validation experiment in `experiments/window_sensitivity/`; the old 256-unit sweep is superseded. The saved fresh predictions and checkpoints are independent of the archived source checkpoint used for downstream fault detection. Checkpoint evaluation protocols are documented in [VALIDATION.md](VALIDATION.md).

Latency and training time are hardware-specific. Original benchmark metadata records Python 3.11.14, PyTorch 2.5.1 and an NVIDIA RTX 1000 Ada Generation Laptop GPU. Requirements pin the direct dependencies observed during organization; they are not a complete transitive lockfile. Old NumPy-2 scaler pickles were reserialized for the verified environment without changing learned numerical fields; see VALIDATION.md.
