# Distribution

The Git repository contains source code, numerical references and the updated three-seed sensitivity experiment, including its 75 checkpoints and prediction arrays. It contains no manuscript, revision package or response letter.

The data/model artifact ZIP supplies the remaining full input data, nominal/transfer weights and scalers; extract it inside the repository. The required artifact files are checked with `artifacts/manifest.json`. The data bundle has been refreshed to contain only the currently required artifacts. Sensitivity analysis uses `experiments/window_sensitivity/` in the repository; obsolete window weights are excluded from the bundle.

Source/reference integrity is checked with `source_manifest.json`. Generated figures, caches and evaluation output are ignored. Plots are recreated from numerical inputs rather than distributed as manuscript illustrations.

Final article links will be added when the manuscript is finalized. No license is assigned in this repository; licensing remains an author decision.
