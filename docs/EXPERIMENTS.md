# Experiments and reproduction entries

| Experiment | Entry / numerical evidence |
|---|---|
| Updated window sensitivity | `python reproduce.py windows`; 25 combinations × seeds 42–44; results and weights in `experiments/window_sensitivity/` |
| Source nominal prediction | `python reproduce.py nominal`; source checkpoint and prediction traces |
| Controlled predictor comparison | Archived LSTM and controlled XGBoost/Transformer; `table06_paper_model_accuracy.csv` in generated outputs |
| Healthy target adaptation | Scratch and layer-freezing configurations; 200 checkpoints; validation/test metrics |
| Residual alignment | Source, direct target and layer-0-frozen target residuals |
| Threshold calibration | Source validation residual sweep; fixed-threshold source test evaluation |
| Target-data scaling | Five seeds × eight data budgets × two detectors |
| TL/PROF ablation | Five seeds × four configurations |
| Deployment timing | Fresh input-transfer/forward/synchronization measurements |
| DoE and domain statistics | Cleaned data and saved DoE points; `python reproduce.py figures` after evaluation |

Plots are generated from numerical inputs using `scripts/render_results.py` and `scripts/plotting/`. No manuscript PDFs or illustration assets are required. Generated CSVs and figures live under `results/generated/`, separately from fixed references.

## Relationship to the article

The manuscript is being finalized in a separate writing repository. This repository is the data and reproduction companion; finalized article links and a stable experiment-to-article mapping will be added after publication.
