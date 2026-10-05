# Paper ↔ Code ↔ Results

| Item | Computation / evidence |
|---|---|
| Tables 1, 5 | Sensor descriptions; `configs/protocols.json` distinguishes architectures |
| Table 2 | `table02_counts.csv`: data lengths, windows, aligned faults |
| Table 3 | Saved rounded DoE points; centered discrepancy; three 10×10 projected occupancies; `table03_doe_summary.csv` |
| Table 4 | Cleaned full domain CSVs; `table04_domain_shift.csv` |
| Table 6 | Archived LSTM and controlled XGBoost/Transformer; `table06_paper_model_accuracy.csv` |
| Table 7 | Complete-sequence ratio → healthy-train fraction; `table07_training_budget.csv` |
| Table 8 | Source threshold curve/test; comparison CSVs |
| Table 9 | Five seeds × four TL/PROF configurations; factorial comparison/summary |
| Table 10 | `latency_fresh.csv`; training durations remain historical records |
| Figures 1–7 | Fixed illustrations and platform assets |
| Figure 8 | Cleaned domain distributions |
| Figure 9 | Recomputed archived/controlled model accuracy |
| Figure 10 | Updated 25-combination × three-seed validation study in `experiments/window_sensitivity/`; render with `scripts/plotting/fig10_window_sensitivity.py` |
| Figure 11 | Archived source prediction trace |
| Figure 12 | 200 scratch/transfer weights × validation/test |
| Figure 13 | Archived source/direct target/layer-0-frozen target residuals |
| Figure 14 | Recomputed source threshold curve |
| Figure 15 | 80 target-data ratio detector runs |
| Figure 16 | 20 factorial detector runs |
| Figure 17 | Fresh inference timing with actual scope |

Generated CSVs live in `results/generated/`, charts in its `figures/` subdirectory. The canonical manuscript figures and recorded reference CSVs are immutable inputs.
