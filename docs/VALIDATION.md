# Evaluation protocols

The released nominal and transfer weights are evaluated using Python 3.11.14, PyTorch 2.5.1 and the pinned dependencies. Source threshold calibration, target-data scaling and TL/PROF ablation use the same final-horizon residual decision rule.

| Experiment | Evaluation | Agreement with reference |
|---|---|---|
| Source threshold calibration | 61 thresholds on validation; selected threshold on test | Maximum detection-score difference 1.11e-16 |
| Target-data scaling | Five seeds, eight data budgets, two detectors | All 80 configurations agree |
| TL/PROF ablation | Five seeds, four configurations | All 20 configurations agree |
| Target nominal prediction | 200 checkpoints; train/validation/test; both horizon options | 1200 MSE/R² rows within 3e-4 |
| Nominal summary | MSE/R²/MAPE/RMSE means and standard deviations | 640 entries within 3e-4 |
| Source/controlled benchmark | Archived source and controlled LSTM, XGBoost, Transformer | Within stated floating tolerances |
| Window sensitivity | History lengths 20, 30, 40, 50, 60; horizons 10, 15, 20, 25, 30 | Recorded study and checkpoint artifacts included |

Window sensitivity uses the original experimental summary for Figure 10. Its 256-unit architecture is separate from the 64-unit nominal v3 configuration. Nominal aggregation preserves population SD (`ddof=0`); detector ablation uses sample SD (`ddof=1`).

Generated results are separate from immutable references. Comparisons use tolerance 1e-10 for detection and 3e-4 for floating regression metrics; they do not claim bitwise identity. Table 3 agrees with the printed DoE statistics. Table 4 uses cleaned full-domain data; its printed voltage mean difference has a last-digit rounding difference (158.3747508 versus 158.38).

The archived source scaler is supplied with the weights. Its voltage minimum differs from a scaler newly fitted to the current training CSV; use the supplied scaler for released-weight evaluation. Fresh training is a separate experiment with corrected deep-copy best-state selection and is not promised to produce identical historical weights.

Latency is measured afresh and varies with hardware/load. The timing scope is CPU input transfer + forward + synchronization, without CPU output retrieval or the complete detector loop. Training/fine-tuning durations remain recorded measurements. Conceptual drawings/platform images are retained assets; analytical charts use generated metrics, with recorded window results as described above.

Protocol tests cover temporal partition isolation, final-horizon indexing, immutable raw observations, causal PROF feedback, layer freezing and independent state snapshots. CI runs these tests; full evaluation requires the artifact release.
