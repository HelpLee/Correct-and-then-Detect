# Fresh training

```bash
python train_models.py --mode source --seeds 42
python train_models.py --mode target --seeds 42 43 44 45 46 --ratios 10 20 30 40 50 60 70 80 --freeze 0 01 012 no
```

Source and scratch-target defaults: learning rate 0.005, maximum 100 epochs. Transfer: 0.0001, 500 epochs, weight decay 0.0001. Batch 64, patience 10. Legacy folder 80% represents all 12480 healthy target training rows, not 80% of that partition.

Best-state snapshots now use deep copies, fixing the historical shallow snapshot bug. Runs save weights, their own scalers, history and metadata under `results/generated/training/`. They do not replace released weights. The fix and unresolved source-scaler lineage mean exact historical retraining is not promised.

The public audit evaluates released weights; new training outputs are not automatically substituted into that released-weight path. Retraining comparison is a separate validation scope.
