"""Verify published validation traces, aggregates and checkpoint architecture."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(2)
protocol = json.loads((HERE / 'protocol.json').read_text())
rows = pd.read_csv(HERE / 'validation_results_per_seed.csv')
summary = pd.read_csv(HERE / 'validation_summary.csv')
assert len(rows) == 75 and len(summary) == 25
assert not rows.duplicated(['n_past', 'n_future', 'seed']).any()
assert set(rows.seed) == {42, 43, 44}
assert rows.groupby(['n_past', 'n_future']).size().eq(3).all()
assert protocol['test_evaluation'] is False
decisions = pd.read_csv(HERE / 'validation_decision_rows.csv')
assert len(decisions) == protocol['common_validation_decisions'] == 12930
assert decisions.source_row_zero_based.is_unique
assert decisions.source_row_zero_based.between(72800, 88399).all()
truth = None
for row in rows.itertuples():
    folder = HERE / f'p{row.n_past}_h{row.n_future}' / f'seed{row.seed}'
    with np.load(folder / 'validation_predictions.npz') as saved:
        y, prediction = saved['true'], saved['predicted']
        assert y.shape == prediction.shape == (12930,)
        if truth is None:
            truth = y.copy()
        assert np.array_equal(y, truth), 'Validation target times differ'
        mse = np.mean((y - prediction) ** 2)
        r2 = 1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
        mape = np.mean(np.abs((y - prediction) / y)) * 100
        for actual, recorded in [(mse, row.validation_mse), (r2, row.validation_r2),
                                 (mse ** .5, row.validation_rmse), (mape, row.validation_mape_percent)]:
            assert np.isclose(actual, recorded, rtol=1e-10, atol=1e-12)
    state = torch.load(folder / 'best_model.pth', map_location='cpu', weights_only=True)
    assert state['lstm.weight_ih_l0'].shape == (256, 18)
    assert state['lstm.weight_hh_l2'].shape == (256, 64)
    assert state['fc.weight'].shape == (row.n_future, 64)
    assert (folder / 'training_history.csv').exists()
for row in summary.itertuples():
    group = rows[(rows.n_past == row.n_past) & (rows.n_future == row.n_future)]
    for source, destination in [('validation_mse', 'mse'), ('validation_r2', 'r2'),
                                ('validation_rmse', 'rmse'), ('validation_mape_percent', 'mape_percent')]:
        assert np.isclose(group[source].mean(), getattr(row, destination + '_mean'), rtol=1e-10)
        assert np.isclose(group[source].std(ddof=1), getattr(row, destination + '_sd'), rtol=1e-10)
best = summary.loc[summary.mse_mean.idxmin()]
assert (best.n_past, best.n_future) == (60, 10)
print('PASS: 75 traces/checkpoints, 25 three-seed summaries, identical validation targets; best 60/10.')
