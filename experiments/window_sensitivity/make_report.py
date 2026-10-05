"""Summarize only fresh validation results and verify saved prediction metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
HERE=args.input_dir.resolve()
all_runs=pd.DataFrame([json.loads(p.read_text()) for p in HERE.glob('p*_h*/seed*/metrics.json')]).sort_values(['n_past','n_future','seed'])
all_runs.to_csv(HERE/'all_completed_runs_preserved.csv',index=False)
d=all_runs[all_runs.seed.isin([42,43,44])].copy()
assert len(d)==75 and not d.duplicated(['n_past','n_future','seed']).any()
assert d.groupby(['n_past','n_future']).size().eq(3).all()
reference=None
for row in d.itertuples():
    p=np.load(HERE/f'p{row.n_past}_h{row.n_future}'/f'seed{row.seed}'/'validation_predictions.npz')
    if reference is None: reference=p['true']
    assert np.array_equal(reference,p['true']), 'Validation decisions differ across configurations'
    mse=np.mean((p['true']-p['predicted'])**2)
    assert np.isclose(mse,row.validation_mse,rtol=1e-10,atol=1e-12)
d.to_csv(HERE/'validation_results_per_seed.csv',index=False)
s=d.groupby(['n_past','n_future']).agg(n_seeds=('seed','count'),mse_mean=('validation_mse','mean'),mse_sd=('validation_mse','std'),r2_mean=('validation_r2','mean'),r2_sd=('validation_r2','std'),rmse_mean=('validation_rmse','mean'),rmse_sd=('validation_rmse','std'),mape_percent_mean=('validation_mape_percent','mean'),mape_percent_sd=('validation_mape_percent','std')).reset_index()
s['rank']=s.mse_mean.rank(method='min').astype(int)
s.to_csv(HERE/'validation_summary.csv',index=False)
best=s.loc[s.mse_mean.idxmin()]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
colors=['#24537A','#C65D2E','#2C8583','#746491','#B57B22']
for h,c,marker in zip([10,15,20,25,30],colors,['o','s','^','D','P']):
    g=s[s.n_future==h].sort_values('n_past')
    for ax,metric in zip(axes,['r2','mse']):
        ax.errorbar(g.n_past,g[f'{metric}_mean'],yerr=g[f'{metric}_sd'],color=c,marker=marker,
                    markersize=5,linewidth=1.6,capsize=3,label=str(h))
for ax in axes:
    ax.set_xlabel('History-window length'); ax.set_xticks([20,30,40,50,60]); ax.grid(alpha=.2)
axes[0].set_ylabel('Validation R² (mean ± sample SD)')
axes[1].set_ylabel('Validation MSE (mean ± sample SD)')
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=5,title='Prediction horizon · seeds 42, 43, 44')
fig.savefig(HERE/'validation_sensitivity.png',dpi=240)
fig.savefig(HERE/'validation_sensitivity.pdf')
plt.close(fig)
protocol=json.loads((HERE/'protocol.json').read_text())
lines=['# Fresh source-domain window sensitivity study','',
       'The main comparison uses 75 models trained from scratch: 25 history/horizon combinations × seeds 42, 43, 44. Completed runs were preserved and reused. No archived metrics, weights or scalers were reused.','',
       '- Model: 18-input, 3-layer LSTM with 64 hidden units, dropout 0.2 and a direct multi-step linear output head.',
       '- Source healthy data: `legacy/codes/data/source_all_data_b1.csv`; 104,000 finite, nonmissing rows. Original row order retained.',
       '- Training: rows [0, 72800); validation: [72800, 88400); final test: [88400, 104000), not evaluated.',
       '- Both MinMax scalers are fitted on the training partition only and saved with the experiment.',
       '- Windows cannot cross partition boundaries or timestamp discontinuities (negative jump or gap >1 second). This also isolates the single timestamp rollback row. Repeated integer-second timestamps are permitted for the approximately 10 Hz observations.',
       f'- All configurations are scored on the same {protocol["common_validation_decisions"]:,} physical validation decision rows. Validation window lengths and forecast lead times differ; the evaluated target times do not.',
       '- Adam, learning rate 0.005, batch size 64, maximum 100 epochs, early-stopping patience 10. Training minimizes all-output MSE; the best epoch minimizes last-output validation MSE, matching the reported metric.',
       '- Each checkpoint, per-epoch training log and validation prediction vector is saved in its combination/seed folder. Standard deviations use ddof=1.',
       '- Nominal forecasting sensitivity only. These results do not measure PROF fault-detection performance. Longer forecast horizons are harder prediction tasks; their lower scores do not by themselves establish the best operational forecast lead time.',
       '- Validation is used for model/epoch selection; these are development scores, not an unbiased final test estimate. Three seeds quantify training variability, not confidence over new datasets.','',
       f'Best by mean validation MSE: **{int(best.n_past)}/{int(best.n_future)}**, MSE {best.mse_mean:.6f} ± {best.mse_sd:.6f}, R² {best.r2_mean:.6f} ± {best.r2_sd:.6f}.','',
       '| History | Horizon | Validation R² mean ± SD | Validation MSE mean ± SD | MSE rank |',
       '|---:|---:|---:|---:|---:|']
for r in s.itertuples():
    cells=[str(r.n_past),str(r.n_future),f'{r.r2_mean:.6f} ± {r.r2_sd:.6f}',f'{r.mse_mean:.6f} ± {r.mse_sd:.6f}',str(r.rank)]
    if r.rank==1: cells=[f'**{cell}**' for cell in cells]
    lines.append('| '+' | '.join(cells)+' |')
lines+=['','Validation checks passed: 75 unique runs; 3 seeds per setting; identical target arrays across all runs; metrics recalculated from saved predictions agree with the recorded results.','',
        'Run with the existing `robot` Python environment: `run_experiment.py`, then `make_report.py`. `protocol.json` records the input hash, preprocessing, training configuration and split boundaries.']
(HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
table=['# 验证集窗口敏感性分析：3 个 seed 的平均结果','',
       '25 种组合，每组包含 seeds 42、43、44。各组合使用同一批 12,930 个验证时刻；按平均 MSE 最低选择最优组合并加粗。R² 越高越好，其余误差指标越低越好。', '',
       '| 历史窗口 | 预测跨度 | 平均 R² | 平均 MSE | 平均 RMSE | 平均 MAPE (%) |',
       '|---:|---:|---:|---:|---:|---:|']
for r in s.itertuples():
    cells=[str(r.n_past),str(r.n_future),f'{r.r2_mean:.6f}',f'{r.mse_mean:.6f}',f'{r.rmse_mean:.6f}',f'{r.mape_percent_mean:.6f}']
    if r.rank==1: cells=[f'**{cell}**' for cell in cells]
    table.append('| '+' | '.join(cells)+' |')
table+=['','CSV 中同时保留各指标的样本标准差（ddof=1）。平均 RMSE 是各 seed 的 RMSE 平均值，不是平均 MSE 的平方根。']
(HERE/'validation_average_results.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
print(s.sort_values('rank').head().to_string(index=False))
print('Saved plots and README; prediction metric validation passed.')
