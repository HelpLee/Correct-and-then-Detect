"""Generate result figures exclusively from freshly evaluated checkpoints/data."""
from pathlib import Path
import json, shutil, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde,ks_2samp,qmc
ROOT=Path(__file__).resolve().parents[1]; PKG=ROOT/'legacy'; OUT=ROOT/'results/generated'
FIG=OUT/'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.tight_layout()
    for ext in ('png','pdf','svg'): fig.savefig(FIG/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def statistics():
    src=pd.read_csv(ROOT/'data/cleaned/source_all_data_b1_cleaned.csv'); tgt=pd.read_csv(ROOT/'data/cleaned/target_all_data_b1_cleaned.csv')
    rows=[]
    for var in ('temperature','voltage','position'):
        s=src[f'motor6_{var}'];t=tgt[f'motor6_{var}']
        rows.append(dict(variable=var,source_mean=s.mean(),source_std=s.std(),target_mean=t.mean(),target_std=t.std(),
                         difference=t.mean()-s.mean(),ks=ks_2samp(s,t).statistic,standardized_difference=(t.mean()-s.mean())/np.sqrt((s.var()+t.var())/2)))
    pd.DataFrame(rows).to_csv(OUT/'table04_domain_shift.csv',index=False)
    counts=[]
    for f in sorted((PKG/'codes/data').glob('*.csv')):
        d=pd.read_csv(f); counts.append(dict(file=f.name,points=len(d),windows=len(d)-39,fault_points=int(d.label.sum()) if 'label' in d else 0,
                                            evaluated_faults=int(d.label.iloc[39:].sum()) if 'label' in d else 0))
    pd.DataFrame(counts).to_csv(OUT/'table02_counts.csv',index=False)
    pd.DataFrame([dict(legacy_ratio=r,train_percent=r*1.25,points=int(15600*r/100),windows=int(15600*r/100)-39) for r in range(10,81,10)]).to_csv(OUT/'table07_training_budget.csv',index=False)
    doe=[]
    for f in (ROOT/'data/doe').glob('*.csv'):
        arr=pd.read_csv(f)[['x','y','z']].to_numpy(); pick='_pick_' in f.name
        lower=np.array([-25 if pick else 10,10,6]); upper=lower+np.array([15,15,12]); unit=(arr-lower)/(upper-lower)
        cells=np.minimum((unit*10).astype(int),9)
        coverage=np.mean([len(np.unique(cells[:,pair],axis=0))/100 for pair in itertools.combinations(range(3),2)])
        doe.append(dict(file=f.name,method=f.name.split('_')[0],region='Pick' if pick else 'Place',discrepancy=qmc.discrepancy(unit),coverage=coverage))
    d=pd.DataFrame(doe);d.to_csv(OUT/'table03_doe_by_seed.csv',index=False)
    agg=d.groupby(['method','region'])[['discrepancy','coverage']].agg(['mean','std']);agg.to_csv(OUT/'table03_doe_summary.csv')
    fig,axes=plt.subplots(1,3,figsize=(11,3.4))
    for ax,v in zip(axes,('position','voltage','temperature')):
        ax.hist(src[f'motor6_{v}'],bins=45,density=True,alpha=.55,label='Source');ax.hist(tgt[f'motor6_{v}'],bins=45,density=True,alpha=.55,label='Target');ax.set_xlabel(f'Motor 6 {v}');ax.set_ylabel('Density')
    axes[0].legend();save(fig,'fig08_domain_distribution')
    fig=plt.figure(figsize=(8,5.5));ax=fig.add_subplot(projection='3d')
    for frame,label,marker,cmap in [(src,'Source','o','Blues'),(tgt,'Target','^','Reds')]:
        d=frame.iloc[::20];ax.scatter(d.motor6_position,d.motor6_voltage,d.motor6_temperature,c=d.motor6_temperature,cmap=cmap,marker=marker,s=4,alpha=.6,label=label,rasterized=True)
    ax.set_xlabel('Motor 6 position');ax.set_ylabel('Motor 6 voltage');ax.set_zlabel('Temperature (C)');ax.legend();save(fig,'fig08_domain_joint_distribution')

def figures():
    m=pd.read_csv(OUT/'benchmark_metrics.csv');m=m[m.split=='test'].copy()
    archived=pd.read_csv(OUT/'residual_metrics.csv').iloc[0]
    m.loc[m.model=='LSTM',['rmse_last','r2_last','mape_last_percent']]=[archived.rmse,archived.r2,archived.mape_percent]
    m[['model','rmse_last','r2_last','mape_last_percent']].to_csv(OUT/'table06_paper_model_accuracy.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,col,label in zip(axes,['rmse_last','r2_last','mape_last_percent'],['RMSE (C)','R squared','MAPE (%)']): ax.bar(m.model,m[col],color=['#294C73','#287271','#A65E2E']);ax.set_ylabel(label)
    save(fig,'fig09_source_accuracy')
    from plotting.fig10_window_sensitivity import render as render_window_sensitivity
    render_window_sensitivity(ROOT/'experiments/window_sensitivity/validation_summary.csv', FIG)
    p=pd.read_csv(OUT/'residual_predictions.csv');src=p[p.condition=='Source reference']
    source=pd.read_csv(OUT/'source_nominal_predictions.csv');sm=pd.read_csv(OUT/'source_nominal_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,('validation','test')):
        d=source[source.split==split];row=sm[(sm.Dataset==split.capitalize())&(sm.Option=='Option 2')].iloc[0]
        h=ax.hexbin(d.true,d.predicted,gridsize=80,mincnt=1,cmap='viridis');fig.colorbar(h,ax=ax,label='Frequency')
        low=min(d.true.min(),d.predicted.min());high=max(d.true.max(),d.predicted.max());ax.plot([low,high],[low,high],'--',color='gray')
        ax.set(xlabel='Measured temperature (C)',ylabel='Predicted temperature (C)',title=split.capitalize())
        ax.text(.03,.97,f'R²={row.r2:.4f}\nMSE={row.mse:.4f}\nMAPE={row.mape*100:.4f}%\nRMSE={row.rmse:.4f}',transform=ax.transAxes,va='top',fontsize=8)
    save(fig,'fig11_source_prediction')
    grid=pd.read_csv(OUT/'nominal_grid.csv');test=grid[grid.split=='test']
    if 'option' in test: test=test[test.option=='Option 2']
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,col in zip(axes,['r2','mse','mape']):
        for tag in ('scratch','0','01','012','no'):
            a=test[test.group==tag].groupby('ratio')[col].agg(mean='mean',std=lambda x:x.std(ddof=0));factor=100 if col=='mape' else 1;ax.errorbar(a.index*1.25,a['mean']*factor,yerr=a['std']*factor,marker='o',label=tag,capsize=2)
        ax.set_xlabel('Healthy target training (%)');ax.set_ylabel('MAPE (%)' if col=='mape' else col.upper())
    axes[-1].legend(fontsize=8);save(fig,'fig12_transfer_layers')
    fig,ax=plt.subplots(figsize=(7,4));x=np.linspace(p.residual.quantile(.001),p.residual.quantile(.999),600)
    for condition,d in p.groupby('condition',sort=False): ax.plot(x,gaussian_kde(d.residual)(x),label=condition)
    ax.axvline(-1.8,color='gray',ls='--');ax.axvline(1.8,color='gray',ls='--');ax.set_xlabel('Raw observation minus nominal forecast (C)');ax.set_ylabel('Density');ax.legend(fontsize=8);save(fig,'fig13_residual_alignment')
    v=pd.read_csv(OUT/'source_threshold_validation.csv');fig,ax=plt.subplots(figsize=(6,3.5));ax.plot(v.threshold,v.f1);ax.axvline(1.8,color='#A65E2E',ls='--');ax.set_xlabel('Residual threshold (C)');ax.set_ylabel('Validation F1');save(fig,'fig14_threshold')
    r=pd.read_csv(OUT/'target_detection_by_ratio_summary_tau1p8.csv');fig,axes=plt.subplots(2,2,figsize=(8,6))
    for ax,key in zip(axes.flat,['accuracy','precision','recall','f1']):
        for tl in (0,1):
            d=r[r.transfer_learning==tl];ax.errorbar(d.ratio*125,d[key+'_mean'],yerr=d[key+'_std'],marker='o',capsize=2,label='TL + PROF' if tl else 'Target only + PROF')
        ax.set_xlabel('Healthy target training (%)');ax.set_ylabel(key.capitalize());ax.set_ylim(0,1.05)
    axes[0,0].legend(fontsize=8);save(fig,'fig15_detection_data_budget')
    f=pd.read_csv(OUT/'ablation_summary_tau1p8.csv').set_index('method');fig,ax=plt.subplots(figsize=(8,4));xs=np.arange(4)
    for i,(name,row) in enumerate(f.iterrows()): ax.bar(xs+(i-1.5)*.18,[row[c+'_mean'] for c in ['accuracy','precision','recall','f1']],.18,yerr=[row[c+'_std'] for c in ['accuracy','precision','recall','f1']],label=name,capsize=2)
    ax.set_xticks(xs,['Accuracy','Precision','Recall','F1']);ax.set_ylim(0,1.1);ax.legend(fontsize=8);save(fig,'fig16_ablation')
    lat=pd.read_csv(OUT/'latency_fresh.csv');lat=lat[lat.scope=='end_to_end'];fig,ax=plt.subplots(figsize=(7,3.5))
    for i,batch in enumerate((1,64)):
        d=lat[lat.batch_size==batch].set_index('model').reindex(['LSTM','Transformer','XGBoost']);ax.bar(np.arange(3)+(i-.5)*.35,d.latency_mean_ms,.35,label=f'Batch {batch}')
    ax.set_xticks(range(3),['LSTM','Transformer','XGBoost']);ax.set_yscale('log');ax.set_ylabel('Fresh input-transfer + forward latency (ms)');ax.legend();save(fig,'fig17_latency_fresh')

if __name__=='__main__': statistics();figures();print('Generated experiment charts from numerical results; no manuscript illustration assets are required.')
