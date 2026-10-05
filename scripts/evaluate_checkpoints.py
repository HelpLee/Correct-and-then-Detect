"""Read-only audit of released checkpoints; all writes go to generated outputs."""
from pathlib import Path
import importlib.util, json, sys, argparse, time
import numpy as np
import pandas as pd
import torch, joblib
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import wasserstein_distance, ks_2samp

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'legacy'
OUT = ROOT / 'results' / 'generated'
DATA = PKG / 'codes/data'
MODELS = PKG / 'codes/models_reg_v3'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT.mkdir(exist_ok=True)
CHECKS = []
sys.path.insert(0, str(ROOT / 'src'))

def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def compare(name, actual, reference, keys, columns, tolerance=3e-4):
    merged = actual.merge(reference, on=keys, suffixes=('_new', '_ref'), validate='one_to_one')
    if len(merged) != len(reference) or len(actual) != len(reference):
        raise ValueError(f'{name}: incomplete comparison')
    for col in columns:
        merged[col+'_delta'] = merged[col+'_new'] - merged[col+'_ref']
    merged.to_csv(OUT / (name+'_comparison.csv'), index=False)
    delta = float(merged[[c+'_delta' for c in columns]].abs().max().max())
    CHECKS.append(dict(name=name, rows=len(merged), maximum_absolute_delta=delta,
                       tolerance=tolerance, status='PASS' if delta <= tolerance else 'MISMATCH'))
    print(CHECKS[-1], flush=True)

from correct_detect.data import features, windows
from correct_detect.models import CheckpointLSTM as LSTM

def load_model(path):
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model = LSTM(state).to(DEVICE); model.load_state_dict(state); model.eval(); return model

@torch.inference_mode()
def predict(model,x):
    return np.concatenate([model(b.to(DEVICE)).cpu().numpy() for b in x.split(512)])

def physical(a,ts):
    return ts.inverse_transform(a.reshape(-1,1)).reshape(a.shape)

def metrics(y,p):
    return dict(mse=float(mean_squared_error(y,p)), rmse=float(mean_squared_error(y,p)**.5),
                r2=float(r2_score(y,p)), mape=float(mean_absolute_percentage_error(y,p)))

def nominal():
    fs=joblib.load(MODELS/'feature_scaler_v3_b1.pkl'); ts=joblib.load(MODELS/'target_scaler_v3_b1.pkl')
    src=pd.read_csv(DATA/'source_all_data_b1.csv')
    source_splits={'validation':src.iloc[int(.7*len(src)):int(.85*len(src))], 'test':src.iloc[int(.85*len(src)):]}
    tgt={s:pd.read_csv(DATA/f'target_{s}_data_b1.csv') for s in ('train','val','test')}
    predictions=[]; residual_metrics=[]
    src_model=load_model(MODELS/'source_best_model_v3_b1.pth')
    source_rows=[];source_predictions=[]
    for split,frame in [('train',src.iloc[:int(.7*len(src))]),*source_splits.items()]:
        x,y=windows(frame,fs,ts);pp=physical(predict(src_model,x),ts);yy=physical(y,ts)
        source_predictions.append(pd.DataFrame(dict(split=split,true=yy[:,-1],predicted=pp[:,-1])))
        for option in ('Option 1','Option 2'):
            true,pred=(yy.ravel(),pp.ravel()) if option=='Option 1' else (yy[:,-1],pp[:,-1])
            source_rows.append(dict(Dataset=split.capitalize(),Option=option,**metrics(true,pred)))
    source_metrics=pd.DataFrame(source_rows);source_metrics.to_csv(OUT/'source_nominal_metrics.csv',index=False)
    pd.concat(source_predictions).to_csv(OUT/'source_nominal_predictions.csv',index=False)
    source_ref=pd.read_csv(ROOT/'results/reference/source_nominal_metrics.csv').rename(columns={'MSE':'mse','R2':'r2'})
    compare('source_nominal',source_metrics,source_ref,['Dataset','Option'],['mse','r2'])
    for label,frame,model in [('Source reference',source_splits['test'],src_model),
                             ('Target before TL (direct source model)',tgt['test'],src_model),
                             ('Target after TL',tgt['test'],load_model(MODELS/'experiments_transfer_v3_b1_42/exp_0_80%/transfer_best_model.pth'))]:
        x,y=windows(frame,fs,ts); pred=physical(predict(model,x),ts)[:,-1]; true=physical(y,ts)[:,-1]
        e=true-pred; row=dict(condition=label,n_test_windows=len(e),**metrics(true,pred))
        row['mape_percent']=row.pop('mape')*100; row['residual_mean']=float(e.mean()); row['residual_std']=float(e.std())
        residual_metrics.append(row); predictions.append(pd.DataFrame(dict(condition=label,true=true,predicted=pred,residual=e)))
    predframe=pd.concat(predictions); ref_e=predictions[0].residual.to_numpy()
    for row,frame in zip(residual_metrics,predictions):
        row['wasserstein_to_source']=wasserstein_distance(ref_e,frame.residual); row['ks_to_source']=ks_2samp(ref_e,frame.residual).statistic
    residual=pd.DataFrame(residual_metrics); residual.to_csv(OUT/'residual_metrics.csv',index=False); predframe.to_csv(OUT/'residual_predictions.csv',index=False)
    refdir=PKG/'codes/model_comparison_source_611/results'
    compare('residual_alignment',residual,pd.read_csv(refdir/'residual_alignment_before_after_metrics.csv'),['condition'],['mse','rmse','r2','mape_percent','wasserstein_to_source','ks_to_source'])
    rows=[]
    for seed in range(42,47):
        for ratio in range(10,81,10):
            healthy=tgt['train'].iloc[:int(len(tgt['train'])*ratio/80)]
            scratch_fs=MinMaxScaler().fit(features(healthy)); scratch_ts=MinMaxScaler().fit(healthy[['motor6_temperature']])
            for tag in ('scratch','0','01','012','no'):
                folder=MODELS/(f'experiments_target_v3_b1_{seed}/exp_{ratio}%' if tag=='scratch' else f'experiments_transfer_v3_b1_{seed}/exp_{tag}_{ratio}%')
                model=load_model(folder/('target_best_model_v3.pth' if tag=='scratch' else 'transfer_best_model.pth'))
                fsc,tsc=(scratch_fs,scratch_ts) if tag=='scratch' else (fs,ts)
                reference=pd.read_csv(next(folder.glob('metrics*.csv')))
                for split,name in [('train','Train'),('val','Validation'),('test','Test')]:
                    frame=healthy if split=='train' else tgt[split]
                    x,y=windows(frame,fsc,tsc); p=physical(predict(model,x),tsc); true=physical(y,tsc)
                    for option in ('Option 1','Option 2'):
                        yy,pp=(true.ravel(),p.ravel()) if option=='Option 1' else (true[:,-1],p[:,-1])
                        m=metrics(yy.astype('float64'),pp.astype('float64'))
                        old=reference.loc[(reference.Dataset==name)&(reference.Option==option)].iloc[0]
                        rows.append(dict(seed=seed,ratio=ratio,group=tag,split=split,option=option,**m,reference_mse=old.MSE,reference_r2=old.R2,
                                         mse_delta=m['mse']-old.MSE,r2_delta=m['r2']-old.R2))
        print('nominal grid seed',seed,'completed',flush=True)
    grid=pd.DataFrame(rows); grid.to_csv(OUT/'nominal_grid.csv',index=False)
    delta=float(grid[['mse_delta','r2_delta']].abs().max().max())
    CHECKS.append(dict(name='200_target_checkpoints',rows=len(grid),maximum_absolute_delta=delta,tolerance=3e-4,status='PASS' if delta<3e-4 else 'MISMATCH'))
    print(CHECKS[-1],flush=True)
    summary_rows=[]
    for filename,default_group in [('metrics_summary_target.csv','scratch'),('metrics_summary_transfer.csv',None)]:
        reference=pd.read_csv(MODELS/filename)
        for _,old in reference.iterrows():
            tag=default_group if default_group else str(old['Group'])
            for split,prefix in [('val','Val'),('test','Test')]:
                current=grid[(grid.group==tag)&(grid.ratio==old['Ratio (%)'])&(grid.split==split)&(grid.option=='Option 2')]
                if len(current)!=5: raise ValueError('Expected five seeds per nominal summary.')
                for key in ('r2','mse','mape','rmse'):
                    for stat in ('mean','std'):
                        value=float(current[key].mean() if stat=='mean' else current[key].std(ddof=0)); refvalue=float(old[f'{prefix}_{key.upper()}_{stat}'])
                        summary_rows.append(dict(group=tag,ratio=old['Ratio (%)'],split=split,metric=key,stat=stat,value=value,reference=refvalue,delta=value-refvalue))
    summaries=pd.DataFrame(summary_rows);summaries.to_csv(OUT/'nominal_summary_comparison.csv',index=False)
    delta=float(summaries.delta.abs().max());CHECKS.append(dict(name='nominal_mean_std_all_metrics',rows=len(summaries),maximum_absolute_delta=delta,tolerance=3e-4,status='PASS' if delta<3e-4 else 'MISMATCH'))
    print(CHECKS[-1],flush=True)
    # Figure 10 uses the new three-seed fresh validation experiment.
    pd.read_csv(ROOT/'experiments/window_sensitivity/validation_summary.csv').to_csv(OUT/'window_sweep.csv',index=False)
    # Controlled benchmark weights are kept distinct from the archived source model.
    bench=module(PKG/'codes/model_comparison_source_611/run_source_model_comparison.py')
    tensors,bfs,bts,_=bench.load_data(); brefs=pd.read_csv(refdir/'source_model_metrics.csv'); brows=[]; latency=[]
    import xgboost as xgb
    for name,file in [('LSTM','lstm_seed42.pth'),('Transformer','transformer_seed42.pth'),('XGBoost','xgboost_seed42.json')]:
        if name=='XGBoost': model=xgb.XGBRegressor(); model.load_model(refdir/'checkpoints'/file)
        else:
            model=bench.MODEL_FACTORIES[name](18).to(DEVICE); model.load_state_dict(torch.load(refdir/'checkpoints'/file,map_location=DEVICE,weights_only=True)); model.eval()
        for split in ('validation','test'):
            x,y=tensors[split]; p=model.predict(x.numpy().reshape(len(x),-1)) if name=='XGBoost' else bench.predict(model,x,DEVICE)
            row,_,_=bench.metric_row(name,42,split,y.numpy(),p,bts); brows.append(row)
        for batch in (1,64):
            measures=[bench.benchmark_xgboost_latency(model,18,batch)] if name=='XGBoost' else bench.benchmark_latency(model,18,batch,DEVICE)
            for row in measures: row.update(model=name,batch_size=batch,device=str(DEVICE)); latency.append(row)
    bm=pd.DataFrame(brows); bm.to_csv(OUT/'benchmark_metrics.csv',index=False)
    compare('controlled_benchmark',bm,brefs,['model','seed','split'],['mse_last','r2_last','rmse_last','mape_last_percent','mse_all_horizons','r2_all_horizons'])
    pd.DataFrame(latency).to_csv(OUT/'latency_fresh.csv',index=False)

def source():
    s=module(PKG/'codes/model_comparison_source_611/run_source_threshold_consistent.py')
    for split,path,thresholds,filename in [('validation',s.VAL_FAULT,np.round(np.arange(0,6.01,.1),1),'source_threshold_validation.csv'),('test',s.TEST_FAULT,np.array([1.8]),'source_threshold_test_tau1p8.csv')]:
        result,trace=s.evaluate_thresholds(path,thresholds,DEVICE,selected_threshold=1.8); result.to_csv(OUT/filename,index=False); trace.to_csv(OUT/f'source_{split}_trace.csv',index=False)
        compare('source_'+split,result,pd.read_csv(PKG/'analysis'/filename),['threshold'],['accuracy','precision','recall','f1'],1e-10)

def detection():
    from correct_detect import prof as a
    a.TAU_RES=1.8
    source()
    rows=[]; factorial=[]
    for seed in range(42,47):
        for ratio in range(10,81,10):
            a.CHECKPOINT_RATIO_LABEL=f'{ratio}%'; a.TARGET_TRAIN_FRACTION=ratio/80
            for tl in (False,True):
                summary,trace=a.evaluate_configuration(tl,True,seed,DEVICE); summary['ratio']=ratio/100; rows.append(summary)
                if ratio==80:
                    factorial.append(summary.copy()); trace.to_csv(OUT/f'target_seed{seed}_tl{int(tl)}_prof1_trace.csv',index=False)
                    extra,tr=a.evaluate_configuration(tl,False,seed,DEVICE); factorial.append(extra); tr.to_csv(OUT/f'target_seed{seed}_tl{int(tl)}_prof0_trace.csv',index=False)
        pd.DataFrame(rows).to_csv(OUT/'target_detection_by_ratio_tau1p8.csv',index=False)
        print('detection seed',seed,'completed',flush=True)
    raw=pd.DataFrame(rows); fac=pd.DataFrame(factorial); fac.to_csv(OUT/'ablation_by_seed_tau1p8.csv',index=False)
    compare('target_detection',raw,pd.read_csv(PKG/'analysis/target_detection_by_ratio_tau1p8.csv'),['seed','ratio','transfer_learning'],['accuracy','precision','recall','f1'],1e-10)
    compare('factorial',fac,pd.read_csv(PKG/'analysis/ablation_by_seed_tau1p8.csv'),['seed','transfer_learning','prof'],['accuracy','precision','recall','f1'],1e-10)
    for frame,keys,name in [(raw,['transfer_learning','ratio'],'target_detection_by_ratio_summary_tau1p8.csv'),(fac,['method'],'ablation_summary_tau1p8.csv')]:
        agg=frame.groupby(keys)[['accuracy','precision','recall','f1']].agg(['mean','std']); agg.columns=['_'.join(c) for c in agg.columns]; agg.reset_index().to_csv(OUT/name,index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['nominal','detection','source']); args=parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.stage]()
    (OUT/f'{args.stage}_audit.json').write_text(json.dumps(dict(device=str(DEVICE),checks=CHECKS),indent=2),encoding='utf-8')
