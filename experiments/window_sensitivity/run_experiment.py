"""Fresh, leakage-free nominal window sweep; no archived weights or test scoring."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT/'legacy/codes/data/source_all_data_b1.csv'
SEEDS = [42,43,44]
PASTS = [20,30,40,50,60]
FUTURES = [10,15,20,25,30]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class LSTM(nn.Module):
    def __init__(self, future):
        super().__init__()
        self.lstm = nn.LSTM(18,64,3,dropout=.2,batch_first=True)
        self.dropout = nn.Dropout(.2)
        self.fc = nn.Linear(64,future)
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(self.dropout(out[:,-1,:]))

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def segments(frame):
    delta = frame.timestamp.diff().to_numpy()
    cuts = np.r_[0, np.where((delta<0)|(delta>1))[0],len(frame)]
    return [(int(a),int(b)) for a,b in zip(cuts[:-1],cuts[1:])]

def prepare():
    data = pd.read_csv(DATA)
    assert data.shape == (104000,19) and not data.isna().any().any()
    assert np.isfinite(data.to_numpy()).all()
    features = [c for c in data.columns if c != 'timestamp']
    train = data.iloc[:72800].reset_index(drop=True)
    val = data.iloc[72800:88400].reset_index(drop=True)
    assert train.timestamp.max() < val.timestamp.min()
    fs = MinMaxScaler().fit(train[features])
    ts = MinMaxScaler().fit(train[['motor6_temperature']])
    if not (HERE/'protocol.json').exists():
        joblib.dump(fs,HERE/'feature_scaler_train_only.pkl')
        joblib.dump(ts,HERE/'target_scaler_train_only.pkl')
    frames = {'train':train,'validation':val}
    arrays = {k:(fs.transform(v[features]).astype('float32'),ts.transform(v[['motor6_temperature']]).astype('float32').ravel()) for k,v in frames.items()}
    segs = {k:segments(v) for k,v in frames.items()}
    # Every comparison uses exactly the same physical validation decision rows.
    common = np.concatenate([np.arange(a+max(PASTS)+max(FUTURES)-1,b) for a,b in segs['validation'] if b-a>=max(PASTS)+max(FUTURES)])
    if not (HERE/'protocol.json').exists():
        pd.DataFrame({'source_row_zero_based':common+72800,'timestamp':val.timestamp.to_numpy()[common]}).to_csv(HERE/'validation_decision_rows.csv',index=False)
    protocol = dict(data=str(DATA.relative_to(ROOT)),data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                    split_rows={'train':[0,72800],'validation':[72800,88400],'test_reserved':[88400,104000]},
                    features=features,segments=segs,common_validation_decisions=len(common),
                    seeds=SEEDS,history_lengths=PASTS,prediction_horizons=FUTURES,
                    model={'hidden':64,'layers':3,'dropout':.2,'input':18,'direct_multi_output':True},
                    training={'batch_size':64,'optimizer':'Adam','learning_rate':.005,'maximum_epochs':100,'patience':10,'loss':'all-output scaled MSE','early_stopping':'last-output MSE on common validation decisions'},
                    selection='lowest mean validation last-output physical MSE across seeds; same validation decisions for every setting',
                    preprocessing='MinMax fitted only on training partition; no clipping, imputation, archived scaler, or archived checkpoint',
                    discontinuities='window must stay inside one contiguous segment; boundary when timestamp delta <0 or >1 second; repeated integer timestamps permitted',
                    scope='nominal healthy forecasting only, no PROF fault-feedback loop',
                    test_evaluation=False,torch_version=torch.__version__,device=str(DEVICE),
                    gpu=torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else None)
    if not (HERE/'protocol.json').exists():
        (HERE/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    else:
        saved = json.loads((HERE/'protocol.json').read_text())
        current = json.loads(json.dumps(protocol))
        for key in ('data_sha256','split_rows','features','segments','common_validation_decisions','seeds','history_lengths','prediction_horizons','model','training','selection','preprocessing','discontinuities','scope','test_evaluation'):
            assert saved[key] == current[key], f'Protocol field {key} changed; use a new output folder'
    return arrays,segs,common,ts

def windows(arrays,segs,common,past,future):
    tensors={}
    for name,(x,y) in arrays.items():
        ends = common if name=='validation' else np.concatenate([np.arange(a+past+future-1,b) for a,b in segs[name] if b-a>=past+future])
        starts = ends-past-future+1
        xx=np.stack([x[i:i+past] for i in starts])
        yy=np.stack([y[i+past:i+past+future] for i in starts])
        tensors[name]=(torch.from_numpy(xx).to(DEVICE),torch.from_numpy(yy).to(DEVICE))
    return tensors

def train(tensors,past,future,seed,ts):
    seed_everything(seed)
    model=LSTM(future).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=.005)
    tx,ty=tensors['train']; vx,vy=tensors['validation']
    best=float('inf'); stale=0; history=[]; started=time.perf_counter()
    folder=HERE/f'p{past}_h{future}'/f'seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,101):
        model.train(); total=torch.zeros((),device=DEVICE)
        order=torch.randperm(len(tx),device=DEVICE)
        for ids in order.split(64):
            opt.zero_grad(set_to_none=True)
            loss=(model(tx[ids])-ty[ids]).square().mean()
            loss.backward(); opt.step()
            total+=loss.detach()*len(ids)
        model.eval()
        with torch.inference_mode():
            pred=torch.cat([model(b) for b in vx.split(512)])
            score=(pred[:,-1]-vy[:,-1]).square().mean().item()
        assert np.isfinite(score)
        history.append(dict(epoch=epoch,train_scaled_mse=total.item()/len(tx),validation_last_scaled_mse=score))
        if score<best:
            best=score; best_epoch=epoch; stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if epoch%10==0: print(f'  {past}/{future} seed {seed} epoch {epoch}, best val MSE={best/ts.scale_[0]**2:.5f}',flush=True)
        if stale>=10: break
    torch.save(best_state,folder/'best_model.pth')
    pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode():
        pred=torch.cat([model(b) for b in vx.split(512)])[:,-1].cpu().numpy().astype('float64')
    truth=vy[:,-1].cpu().numpy().astype('float64')
    pred=(pred-ts.min_[0])/ts.scale_[0]; truth=(truth-ts.min_[0])/ts.scale_[0]
    residual=truth-pred; mse=float(np.mean(residual**2)); r2=float(1-np.sum(residual**2)/np.sum((truth-truth.mean())**2))
    np.savez_compressed(folder/'validation_predictions.npz',true=truth,predicted=pred)
    row=dict(n_past=past,n_future=future,seed=seed,validation_mse=mse,validation_r2=r2,validation_rmse=mse**.5,
             validation_mape_percent=float(np.mean(np.abs(residual/truth))*100),train_windows=len(tx),validation_windows=len(vx),best_epoch=best_epoch,epochs_run=epoch,seconds=time.perf_counter()-started)
    (folder/'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    return row

def main():
    global HERE
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=SEEDS,default=42,
                        help='One seed per combination; default 42. Completed runs are reused.')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/generated/window_sensitivity',
                        help='Save fresh runs separately from the published experiment.')
    args=parser.parse_args()
    HERE = args.output_dir.resolve()
    selected_seeds=[args.seed] if args.seed is not None else SEEDS
    suffix=f'_seed{args.seed}' if args.seed is not None else '_per_seed'
    HERE.mkdir(parents=True,exist_ok=True)
    arrays,segs,common,ts=prepare()
    rows=[]
    print(f'Start 25 combinations x seeds {selected_seeds} on {DEVICE}; {len(common)} shared validation decisions',flush=True)
    for past in PASTS:
        for future in FUTURES:
            tensors=windows(arrays,segs,common,past,future)
            for seed in selected_seeds:
                path=HERE/f'p{past}_h{future}'/f'seed{seed}'/'metrics.json'
                row=json.loads(path.read_text()) if path.exists() else train(tensors,past,future,seed,ts)
                rows.append(row)
                pd.DataFrame(rows).to_csv(HERE/f'validation_results{suffix}.csv',index=False)
                print(f'DONE {len(rows)}/{25*len(selected_seeds)}: {past}/{future} seed={seed} R2={row["validation_r2"]:.6f} MSE={row["validation_mse"]:.6f}',flush=True)
            del tensors
    if args.seed is not None: return
    d=pd.DataFrame(rows)
    summary=d.groupby(['n_past','n_future']).agg(n_seeds=('seed','count'),mse_mean=('validation_mse','mean'),mse_sd=('validation_mse','std'),r2_mean=('validation_r2','mean'),r2_sd=('validation_r2','std')).reset_index()
    summary['rank']=summary.mse_mean.rank(method='min').astype(int)
    summary.to_csv(HERE/'validation_summary.csv',index=False)
    print(summary.sort_values('rank').to_string(index=False),flush=True)

if __name__=='__main__': main()
