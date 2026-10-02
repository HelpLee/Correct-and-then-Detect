"""Train the archived LSTM architectures, saving corrected reruns separately."""
from pathlib import Path
from copy import deepcopy
import argparse
import json
import random
import time
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import MinMaxScaler

ROOT=Path(__file__).resolve().parent
LEGACY=ROOT/'legacy'
sys.path.insert(0,str(ROOT/'src'))
from correct_detect.models import freeze_layers
PROFILES=json.loads((ROOT/'configs/protocols.json').read_text(encoding='utf-8'))
TARGET='motor6_temperature'

class LSTM(nn.Module):
    def __init__(self,inputs):
        super().__init__()
        self.lstm=nn.LSTM(inputs,64,3,dropout=.2,batch_first=True)
        self.dropout=nn.Dropout(.2)
        self.fc=nn.Linear(64,10)
    def forward(self,x):
        y,_=self.lstm(x)
        return self.fc(self.dropout(y[:,-1,:]))

def windows(frame,fs,ts):
    x=fs.transform(frame.drop(columns=['timestamp'])).astype('float32')
    y=ts.transform(frame[[TARGET]]).astype('float32')[:,0]
    count=len(frame)-39
    if count<1: raise ValueError('At least 40 observations are required.')
    return TensorDataset(torch.from_numpy(np.stack([x[i:i+30] for i in range(count)])),
                         torch.from_numpy(np.stack([y[i+30:i+40] for i in range(count)])))

def train(train_frame,val_frame,fs,ts,source,freeze,seed,epochs,lr,weight_decay,destination,device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    train_loader=DataLoader(windows(train_frame,fs,ts),batch_size=64,shuffle=True)
    val_loader=DataLoader(windows(val_frame,fs,ts),batch_size=64,shuffle=False)
    model=LSTM(len(fs.feature_names_in_)).to(device)
    if source:
        model.load_state_dict(torch.load(source,map_location=device,weights_only=True))
        freeze_layers(model,freeze)
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=lr,weight_decay=weight_decay)
    loss_fn=nn.MSELoss(); best=float('inf'); stale=0; history=[]; state=None
    started=time.perf_counter()
    for epoch in range(epochs):
        model.train(); loss_sum=0
        for x,y in train_loader:
            optimizer.zero_grad(); loss=loss_fn(model(x.to(device)),y.to(device)); loss.backward(); optimizer.step(); loss_sum+=loss.item()
        model.eval(); val=0
        with torch.no_grad():
            for x,y in val_loader: val+=loss_fn(model(x.to(device)),y.to(device)).item()
        val/=len(val_loader)
        history.append({'epoch':epoch+1,'train_loss':loss_sum/len(train_loader),'val_loss':val})
        if val<best: best=val; state=deepcopy(model.state_dict()); stale=0
        else: stale+=1
        print(destination.name,'seed',seed,'epoch',epoch+1,'validation',val,flush=True)
        if stale>=10: break
    destination.mkdir(parents=True,exist_ok=True)
    torch.save(state,destination/'best_model.pth')
    joblib.dump(fs,destination/'feature_scaler.pkl'); joblib.dump(ts,destination/'target_scaler.pkl')
    pd.DataFrame(history).to_csv(destination/'training_history.csv',index=False)
    (destination/'metadata.json').write_text(json.dumps({'seed':seed,'freeze':freeze,'source_checkpoint':str(source) if source else None,
        'training_rows':len(train_frame),'validation_rows':len(val_frame),'hidden_dim':64,'layers':3,'past':30,'future':10,
        'max_epochs':epochs,'learning_rate':lr,'weight_decay':weight_decay,'validation_best_loss':best,
        'snapshot':'deepcopy','device':str(device),'elapsed_seconds':time.perf_counter()-started},indent=2),encoding='utf8')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['source','target'],required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--ratios',nargs='+',type=int,choices=range(10,81,10),default=[80],help='Legacy percent of complete target sequence; 80 = all healthy target training rows')
    p.add_argument('--freeze',nargs='+',choices=['0','01','012','no'],default=['0'])
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--output',type=Path,default=ROOT/'results/generated/training')
    p.add_argument('--source-checkpoint',type=Path,default=LEGACY/'codes/models_reg_v3/source_best_model_v3_b1.pth')
    a=p.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.epochs is not None and a.epochs<1: p.error('--epochs must be positive')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    data=LEGACY/'codes/data'; modeldir=LEGACY/'codes/models_reg_v3'
    if a.mode=='source':
        full=pd.read_csv(data/'source_all_data_b1.csv').dropna(); n=len(full)
        training=full.iloc[:int(.7*n)]; validation=full.iloc[int(.7*n):int(.85*n)]
        fs=MinMaxScaler().fit(training.drop(columns=['timestamp'])); ts=MinMaxScaler().fit(training[[TARGET]])
        cfg=PROFILES['archived_source']
        for seed in a.seeds: train(training,validation,fs,ts,None,'no',seed,a.epochs or cfg['epochs'],cfg['learning_rate'],0,a.output/f'source_seed{seed}',device)
    else:
        full=pd.read_csv(data/'target_all_data_b1.csv').dropna(); n=len(full)
        validation=full.iloc[int(.8*n):int(.9*n)]
        for seed in a.seeds:
            for ratio in a.ratios:
                training=full.iloc[:int((ratio/100)*n)]
                fs=MinMaxScaler().fit(training.drop(columns=['timestamp'])); ts=MinMaxScaler().fit(training[[TARGET]])
                cfg=PROFILES['target_scratch']
                train(training,validation,fs,ts,None,'no',seed,a.epochs or cfg['epochs'],cfg['learning_rate'],cfg['weight_decay'],a.output/f'target_seed{seed}'/f'ratio{ratio}',device)
                fs=joblib.load(modeldir/'feature_scaler_v3_b1.pkl'); ts=joblib.load(modeldir/'target_scaler_v3_b1.pkl')
                for freeze in a.freeze:
                    cfg=PROFILES['transfer']
                    train(training,validation,fs,ts,a.source_checkpoint,freeze,seed,a.epochs or cfg['epochs'],cfg['learning_rate'],cfg['weight_decay'],a.output/f'transfer_seed{seed}'/f'{freeze}_ratio{ratio}',device)

if __name__=='__main__': main()
