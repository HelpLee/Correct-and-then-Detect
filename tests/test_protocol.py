import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from correct_detect.data import windows
from correct_detect.models import CheckpointLSTM,freeze_layers
from correct_detect import prof

class ProtocolTests(unittest.TestCase):
    def test_windows_do_not_cross_partition_or_use_future_targets(self):
        d=pd.DataFrame({'timestamp':np.arange(80),'motor6_temperature':np.arange(80)})
        fs=MinMaxScaler().fit(d[['motor6_temperature']]);ts=MinMaxScaler().fit(d[['motor6_temperature']])
        x,y=windows(d.iloc[:40],fs,ts)
        self.assertEqual(tuple(x.shape),(1,30,1))
        np.testing.assert_allclose(ts.inverse_transform(x[0,:,0,None]),np.arange(30)[:,None],atol=1e-5)
        np.testing.assert_allclose(ts.inverse_transform(y.reshape(-1,1)),np.arange(30,40)[:,None],atol=1e-5)
        with self.assertRaises(ValueError):windows(d.iloc[40:79],fs,ts)

    def test_frozen_layers_and_independent_best_snapshot(self):
        from copy import deepcopy
        class M(nn.Module):
            def __init__(self):
                super().__init__();self.lstm=nn.LSTM(18,64,3,batch_first=True);self.fc=nn.Linear(64,10)
        m=M();freeze_layers(m,'01')
        for name,p in m.lstm.named_parameters():self.assertEqual(p.requires_grad,'_l2' in name)
        self.assertTrue(m.fc.weight.requires_grad)
        state=deepcopy(m.state_dict());loaded=CheckpointLSTM(state);loaded.load_state_dict(state)
        before=state['fc.weight'].clone()
        with torch.no_grad():m.fc.weight.add_(1)
        torch.testing.assert_close(state['fc.weight'],before)

    def test_prof_preserves_raw_truth_and_corrects_only_future_feedback(self):
        class LastValue(nn.Module):
            def load_state_dict(self,*args,**kwargs):pass
            def forward(self,x):return x[:,-1,0,None].repeat(1,10)
        raw=np.ones(90,dtype='float64');raw[39:45]=5
        labels=np.zeros(90,dtype=int);labels[39:45]=1
        feature=(raw/10)[:,None];scaled=raw/10
        ts=MinMaxScaler().fit(pd.DataFrame({'motor6_temperature':[0,10]}))
        def data(_):return feature,scaled,labels,0,ts
        path=Path(__file__).resolve()
        with patch.object(prof,'load_data_and_scalers',data),patch.object(prof,'checkpoint_path',return_value=path),patch.object(prof,'LSTMModel',return_value=LastValue()),patch.object(prof,'TAU_RES',1.8),patch.object(prof,'CODES_DIR',path.parent),patch.object(prof.torch,'load',return_value={}):
            corrected,trace=prof.evaluate_configuration(False,True,42,torch.device('cpu'))
            baseline,btrace=prof.evaluate_configuration(False,False,42,torch.device('cpu'))
        np.testing.assert_array_equal(feature[:,0],raw/10);np.testing.assert_array_equal(scaled,raw/10)
        self.assertEqual(trace.decision_index.iloc[0],39)
        np.testing.assert_allclose(trace.observed_temperature,raw[39:],atol=1e-6)
        self.assertEqual(corrected['fp'],0);self.assertGreater(baseline['fp'],0)
        self.assertEqual(corrected['tp'],6)
        # The first correction can enter the input only at window index 10.
        np.testing.assert_allclose(trace.predicted_temperature.iloc[:10],btrace.predicted_temperature.iloc[:10])

if __name__=='__main__':unittest.main()
