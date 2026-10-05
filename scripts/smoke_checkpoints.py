from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from correct_detect.models import CheckpointLSTM
torch.set_num_threads(4)
paths=sorted((ROOT/'legacy/codes/models_reg_v3').rglob('*.pth'))+sorted((ROOT/'experiments/window_sensitivity').rglob('*.pth'))
if not paths: raise FileNotFoundError('Extract the accompanying artifact bundle before running checkpoint checks.')
for path in paths:
    state=torch.load(path,map_location='cpu',weights_only=True);model=CheckpointLSTM(state);model.load_state_dict(state);model.eval()
    future=state[model.name+'.weight'].shape[0]
    with torch.inference_mode(): out=model(torch.zeros(1,30,18))
    assert out.shape==(1,future) and torch.isfinite(out).all(),path
print(f'PASS: loaded and inferred {len(paths)} archived LSTM checkpoints.')
