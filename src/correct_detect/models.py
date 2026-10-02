"""Load archived fc and controlled head LSTMs without conflating their weights."""
from torch import nn

class CheckpointLSTM(nn.Module):
    def __init__(self, state):
        super().__init__()
        gates, inputs = state['lstm.weight_ih_l0'].shape
        layers = len([key for key in state if key.startswith('lstm.weight_ih_l')])
        self.lstm = nn.LSTM(inputs, gates // 4, layers, batch_first=True, dropout=.2)
        self.name = 'fc' if 'fc.weight' in state else 'head'
        setattr(self, self.name, nn.Linear(gates // 4, state[self.name + '.weight'].shape[0]))

    def forward(self, x):
        output, _ = self.lstm(x)
        return getattr(self, self.name)(output[:, -1])

def freeze_layers(model, tag):
    """Freeze exactly the indicated recurrent layers; the forecast head adapts."""
    if tag not in {'0', '01', '012', 'no'}:
        raise ValueError(f'Unknown freeze profile: {tag}')
    for name, parameter in model.lstm.named_parameters():
        parameter.requires_grad = not any(f'_l{layer}' in name for layer in tag if layer.isdigit())
