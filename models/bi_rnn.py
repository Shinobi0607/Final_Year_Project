import torch
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        
        super(BiRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, output_dim)  

    def forward(self, x):
        
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim)  
        rnn_out, _ = self.rnn(x, h0)
        
        out = self.fc(rnn_out[:, -1, :])  
        return out

    def get_weights(self):
        
        return {name: param.detach().cpu().numpy() for name, param in self.state_dict().items()}

    def set_weights(self, weights):
        
        state_dict = {name: torch.tensor(value) for name, value in weights.items()}
        self.load_state_dict(state_dict)
