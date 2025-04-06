import torch
import torch.nn as nn

class NestedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        
        super(NestedLSTM, self).__init__()
        self.nested_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        
        lstm_out, _ = self.nested_lstm(x)
        output = self.fc(lstm_out[:, -1, :])  
        return output
