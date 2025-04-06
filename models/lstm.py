import torch
import torch.nn as nn
import torch.optim as optim

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, c_prev):
        
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        
        
        if h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
        
        
        combined = torch.cat([x_t, h_prev], dim=1)
        
        f_gate = torch.sigmoid(self.forget_gate(combined))
        i_gate = torch.sigmoid(self.input_gate(combined))
        c_hat = torch.tanh(self.cell_gate(combined))
        o_gate = torch.sigmoid(self.output_gate(combined))
        
        
        c_t = f_gate * c_prev + i_gate * c_hat
        h_t = o_gate * torch.tanh(c_t)
        
        return h_t, c_t

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.cell = LSTMCell(input_dim, hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, inputs):
        
        batch_size = inputs.size(0)
        
        h_prev = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        c_prev = torch.zeros(batch_size, self.hidden_dim, device=inputs.device)
        
        for t in range(inputs.size(1)):
            x_t = inputs[:, t, :]  
            h_prev, c_prev = self.cell(x_t, h_prev, c_prev)
        
        
        output = self.output_layer(h_prev)
        
        return output

    def get_weights(self):
        return {name: param.data for name, param in self.state_dict().items()}