import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.lstm import LstmNetwork, LstmParam
from models.bi_rnn import BiRNN
from models.nested_lstm import NestedLSTM  

def get_model(model_class_name):
    
    if model_class_name == "LstmNetwork":
        mem_cell_ct = 100  
        x_dim = 28  
        lstm_param = LstmParam(mem_cell_ct, x_dim)
        return LstmNetwork(lstm_param)

    elif model_class_name == "BiRNN":
        input_dim = 28  
        hidden_dim = 100  
        output_dim = 1  
        num_layers = 2  
        return BiRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    elif model_class_name == "NestedLSTM":
        input_dim = 28  
        hidden_dim = 100  
        output_dim = 1  
        num_layers = 2  
        return NestedLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

    else:
        raise Exception(f"Unknown model class name: {model_class_name}")
