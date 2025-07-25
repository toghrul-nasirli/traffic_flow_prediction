import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, horizon=12):
        super(LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        self.lstm = nn.LSTM(
            input_size=num_nodes * input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        self.fc = nn.Linear(hidden_dim, num_nodes * horizon)
        
    def forward(self, x):
        # x shape: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        
        # Reshape for LSTM
        x = x.reshape(batch_size, seq_len, -1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        out = self.fc(lstm_out[:, -1, :])
        
        # Reshape output
        out = out.reshape(batch_size, self.horizon, num_nodes)
        
        return out