import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphWaveNet(nn.Module):
    """Ultra-simple GraphWaveNet without complex operations"""
    def __init__(self, num_nodes, in_dim=1, out_dim=1, hidden_dim=32, 
                 num_layers=8, kernel_size=2, horizon=12, supports=None):
        super(GraphWaveNet, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        # Simple MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Graph attention weights
        self.graph_weight = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.01)
        
        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, num_nodes * horizon)
        )
        
    def forward(self, x, adj_matrix=None):
        # x shape: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")
        
        # Simple graph attention
        attn = torch.softmax(self.graph_weight, dim=1)
        if adj_matrix is not None:
            # Use provided adjacency as guidance
            adj = torch.tensor(adj_matrix, device=x.device, dtype=torch.float32)
            adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)
            attn = 0.5 * attn + 0.5 * adj
        
        # Apply graph attention to smooth features
        x_smooth = []
        for i in range(seq_len):
            # For each time step, mix node features using attention
            x_t = x[:, i, :]  # (batch, num_nodes)
            x_t_smooth = torch.matmul(x_t, attn.t())  # (batch, num_nodes)
            x_smooth.append(x_t_smooth)
        
        x = torch.stack(x_smooth, dim=1)  # (batch, seq_len, num_nodes)
        
        # Encode each time step
        x_encoded = []
        for i in range(seq_len):
            x_t = x[:, i, :]  # (batch, num_nodes)
            x_t_enc = self.encoder(x_t)  # (batch, hidden_dim)
            x_encoded.append(x_t_enc)
        
        x_encoded = torch.stack(x_encoded, dim=1)  # (batch, seq_len, hidden_dim)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(x_encoded)  # (batch, seq_len, hidden_dim)
        
        # Use last hidden state
        h_last = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Decode to predictions
        output = self.decoder(h_last)  # (batch, num_nodes * horizon)
        output = output.view(batch_size, self.horizon, num_nodes)
        
        return output