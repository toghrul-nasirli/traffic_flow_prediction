import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class TGCNCell(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, K=3):
        super(TGCNCell, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        
        # Graph convolution for input
        self.graph_conv1 = ChebConv(in_channels, hidden_channels, K)
        # Graph convolution for hidden state
        self.graph_conv2 = ChebConv(hidden_channels, hidden_channels, K)
        
        # GRU gates
        self.update_gate = nn.Linear(2 * hidden_channels, hidden_channels)
        self.reset_gate = nn.Linear(2 * hidden_channels, hidden_channels)
        self.candidate = nn.Linear(2 * hidden_channels, hidden_channels)
        
    def forward(self, x, edge_index, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
            
        # Graph convolutions
        input_conv = self.graph_conv1(x, edge_index)
        hidden_conv = self.graph_conv2(h, edge_index)
        
        # Concatenate
        combined = torch.cat([input_conv, hidden_conv], dim=1)
        
        # GRU computations
        z = torch.sigmoid(self.update_gate(combined))
        r = torch.sigmoid(self.reset_gate(combined))
        
        combined_candidate = torch.cat([input_conv, r * hidden_conv], dim=1)
        h_tilde = torch.tanh(self.candidate(combined_candidate))
        
        h_new = z * h + (1 - z) * h_tilde
        
        return h_new

class TGCN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, horizon, K=3):
        super(TGCN, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        self.tgcn_cell = TGCNCell(num_nodes, input_dim, hidden_dim, K)
        self.output_layer = nn.Linear(hidden_dim, output_dim * horizon)
        
    def forward(self, x, edge_index):
        # x shape: (batch, seq_len, num_nodes, features)
        batch_size, seq_len, num_nodes, _ = x.shape
        
        h = None
        for t in range(seq_len):
            # Reshape for graph processing
            x_t = x[:, t, :, :].reshape(-1, x.size(-1))
            h = self.tgcn_cell(x_t, edge_index, h)
            
        # Output prediction
        out = self.output_layer(h)
        out = out.reshape(batch_size, num_nodes, self.horizon, -1)
        out = out.permute(0, 2, 1, 3).squeeze(-1)
        
        return out