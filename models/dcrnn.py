import torch
import torch.nn as nn

class DCRNN(nn.Module):
    def __init__(self, num_nodes, input_dim=1, hidden_dim=64, output_dim=1, horizon=12):
        super(DCRNN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.horizon = horizon
        # One GRU per node (processes each node's time series independently)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adj_matrix=None):
        # x: (batch, seq_len, num_nodes)
        batch, seq_len, num_nodes = x.shape
        assert num_nodes == self.num_nodes, f"Input num_nodes {num_nodes} != model num_nodes {self.num_nodes}"
        # Reshape to (batch * num_nodes, seq_len, 1)
        x = x.permute(0, 2, 1).contiguous().view(batch * num_nodes, seq_len, 1)
        out, _ = self.gru(x)  # (batch * num_nodes, seq_len, hidden_dim)
        # Take last hidden state
        last_hidden = out[:, -1, :]  # (batch * num_nodes, hidden_dim)
        pred = self.proj(last_hidden)  # (batch * num_nodes, output_dim)
        # Repeat prediction for horizon steps
        pred = pred.view(batch, num_nodes, self.output_dim)
        pred = pred.permute(0, 2, 1)  # (batch, output_dim, num_nodes)
        pred = pred.unsqueeze(1).repeat(1, self.horizon, 1, 1)  # (batch, horizon, output_dim, num_nodes)
        pred = pred.squeeze(2) if self.output_dim == 1 else pred  # (batch, horizon, num_nodes)
        return pred