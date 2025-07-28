# staeformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SimpleAttention, self).__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Simplified single-head attention
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, V)
        return output

class SimplifiedSTBlock(nn.Module):
    def __init__(self, d_model, num_nodes, dropout=0.1):
        super(SimplifiedSTBlock, self).__init__()
        
        # Simplified spatial-temporal processing
        self.spatial_conv = nn.Conv2d(d_model, d_model, kernel_size=(1, 3), padding=(0, 1))
        self.temporal_conv = nn.Conv2d(d_model, d_model, kernel_size=(3, 1), padding=(1, 0))
        
        self.attention = SimpleAttention(d_model, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Smaller FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, time, nodes, features)
        batch_size, time_steps, num_nodes, features = x.shape
        
        # Permute for conv operations
        x_conv = x.permute(0, 3, 1, 2)  # (batch, features, time, nodes)
        
        # Spatial-temporal convolutions
        spatial_out = self.spatial_conv(x_conv)
        temporal_out = self.temporal_conv(spatial_out)
        
        # Back to original shape
        x_st = temporal_out.permute(0, 2, 3, 1)  # (batch, time, nodes, features)
        
        # Residual connection
        x = x + self.dropout(x_st)
        x = self.norm1(x)
        
        # Simplified attention on flattened spatial-temporal features
        b, t, n, f = x.shape
        x_flat = x.reshape(b, t * n, f)
        x_att = self.attention(x_flat)
        x_att = x_att.reshape(b, t, n, f)
        
        # Residual and FFN
        x = x + self.dropout(x_att)
        x = self.norm2(x)
        
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)
        
        return x

# Keep the same class name so your train.py doesn't need changes
class STAEformer(nn.Module):
    def __init__(self, num_nodes, input_dim=1, d_model=None, num_heads=None, 
                 num_layers=None, horizon=12, dropout=0.1):
        super(STAEformer, self).__init__()
        
        # Use simplified parameters regardless of what's passed
        # This ensures compatibility with your train.py
        d_model = 32  # Force smaller dimension
        num_layers = 2  # Force fewer layers
        
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Simple learnable embeddings
        self.temporal_embedding = nn.Parameter(torch.randn(1, 100, 1, d_model) * 0.02)
        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, num_nodes, d_model) * 0.02)
        
        # Fewer layers
        self.blocks = nn.ModuleList([
            SimplifiedSTBlock(d_model, num_nodes, dropout)
            for _ in range(num_layers)
        ])
        
        # Direct output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, adj_matrix=None):
        # x shape: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        
        # Add feature dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)  # (batch, seq_len, num_nodes, 1)
        
        # Input embedding
        x = self.input_embedding(x)
        
        # Add embeddings
        x = x + self.temporal_embedding[:, :seq_len, :, :]
        x = x + self.spatial_embedding
        
        # Apply blocks
        for block in self.blocks:
            x = block(x)
        
        # Use mean of last few timesteps for more stable prediction
        if seq_len >= 3:
            x_final = x[:, -3:, :, :].mean(dim=1)  # Average last 3 timesteps
        else:
            x_final = x[:, -1, :, :]  # Use last timestep
        
        # Output projection
        out = self.output_projection(x_final)  # (batch, num_nodes, horizon)
        out = out.permute(0, 2, 1)  # (batch, horizon, num_nodes)
        
        return out