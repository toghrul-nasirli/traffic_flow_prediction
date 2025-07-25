import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        
        return output, attention

class SpatialAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_nodes):
        super(SpatialAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.num_nodes = num_nodes
        
    def forward(self, x, adj_matrix=None):
        # x shape: (batch, time, nodes, features)
        batch_size, time_steps, num_nodes, features = x.shape
        
        # Reshape for spatial attention
        x = x.permute(0, 1, 2, 3).reshape(batch_size * time_steps, num_nodes, features)
        
        # Apply attention
        out, _ = self.attention(x, x, x)
        
        # Reshape back
        out = out.reshape(batch_size, time_steps, num_nodes, features)
        
        return out

class TemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(TemporalAttention, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        
    def forward(self, x):
        # x shape: (batch, time, nodes, features)
        batch_size, time_steps, num_nodes, features = x.shape
        
        # Reshape for temporal attention
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, time_steps, features)
        
        # Apply attention
        out, _ = self.attention(x, x, x)
        
        # Reshape back
        out = out.reshape(batch_size, num_nodes, time_steps, features)
        out = out.permute(0, 2, 1, 3)
        
        return out

class STAEformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_nodes, dropout=0.1):
        super(STAEformerBlock, self).__init__()
        self.spatial_attention = SpatialAttention(d_model, num_heads, num_nodes)
        self.temporal_attention = TemporalAttention(d_model, num_heads)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_matrix=None):
        # Spatial attention
        x_spatial = self.spatial_attention(x, adj_matrix)
        x = x + self.dropout(x_spatial)
        x = self.norm1(x)
        
        # Temporal attention
        x_temporal = self.temporal_attention(x)
        x = x + self.dropout(x_temporal)
        x = self.norm2(x)
        
        # FFN
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)
        x = self.norm3(x)
        
        return x

class STAEformer(nn.Module):
    def __init__(self, num_nodes, input_dim=1, d_model=64, num_heads=8, 
                 num_layers=3, horizon=12, dropout=0.1):
        super(STAEformer, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 100, 1, d_model)  # max seq len = 100
        )
        
        # Spatial embedding
        self.spatial_embedding = nn.Parameter(
            torch.randn(1, 1, num_nodes, d_model)
        )
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            STAEformerBlock(d_model, num_heads, num_nodes, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(d_model, horizon)
        
    def forward(self, x, adj_matrix=None):
        # x shape: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        
        # Add feature dimension
        x = x.unsqueeze(-1)  # (batch, seq_len, num_nodes, 1)
        
        # Input embedding
        x = self.input_embedding(x)  # (batch, seq_len, num_nodes, d_model)
        
        # Add positional and spatial embeddings
        x = x + self.positional_encoding[:, :seq_len, :, :]
        x = x + self.spatial_embedding
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, adj_matrix)
            
        # Output projection
        out = self.output_layer(x[:, -1, :, :])  # Use last time step
        out = out.permute(0, 2, 1)  # (batch, horizon, num_nodes)
        
        return out