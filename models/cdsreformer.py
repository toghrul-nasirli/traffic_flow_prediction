import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CDSAttention(nn.Module):
    """Cross-Dimensional Sparse Attention"""
    def __init__(self, d_model, num_heads, num_nodes, dropout=0.1):
        super(CDSAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_nodes = num_nodes
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Learnable sparse masks
        self.temporal_mask = nn.Parameter(torch.ones(1, 1))
        self.spatial_mask = nn.Parameter(torch.ones(1, num_nodes, num_nodes))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, num_nodes, d_model = x.shape
        
        # Reshape for cross-dimensional attention
        x_flat = x.reshape(batch_size, seq_len * num_nodes, d_model)
        
        Q = self.W_q(x_flat).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x_flat).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x_flat).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply learned sparse masks
        scores = scores.view(batch_size, self.num_heads, seq_len, num_nodes, seq_len, num_nodes)
        
        # Apply temporal mask
        temporal_mask = torch.sigmoid(self.temporal_mask)
        scores = scores * temporal_mask.view(1, 1, 1, 1, 1, 1)
        
        # Apply spatial mask
        spatial_mask = torch.sigmoid(self.spatial_mask)
        scores = scores * spatial_mask.view(1, 1, 1, num_nodes, 1, num_nodes)
        
        scores = scores.view(batch_size, self.num_heads, seq_len * num_nodes, seq_len * num_nodes)
        
        # Softmax and dropout
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len * num_nodes, self.d_model
        )
        
        output = self.W_o(context)
        output = output.view(batch_size, seq_len, num_nodes, d_model)
        
        return output

class CDSReFormerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_nodes, dropout=0.1):
        super(CDSReFormerBlock, self).__init__()
        self.cds_attention = CDSAttention(d_model, num_heads, num_nodes, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Cross-dimensional sparse attention
        attn_out = self.cds_attention(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed forward network
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        
        return x

class CDSReFormer(nn.Module):
    def __init__(self, num_nodes, input_dim=1, d_model=64, num_heads=8, 
                 num_layers=3, horizon=12, dropout=0.1):
        super(CDSReFormer, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Learnable embeddings
        self.temporal_embedding = nn.Parameter(torch.randn(1, 100, 1, d_model))
        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, num_nodes, d_model))
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            CDSReFormerBlock(d_model, num_heads, num_nodes, dropout)
            for _ in range(num_layers)
        ])
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, horizon)
        )
        
    def forward(self, x, adj_matrix=None):
        # x shape: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        
        # Add feature dimension
        x = x.unsqueeze(-1)  # (batch, seq_len, num_nodes, 1)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add embeddings
        x = x + self.temporal_embedding[:, :seq_len, :, :]
        x = x + self.spatial_embedding
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Generate predictions
        # Use adaptive pooling over time dimension
        x = x.mean(dim=1)  # (batch, num_nodes, d_model)
        
        # Prediction head
        out = self.prediction_head(x)  # (batch, num_nodes, horizon)
        out = out.permute(0, 2, 1)  # (batch, horizon, num_nodes)
        
        return out