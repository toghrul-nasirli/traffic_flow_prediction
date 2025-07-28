# cdsreformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CDSReFormer(nn.Module):
    """Ultra-light CDSReFormer - as fast as LSTM"""
    def __init__(self, num_nodes, input_dim=1, d_model=None, num_heads=None, 
                 num_layers=None, horizon=12, dropout=0.1):
        super(CDSReFormer, self).__init__()
        
        # Ultra-light configuration
        d_model = 16  # Very small hidden dimension
        
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        # Efficient 1D convolutions for cross-dimensional modeling
        self.temporal_conv = nn.Conv1d(num_nodes, d_model, kernel_size=3, padding=1)
        self.spatial_conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        # Simple RNN for sequential processing
        self.rnn = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        
        # Output layers
        self.output_linear = nn.Linear(d_model, horizon * num_nodes)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, adj_matrix=None):
        # x shape: (batch, seq_len, num_nodes)
        batch_size, seq_len, num_nodes = x.shape
        
        # Temporal convolution
        x_conv = x.transpose(1, 2)  # (batch, num_nodes, seq_len)
        x_temporal = self.temporal_conv(x_conv)  # (batch, d_model, seq_len)
        x_temporal = self.activation(x_temporal)
        
        # Spatial convolution
        x_spatial = self.spatial_conv(x_temporal)  # (batch, d_model, seq_len)
        x_spatial = self.dropout(x_spatial)
        
        # Transpose back for RNN
        x_rnn_input = x_spatial.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # RNN processing
        rnn_out, _ = self.rnn(x_rnn_input)
        
        # Use last output
        last_out = rnn_out[:, -1, :]  # (batch, d_model)
        
        # Generate predictions
        output = self.output_linear(last_out)  # (batch, horizon * num_nodes)
        output = output.view(batch_size, self.horizon, num_nodes)
        
        return output