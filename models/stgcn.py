import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=(kernel_size, 1), 
                             padding=(kernel_size//2, 0))
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        
    def forward(self, x):
        # x shape: (batch, channels, time, nodes)
        out = F.relu(self.conv(x))
        res = self.residual(x)
        return F.relu(out + res)

class SpatialConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, K=3):
        super(SpatialConvLayer, self).__init__()
        self.cheb_conv = ChebConv(in_channels, out_channels, K)
        
    def forward(self, x, edge_index):
        # x shape: (batch, channels, time, nodes)
        batch_size, channels, time_steps, num_nodes = x.shape
        
        # Reshape for graph convolution
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch, time, nodes, channels)
        x = x.view(-1, channels)  # (batch*time*nodes, channels)
        
        # Apply graph convolution
        out = self.cheb_conv(x, edge_index)
        
        # Reshape back
        out = out.view(batch_size, time_steps, num_nodes, -1)
        out = out.permute(0, 3, 1, 2)  # (batch, channels, time, nodes)
        
        return F.relu(out)

class STConvBlock(nn.Module):
    def __init__(self, in_channels, spatial_channels, temporal_channels, 
                 num_nodes, K=3):
        super(STConvBlock, self).__init__()
        self.temporal_conv1 = TemporalConvLayer(in_channels, temporal_channels)
        self.spatial_conv = SpatialConvLayer(temporal_channels, spatial_channels, num_nodes, K)
        self.temporal_conv2 = TemporalConvLayer(spatial_channels, temporal_channels)
        self.layer_norm = nn.LayerNorm(temporal_channels)  # Normalize over channels
        
    def forward(self, x, edge_index):
        # Temporal convolution
        out = self.temporal_conv1(x)
        # Spatial convolution
        out = self.spatial_conv(out, edge_index)
        # Temporal convolution
        out = self.temporal_conv2(out)
        # Layer normalization
        # out shape: (batch, channels, time, nodes)
        out = out.permute(0, 2, 3, 1)  # (batch, time, nodes, channels)
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2)  # (batch, channels, time, nodes)
        return out

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels=1, hidden_channels=64, 
                 out_channels=1, num_layers=2, K=3, horizon=12):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        
        self.blocks = nn.ModuleList()
        self.blocks.append(STConvBlock(in_channels, hidden_channels, 
                                      hidden_channels, num_nodes, K))
        
        for _ in range(num_layers - 1):
            self.blocks.append(STConvBlock(hidden_channels, hidden_channels, 
                                          hidden_channels, num_nodes, K))
            
        self.output_layer = nn.Conv2d(hidden_channels, horizon, 
                                     kernel_size=(1, 1))
        
    def forward(self, x, edge_index):
        # x shape: (batch, seq_len, num_nodes)
        # Add channel dimension and permute
        x = x.unsqueeze(1).permute(0, 1, 2, 3)  # (batch, 1, seq_len, num_nodes)
        
        for block in self.blocks:
            x = block(x, edge_index)
            
        # Output layer
        out = self.output_layer(x)  # (batch, horizon, seq_len, num_nodes)
        out = out[:, :, -1, :]     # (batch, horizon, num_nodes)
        return out