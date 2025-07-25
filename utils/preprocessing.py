import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
import networkx as nx

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load PEMS CSV data"""
        df = pd.read_csv(self.data_path)
        return df
    
    def create_adjacency_matrix(self, df):
        """Create adjacency matrix from edge list"""
        # Get unique nodes
        nodes_from = set(df['from'].unique())
        nodes_to = set(df['to'].unique())
        nodes = sorted(list(nodes_from | nodes_to))
        n_nodes = len(nodes)
        
        print(f"Number of unique nodes in graph: {n_nodes}")
        print(f"Node IDs range: {min(nodes)} to {max(nodes)}")
        
        # Create mapping from node ID to index
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Initialize adjacency matrix
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Fill adjacency matrix with costs (inverse for similarity)
        for _, row in df.iterrows():
            i = node_to_idx[row['from']]
            j = node_to_idx[row['to']]
            # Use inverse of cost as edge weight (closer nodes have higher weight)
            adj_matrix[i, j] = 1.0 / (row['cost'] + 1e-6)
            adj_matrix[j, i] = adj_matrix[i, j]  # Symmetric for undirected graph
            
        # Add self-loops with weight 1
        np.fill_diagonal(adj_matrix, 1.0)
        
        return adj_matrix, node_to_idx, nodes
    
    def generate_synthetic_traffic_data(self, n_nodes, n_timesteps=2000):
        """Generate synthetic traffic flow data for demonstration"""
        # Since we only have edge data, we'll generate synthetic traffic patterns
        np.random.seed(42)
        
        # Generate base traffic patterns
        time = np.arange(n_timesteps)
        daily_pattern = 50 + 30 * np.sin(2 * np.pi * time / 288)  # Daily cycle (288 = 24h * 12 (5-min intervals))
        weekly_pattern = 10 * np.sin(2 * np.pi * time / (288 * 7))  # Weekly cycle
        
        # Generate traffic data for each node
        traffic_data = []
        for i in range(n_nodes):
            # Add node-specific variations
            node_pattern = daily_pattern + weekly_pattern
            node_pattern += np.random.normal(0, 5, n_timesteps)  # Random noise
            node_pattern += 20 * np.sin(2 * np.pi * time / 288 + np.random.rand() * 2 * np.pi)  # Phase shift
            # Ensure non-negative values
            node_pattern = np.maximum(node_pattern, 0)
            traffic_data.append(node_pattern)
            
        traffic_data = np.array(traffic_data).T  # Shape: (timesteps, nodes)
        return traffic_data
    
    def z_score_normalize(self, data):
        """Apply Z-score normalization"""
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / (self.std + 1e-8)
    
    def z_score_denormalize(self, data):
        """Denormalize Z-score normalized data"""
        return data * self.std + self.mean
    
    def create_sequences(self, data, seq_len, horizon, step=1):
        """Create sequences for time series prediction"""
        sequences = []
        targets = []
        
        for i in range(0, len(data) - seq_len - horizon + 1, step):
            sequences.append(data[i:i+seq_len])
            targets.append(data[i+seq_len:i+seq_len+horizon])
            
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, seq_len=12, horizons=[3, 12], train_ratio=0.7, val_ratio=0.1):
        """Prepare data for training"""
        # Load edge data
        df = self.load_data()
        print(f"Loaded {len(df)} edges from CSV")
        
        # Create adjacency matrix
        adj_matrix, node_mapping, node_list = self.create_adjacency_matrix(df)
        n_nodes = len(node_mapping)
        
        print(f"Created adjacency matrix of shape: {adj_matrix.shape}")
        
        # Generate synthetic traffic data
        traffic_data = self.generate_synthetic_traffic_data(n_nodes)
        print(f"Generated traffic data of shape: {traffic_data.shape}")
        
        # Normalize data
        normalized_data = self.z_score_normalize(traffic_data)
        
        # Split data
        n_samples = len(normalized_data)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        train_data = normalized_data[:train_size]
        val_data = normalized_data[train_size:train_size+val_size]
        test_data = normalized_data[train_size+val_size:]
        
        print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Create sequences for different horizons
        data_dict = {}
        for horizon in horizons:
            train_x, train_y = self.create_sequences(train_data, seq_len, horizon)
            val_x, val_y = self.create_sequences(val_data, seq_len, horizon)
            test_x, test_y = self.create_sequences(test_data, seq_len, horizon)
            
            print(f"Horizon {horizon} - Train sequences: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}")
            
            data_dict[f'horizon_{horizon}'] = {
                'train': (train_x, train_y),
                'val': (val_x, val_y),
                'test': (test_x, test_y)
            }
        
        return data_dict, adj_matrix, self.mean, self.std