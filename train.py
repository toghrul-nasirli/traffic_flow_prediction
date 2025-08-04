#train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import yaml
import os
from utils.preprocessing import DataPreprocessor
from utils.metrics import calculate_metrics
from models.lstm import LSTM
from models.tgcn import TGCN
from models.stgcn import STGCN
from models.graphwavenet import GraphWaveNet
from models.staeformer import STAEformer
from models.cdsreformer import CDSReFormer
from models.dcrnn import DCRNN

class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def create_model(self, model_name, num_nodes, horizon):
        """Create model based on name"""
        if model_name == 'LSTM':
            return LSTM(num_nodes=num_nodes, horizon=horizon).to(self.device)
        elif model_name == 'T-GCN':
            return TGCN(num_nodes=num_nodes, input_dim=1, hidden_dim=64, 
                       output_dim=1, horizon=horizon).to(self.device)
        elif model_name == 'ST-GCN':
            return STGCN(num_nodes=num_nodes, horizon=horizon).to(self.device)
        elif model_name == 'GraphWaveNet':
            return GraphWaveNet(num_nodes=num_nodes, horizon=horizon).to(self.device)
        elif model_name == 'STAEformer':
            return STAEformer(num_nodes=num_nodes, horizon=horizon).to(self.device)
        elif model_name == 'CDSReFormer':
            return CDSReFormer(num_nodes=num_nodes, horizon=horizon).to(self.device)
        elif model_name == 'DCRNN':
            return DCRNN(num_nodes=num_nodes, horizon=horizon).to(self.device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_epoch(self, model, dataloader, criterion, optimizer, adj_matrix=None):
        """Train for one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass (handle different model types)
                if isinstance(model, LSTM):
                    output = model(batch_x)
                elif isinstance(model, (TGCN, STGCN)):
                    # These models need edge_index format
                    edge_index = self.adj_to_edge_index(adj_matrix)
                    if isinstance(model, TGCN):
                        # TGCN expects an extra dimension
                        output = model(batch_x.unsqueeze(-1), edge_index)
                    else:
                        # STGCN handles dimensions internally
                        output = model(batch_x, edge_index)
                elif isinstance(model, GraphWaveNet):
                    # GraphWaveNet uses adjacency matrix directly
                    output = model(batch_x, adj_matrix)
                elif isinstance(model, (STAEformer, CDSReFormer)):
                    # Transformer-based models
                    output = model(batch_x, adj_matrix)
                elif isinstance(model, DCRNN):
                    # DCRNN uses adjacency matrix
                    output = model(batch_x, adj_matrix)
                else:
                    # Default case
                    output = model(batch_x)
                    
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                print(f"Batch shapes - X: {batch_x.shape}, Y: {batch_y.shape}")
                continue
            
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def evaluate(self, model, dataloader, mean, std, adj_matrix=None):
        """Evaluate model"""
        model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                try:
                    # Forward pass (handle different model types)
                    if isinstance(model, LSTM):
                        output = model(batch_x)
                    elif isinstance(model, (TGCN, STGCN)):
                        # These models need edge_index format
                        edge_index = self.adj_to_edge_index(adj_matrix)
                        if isinstance(model, TGCN):
                            # TGCN expects an extra dimension
                            output = model(batch_x.unsqueeze(-1), edge_index)
                        else:
                            # STGCN handles dimensions internally
                            output = model(batch_x, edge_index)
                    elif isinstance(model, GraphWaveNet):
                        # GraphWaveNet uses adjacency matrix directly
                        output = model(batch_x, adj_matrix)
                    elif isinstance(model, (STAEformer, CDSReFormer)):
                        # Transformer-based models
                        output = model(batch_x, adj_matrix)
                    elif isinstance(model, DCRNN):
                        # DCRNN uses adjacency matrix
                        output = model(batch_x, adj_matrix)
                    else:
                        # Default case
                        output = model(batch_x)
                    
                    # Denormalize predictions and targets
                    output_denorm = output.cpu().numpy() * std + mean
                    batch_y_denorm = batch_y.cpu().numpy() * std + mean
                    
                    all_preds.append(output_denorm)
                    all_true.append(batch_y_denorm)
                    
                except RuntimeError as e:
                    print(f"Error in evaluation batch: {e}")
                    continue
                
        if len(all_preds) == 0:
            return {'MAE': float('inf'), 'RMSE': float('inf'), 'SMAPE': float('inf')}
            
        all_preds = np.concatenate(all_preds, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_true)
        
        return metrics
    
    def adj_to_edge_index(self, adj_matrix):
        """Convert adjacency matrix to edge index for PyTorch Geometric"""
        edge_index = torch.nonzero(torch.tensor(adj_matrix), as_tuple=False).t()
        return edge_index.to(self.device)
    
    def train_model(self, model_name, data_dict, adj_matrix, mean, std, horizon_key):
        """Train a single model"""
        print(f"\nTraining {model_name} for {horizon_key}")
        
        # Get data
        train_x, train_y = data_dict[horizon_key]['train']
        val_x, val_y = data_dict[horizon_key]['val']
        test_x, test_y = data_dict[horizon_key]['test']
        
        print(f"Data shapes - Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}")
        print(f"Adjacency matrix shape: {adj_matrix.shape}")
        
        # Verify dimensions match
        num_nodes = train_x.shape[2]
        if adj_matrix.shape[0] != num_nodes:
            print(f"WARNING: Adjacency matrix has {adj_matrix.shape[0]} nodes but data has {num_nodes} nodes")
            # This shouldn't happen with the fixed preprocessing, but let's handle it
            if adj_matrix.shape[0] > num_nodes:
                print(f"Truncating adjacency matrix to match data dimensions")
                adj_matrix = adj_matrix[:num_nodes, :num_nodes]
            else:
                raise ValueError("Adjacency matrix has fewer nodes than data")
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(train_x), 
            torch.FloatTensor(train_y)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_x), 
            torch.FloatTensor(val_y)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(test_x), 
            torch.FloatTensor(test_y)
        )
        
        # Create dataloaders
        batch_size = self.config['training']['batch_size']
        
        # Important: Use drop_last=True for training to avoid batch size issues with GraphWaveNet
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of test batches: {len(test_loader)}")
        
        # Create model
        horizon = train_y.shape[1]
        model = self.create_model(model_name, num_nodes, horizon)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Setup training
        criterion = nn.MSELoss()
        learning_rate = self.config['training']['learning_rate']
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # Remove verbose parameter as it's not available in all PyTorch versions
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        max_epochs = self.config['training']['epochs']
        early_stopping_patience = self.config['training']['early_stopping_patience']
        
        print(f"Starting training for {max_epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        for epoch in range(max_epochs):
            # Train epoch
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer, adj_matrix)
            
            if train_loss == float('inf'):
                print(f"Training failed at epoch {epoch}")
                break
            
            # Evaluate on validation set
            val_metrics = self.evaluate(model, val_loader, mean, std, adj_matrix)
            
            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['MAE'])
            new_lr = optimizer.param_groups[0]['lr']
            
            # Print if learning rate changed
            if new_lr != current_lr:
                print(f"  → Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
            
            # Check for improvement
            if val_metrics['MAE'] < best_val_loss:
                best_val_loss = val_metrics['MAE']
                patience_counter = 0
                # Save best model
                checkpoint_path = f'models/best_{model_name}_{horizon_key}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_metrics['MAE'],
                    'val_rmse': val_metrics['RMSE'],
                    'val_smape': val_metrics['SMAPE'],
                    'config': self.config
                }, checkpoint_path)
                print(f"  → New best model saved! Val MAE: {val_metrics['MAE']:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
            # Progress logging
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                      f"Val MAE = {val_metrics['MAE']:.4f}, "
                      f"Val RMSE = {val_metrics['RMSE']:.4f}, "
                      f"Val SMAPE = {val_metrics['SMAPE']:.2f}%, "
                      f"LR = {new_lr:.6f}")
        
        # Load best model and evaluate on test set
        checkpoint_path = f'models/best_{model_name}_{horizon_key}.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from epoch {checkpoint['epoch']} "
                  f"with Val MAE = {checkpoint['val_mae']:.4f}")
        else:
            print(f"\nWarning: Best model checkpoint not found for {model_name}")
        
        # Final evaluation on test set
        test_metrics = self.evaluate(model, test_loader, mean, std, adj_matrix)
        
        print(f"\nFinal Test Results for {model_name} - {horizon_key}:")
        print(f"  MAE:  {test_metrics['MAE']:.4f}")
        print(f"  RMSE: {test_metrics['RMSE']:.4f}")
        print(f"  SMAPE: {test_metrics['SMAPE']:.2f}%")
        
        return test_metrics