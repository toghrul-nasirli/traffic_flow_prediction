#main.py
import os
import pandas as pd
from utils.preprocessing import DataPreprocessor
from train import Trainer
from colorama import init, Fore, Style

init(autoreset=True)

def debug_data_dimensions(data_path):
    """Debug function to check data dimensions"""
    print("\n" + "="*60)
    print("DEBUGGING DATA DIMENSIONS")
    print("="*60)
    
    # Load and inspect raw data
    df = pd.read_csv(data_path)
    print(f"\nRaw data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check node IDs
    from_nodes = set(df['from'].unique())
    to_nodes = set(df['to'].unique())
    all_nodes = from_nodes | to_nodes
    
    print(f"\nUnique 'from' nodes: {len(from_nodes)}")
    print(f"Unique 'to' nodes: {len(to_nodes)}")
    print(f"Total unique nodes: {len(all_nodes)}")
    print(f"Node ID range: {min(all_nodes)} to {max(all_nodes)}")
    
    # Check if node IDs are consecutive
    expected_nodes = set(range(min(all_nodes), max(all_nodes) + 1))
    missing_nodes = expected_nodes - all_nodes
    if missing_nodes:
        print(f"\nWARNING: Missing node IDs: {sorted(missing_nodes)}")
    else:
        print(f"\nNode IDs are consecutive")
    
    # Create preprocessor and check dimensions
    preprocessor = DataPreprocessor(data_path)
    adj_matrix, node_mapping, node_list = preprocessor.create_adjacency_matrix(df)
    
    print(f"\nAdjacency matrix shape: {adj_matrix.shape}")
    print(f"Number of nodes in mapping: {len(node_mapping)}")
    
    return len(all_nodes), adj_matrix.shape[0]

def main():
    # Configuration
    data_path = 'data/raw/PEMS_Data.csv'
    models_to_train = ['LSTM', 'T-GCN', 'ST-GCN', 'DCRNN', 'GraphWaveNet', 'STAEformer', 'CDSReFormer']
    horizons_minutes = [15, 60]  # in minutes
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Debug data dimensions first
    actual_nodes, adj_nodes = debug_data_dimensions(data_path)
    
    # Assuming 5-minute intervals in the data
    horizons_steps = [h // 5 for h in horizons_minutes]  # [3, 12] steps
    
    # Prepare data
    preprocessor = DataPreprocessor(data_path)
    data_dict, adj_matrix, mean, std = preprocessor.prepare_data(
        seq_len=12,  # 1 hour of history
        horizons=horizons_steps,
        train_ratio=0.7,
        val_ratio=0.1
    )
    
    # Verify dimensions
    print("\n" + "="*60)
    print("VERIFYING DATA DIMENSIONS")
    print("="*60)
    
    for horizon_key, data_splits in data_dict.items():
        train_x, train_y = data_splits['train']
        print(f"\n{horizon_key}:")
        print(f"  Train X shape: {train_x.shape}")
        print(f"  Train Y shape: {train_y.shape}")
        print(f"  Number of nodes in data: {train_x.shape[2]}")
        print(f"  Adjacency matrix shape: {adj_matrix.shape}")
        
        if train_x.shape[2] != adj_matrix.shape[0]:
            print(f"  WARNING: Node count mismatch! Data has {train_x.shape[2]} nodes, adj matrix has {adj_matrix.shape[0]}")
    
    # Initialize trainer
    trainer = Trainer('configs/config.yaml')
    
    # Results storage
    results = []
    
    # Train each model for each horizon
    for model_name in models_to_train:
        for i, horizon_steps in enumerate(horizons_steps):
            horizon_key = f'horizon_{horizon_steps}'
            horizon_minutes = horizons_minutes[i]
            
            try:
                print(f"\n{'='*60}")
                print(f"Training {model_name} - {horizon_minutes} min horizon")
                print(f"{'='*60}")
                
                metrics = trainer.train_model(
                    model_name, 
                    data_dict, 
                    adj_matrix, 
                    mean, 
                    std, 
                    horizon_key
                )
                
                results.append({
                    'Model': model_name,
                    'Horizon': f'{horizon_minutes} min',
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE']
                })
                
                print(f"\n{model_name} - {horizon_minutes} min horizon:")
                print(f"MAE: {metrics['MAE']:.4f}")
                print(f"RMSE: {metrics['RMSE']:.4f}")
                print(f"MAPE: {metrics['MAPE']:.2f}%")
                
            except Exception as e:
                print(Fore.RED + f"Error training {model_name}: {str(e)}" + Style.RESET_ALL)
                import traceback
                traceback.print_exc()
                continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/model_comparison.csv', index=False)
    
    # Display summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    
if __name__ == "__main__":
    main()