import os
import yaml
import torch
import pandas as pd
from utils.data_loader import get_data_loaders
from train import Trainer


def main():
    # Paths
    config_path = 'configs/config.yaml'
    data_path = 'data/raw/PEMS_Data.csv'
    os.makedirs('results', exist_ok=True)

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    horizons = data_cfg.get('horizons', [3, 6, 12, 18, 24])  # Default to all horizons
    batch_size = config['training']['batch_size']

    # Prepare data loaders
    dataloaders, adj_matrix, mean, std = get_data_loaders(
        data_path=data_path,
        config_path=config_path
    )

    # Initialize trainer
    trainer = Trainer(config_path)

    # Evaluate each saved model
    results = []
    model_list = ['LSTM', 'T-GCN', 'ST-GCN', 'DCRNN', 'GraphWaveNet', 'STAEformer', 'CDSReFormer']

    for model_name in model_list:
        for horizon in horizons:
            horizon_key = f'horizon_{horizon}'
            horizon_minutes = horizon * 5
            ckpt = f'models/best_{model_name}_{horizon_key}.pth'
            if not os.path.exists(ckpt):
                print(f'Missing checkpoint for {model_name} {horizon_key}, skipping.')
                continue

            # Load checkpoint
            checkpoint = torch.load(ckpt, map_location=trainer.device, weights_only=False)

            # Build and load model
            num_nodes = adj_matrix.shape[0]
            model = trainer.create_model(model_name, num_nodes, horizon)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(trainer.device)

            # Evaluate on test set
            test_loader = dataloaders[horizon_key]['test']
            metrics = trainer.evaluate(model, test_loader, mean, std, adj_matrix)

            results.append({
                'Model': model_name,
                'Horizon': f'{horizon_minutes} min',
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'SMAPE': metrics['SMAPE']
            })
            print(f'{model_name} ({horizon_minutes} min) → MAE: {metrics["MAE"]:.4f}, '
                  f'RMSE: {metrics["RMSE"]:.4f}, SMAPE: {metrics["SMAPE"]:.2f}%')

    # Save evaluation summary
    eval_df = pd.DataFrame(results)
    eval_df.to_csv('results/evaluation_summary.csv', index=False)
    print('Saved evaluation summary to results/evaluation_summary.csv')


if __name__ == '__main__':
    main()
