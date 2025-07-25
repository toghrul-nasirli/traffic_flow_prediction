import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.preprocessing import DataPreprocessor


def get_data_loaders(data_path: str, config_path: str):
    """
    Load data, preprocess, and return PyTorch DataLoaders for each prediction horizon.

    Args:
        data_path (str): Path to the raw PEMS_CSV file.
        config_path (str): Path to the YAML config file.

    Returns:
        dataloaders (dict): Mapping horizon_key -> {train, val, test} DataLoaders.
        adj_matrix (np.ndarray): Adjacency matrix for the graph.
        mean (float): Mean used for normalization.
        std (float): Std deviation used for normalization.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    data_cfg = config['data']
    seq_len = data_cfg.get('sequence_length', 12)
    horizons = data_cfg.get('horizons', [])
    train_ratio = data_cfg.get('train_ratio', 0.7)
    val_ratio = data_cfg.get('val_ratio', 0.1)

    batch_size = config['training']['batch_size']

    # Prepare data
    preprocessor = DataPreprocessor(data_path)
    data_dict, adj_matrix, mean, std = preprocessor.prepare_data(
        seq_len=seq_len,
        horizons=horizons,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )

    # Build DataLoaders
    dataloaders = {}
    for horizon_key, splits in data_dict.items():
        train_x, train_y = splits['train']
        val_x, val_y = splits['val']
        test_x, test_y = splits['test']

        train_ds = TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
        val_ds = TensorDataset(torch.FloatTensor(val_x), torch.FloatTensor(val_y))
        test_ds = TensorDataset(torch.FloatTensor(test_x), torch.FloatTensor(test_y))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        dataloaders[horizon_key] = {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    return dataloaders, adj_matrix, mean, std
