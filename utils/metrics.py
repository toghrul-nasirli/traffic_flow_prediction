import numpy as np
import torch

def mae(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))

def rmse(pred, true):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((pred - true) ** 2))

def smape(pred, true, epsilon=1e-8):
    """Symmetric Mean Absolute Percentage Error"""
    return np.mean(2 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + epsilon)) * 100

def calculate_metrics(pred, true):
    """Calculate all metrics"""
    return {
        'MAE': mae(pred, true),
        'RMSE': rmse(pred, true),
        'SMAPE': smape(pred, true)
    }