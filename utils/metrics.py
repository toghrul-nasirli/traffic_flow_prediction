import numpy as np
import torch

def mae(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))

def rmse(pred, true):
    """Root Mean Square Error"""
    return np.sqrt(np.mean((pred - true) ** 2))

def mape(pred, true, epsilon=1e-8):
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((true - pred) / (true + epsilon))) * 100

def calculate_metrics(pred, true):
    """Calculate all metrics"""
    return {
        'MAE': mae(pred, true),
        'RMSE': rmse(pred, true),
        'MAPE': mape(pred, true)
    }