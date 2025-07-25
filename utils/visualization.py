import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_metrics(results_csv: str = 'results/model_comparison.csv', save_dir: str = None):
    """
    Generate bar charts for MAE, RMSE, and MAPE across models and prediction horizons.

    Args:
        results_csv (str): Path to the CSV file with evaluation results.
        save_dir (str, optional): Directory to save the plots. If None, the plots are shown interactively.
    """
    df = pd.read_csv(results_csv)
    horizons = df['Horizon'].unique()
    metrics = ['MAE', 'RMSE', 'MAPE']

    for horizon in horizons:
        subset = df[df['Horizon'] == horizon]
        models = subset['Model'].tolist()

        for metric in metrics:
            values = subset[metric].tolist()
            plt.figure(figsize=(8, 4))
            plt.bar(models, values)
            plt.title(f'{metric} at {horizon}')
            plt.xlabel('Model')
            plt.ylabel(metric)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{metric.lower()}_{horizon.replace(' ', '_')}.png"
                plt.savefig(os.path.join(save_dir, filename))
                plt.close()
            else:
                plt.show()