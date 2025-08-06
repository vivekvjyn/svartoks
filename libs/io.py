import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def log_metrics(loss, mode, logger):
    """Log evaluation metrics.

    Args:
        loss (float): The loss.
        mode (str): The mode.
        logger (logging.Logger): The logger.
    """
    logger.message(f"\t{mode} Loss: {loss:.4f}")

def save_history(history, run_id):
    """Save evaluation logs to csv.

    Args:
        history (pd.DataFrame): The evaluation history.
        run_id (str): The run ID.
    """
    save_dir = f"results/{run_id}/history"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{run_id}-history.csv'
    history.to_csv(save_path, index=False)

def save_tokenizer(tokenizer, run_id, logger):
    """Save tokenizer to disk.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        run_id (str): The run ID.
        logger (logging.Logger): The logger.
    """
    save_dir = f'checkpoints/{run_id}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'tokenizer.pth')
    torch.save(tokenizer.state_dict(), save_path)
    logger.message(f"Tokenizer saved to {save_path}")

def curve(train_metric, val_metric, optima, run_id, metric, logger):
    """Plot evaluation metrics.

    Args:
        train_metric (list): The training metric.
        val_metric (list): The validation metric.
        optima (int): The optimal epoch.
        run_id (str): The run ID.
        metric (str): The metric name.
        kfold (bool): Whether to plot k-fold metrics.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_metric)) + 1, train_metric, label=f'Train {metric}', color='blue')
    plt.plot(np.arange(len(val_metric)) + 1, val_metric, label=f'Validation {metric}', color='green')
    plt.scatter(optima, val_metric[optima], color='red', marker='x', label='Optimum')
    plt.text(optima, val_metric[optima], f'{val_metric[optima]:.4f}', color='red', ha='center', va='bottom')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    save_dir = f'results/{run_id}/plots'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{metric}.png"
    plt.savefig(save_path)
    logger.message(f"Training curve saved to {save_path}")
    plt.close()

def train_curve(run_id, logger):
    """Plot final training results.

    Args:
        run_id (str): The ID of the run.
    """
    open_dir = f'results/{run_id}/history'
    history = pd.read_csv(f'{open_dir}/{run_id}-history.csv')
    curve(history['train loss'], history['val loss'], np.argmin(history['val loss']), run_id, 'loss', logger)
